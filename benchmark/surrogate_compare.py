"""
Copyright (C) 2018 Chenchao Shou
Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

This code compares modeling accuracy between RBF surrogate of ProSRS algorithm 
and different types of GP surrogates.
"""
from pyDOE import lhs
from surrogates.ProSRS_RBF import RBF_reg
from surrogates.GP import GP_reg
from benchmark_func import get_benchmark_func
import numpy as np
import os
from timeit import default_timer
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('classic')

def doe(n_samp,bd):
    '''
    Generate a design of experiments using Latin hypercube sampling.
    Input:
        n_samp: int, number of samples
        bd: list of tuples, bound
    Output:
        samp: 2d array, doe samples   
    '''
    dim = len(bd)
    unit_X = lhs(dim, samples=n_samp, criterion='maximin') # 2d array in unit cube
    samp = np.zeros_like(unit_X)
    for i in range(dim):
        samp[:,i] = unit_X[:,i]*(bd[i][1]-bd[i][0])+bd[i][0] # scale and shift
    return samp

def gen_train_data(func_name, n_samp):
    """
    Generate training data.
    Input:
        func_name: string, benchmark function name
        n_samp: int, number of samples
    Output:
        train_data: dictionary, training data X and Y
    """
    # get benchmark object
    func = get_benchmark_func(func_name)
    obj_f = func.f # noisy objective function
    func_bd = func.bounds # function domain
    
    # generate training data
    X_samp = doe(n_samp, func_bd)
    Y_samp = obj_f(X_samp) # 1d array
    
    train_data = {'X': X_samp, 'Y': Y_samp}
    
    return train_data
       
def run_test(func_name, train_data, model, model_param):
    """
    Run one accuracy test.
    Input:
        func_name: string, benchmark function name
        train_data: dictionary, training data X and Y
        model: string, surrogate model name
        model_param: dictionary, model parameters
    Output:
        true_rel_error: float, relative error in L2 norm over the function domain
        train_time: float, training time (sec)
    """
    # get benchmark object
    func = get_benchmark_func(func_name)
    true_f = func.tf # true objective function (underlying expectation function)
    func_bd = func.bounds # function domain
    dim = func.input_dim # dimension of the domain
    
    X_samp, Y_samp = train_data['X'], train_data['Y']
    
    # train model
    t1 = default_timer()
    if model == 'RBF':
        train_f,_,_,_ = RBF_reg(X_samp, Y_samp)
    elif model == 'GP':
        train_f,_ = GP_reg(X_samp, Y_samp, kernel=model_param['kernel'], 
                           kernel_param=model_param, n_restarts_optimizer=model_param['n_restart']) 
    else:
        raise ValueError('Unknown model type')    
    t2 = default_timer()
    train_time = t2-t1
    
    # compute relative error using Monte Carlo integration
    # [Note: the conventional quadrature-based integration (e.g., scipy.integrate.nquad)
    # is very slow for high dimensional functions, so we use Monte Carlo methods]
    #
    # here relative error is defined as the L2 norm of the function difference divided by
    # that of the true function: \sqrt{\int_D (train_f(x) - true_f(x))^2 dx}/\sqrt{\int_D (true_f(x))^2 dx},
    # where D is the function domain.
    rd_n = 10000*dim # number of Monte Carlo samples
    rd_pt = np.zeros((rd_n,dim)) # random samples
    for d,bd in enumerate(func_bd):
        rd_pt[:,d] = np.random.uniform(low=bd[0],high=bd[1],size=rd_n)
    true_rel_error = np.sqrt(np.mean((train_f(rd_pt)-true_f(rd_pt))**2)/np.mean(true_f(rd_pt)**2))
        
    return true_rel_error, train_time

def main():
    
    # parameters
    n_samp_arr = np.arange(10, 101, 10, dtype=int) # number of training data
    n_rpt = 10 # number of repeats for the experiment
    seed = 1 # random seed for reproducibility
    benchmark_func_list = ['ackley10', 'alpine1_10', 'griewank10', 'levy10', 'sum_power10', 
                           'sixhumpcamel', 'schaffer2', 'dropwave', 'goldstein', 'rastrigin2', 
                           'power_sum', 'hartmann6'] # list of benchmark functions
    model_list = ['RBF', 'GP', 'GP', 'GP'] # a list of regression models: ['RBF', 'GP']
    model_param_list = [{},
                        {'kernel': 'Matern', 'nu': 2.5, 'n_restart': 10},
                        {'kernel': 'Matern', 'nu': 1.5, 'n_restart': 10},
                        {'kernel': 'RBF', 'n_restart': 10}] # model parameters for each model in model_list
    outdir = 'surrogate_compare_result' # output directory
    
    # sanity check
    assert(len(model_list) == len(model_param_list) and seed >= 0 and n_rpt > 0 and np.min(n_samp_arr) > 0)
    # get constants
    n_func = len(benchmark_func_list)
    n_mod = len(model_list)
    n_lv = len(n_samp_arr)
    # make output directory
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    # generate model labels
    mod_lab_list = []
    for model, model_param in zip(model_list, model_param_list):
        if model == 'RBF':
            mod_lab = model  
        elif model == 'GP':
            if model_param['kernel'] == 'Matern':
                mod_lab = '%s-%s%g' % (model, model_param['kernel'], model_param['nu'])
            elif model_param['kernel'] == 'RBF':
                mod_lab = '%s-%s' % (model, model_param['kernel'])
        mod_lab_list.append(mod_lab)
    
    np.random.seed(int(seed))
    # initializations
    true_rel_error_arr = np.zeros((n_rpt,n_func,n_mod,n_lv))
    train_time_arr = np.zeros((n_rpt,n_func,n_mod,n_lv))
    # run accuracy tests
    for r in range(n_rpt):
        for i, benchmark_func in enumerate(benchmark_func_list):
            for k, n_samp in enumerate(n_samp_arr):
                # generate training data that are common for all different models to ensure fair comparison
                train_data = gen_train_data(benchmark_func, n_samp)
                for j, (model, model_param) in enumerate(zip(model_list, model_param_list)):
                    result = run_test(benchmark_func, train_data, model, model_param)
                    assert(len(result) == 2)
                    true_rel_error_arr[r,i,j,k] = result[0]
                    train_time_arr[r,i,j,k] = result[1]
                    print('Finish running accuracy test with rpt = %d/%d, func = %s (%d/%d), n_samp = %d (%d/%d), mod = %s (%d/%d), t_train = %.2e sec' % \
                          (r+1, n_rpt, benchmark_func, i+1, n_func, n_samp, k+1, n_lv, model, j+1, n_mod, train_time_arr[r,i,j,k]))
    # analyze results
    mean_true_rel_error = np.mean(true_rel_error_arr, axis=0)
    std_true_rel_error = np.std(true_rel_error_arr, axis=0, ddof=1)/np.sqrt(n_rpt) # std of the mean estimate  
    mean_train_time = np.mean(train_time_arr, axis=0)
    std_train_time = np.std(train_time_arr, axis=0, ddof=1)/np.sqrt(n_rpt) # std of the mean estimate
    
    # plot results
    for i, benchmark_func in enumerate(benchmark_func_list):
        # true_rel_error
        plt.figure(0)
        plt.clf()
        for j, mod_lab in enumerate(mod_lab_list):
            plt.errorbar(n_samp_arr, mean_true_rel_error[i,j], yerr=std_true_rel_error[i,j], fmt='-', label=mod_lab)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xticks(n_samp_arr)
        plt.xlim([n_samp_arr[0]-1, n_samp_arr[-1]+1])
        plt.xlabel('number of training data')
        plt.ylabel('relative error')
        plt.savefig(os.path.join(outdir, 'true_rel_error_%s.pdf' % benchmark_func))
        plt.close()
        # train_time
        plt.figure(0)
        plt.clf()
        for j, mod_lab in enumerate(mod_lab_list):
            plt.errorbar(n_samp_arr, mean_train_time[i,j], yerr=std_train_time[i,j], fmt='-', label=mod_lab)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xticks(n_samp_arr)
        plt.xlim([n_samp_arr[0]-1, n_samp_arr[-1]+1])
        plt.xlabel('number of training data')
        plt.ylabel('training time (sec)')
        plt.yscale('log')
        plt.savefig(os.path.join(outdir, 'train_time_%s.pdf' % benchmark_func))
        plt.close()
        # print progress
        print('Finish ploting results for %s (%d/%d)' % (benchmark_func, i+1, n_func))
    
    # save results
    np.savez(os.path.join(outdir, 'result.npz'), 
             # experiment condition parameters
             n_samp_arr=n_samp_arr, n_rpt=n_rpt, seed=seed, benchmark_func_list=benchmark_func_list,
             model_list=model_list, model_param_list=model_param_list, mod_lab_list=mod_lab_list,
             # results
             true_rel_error_arr=true_rel_error_arr, train_time_arr=train_time_arr, 
             mean_true_rel_error=mean_true_rel_error, std_true_rel_error=std_true_rel_error, 
             mean_train_time=mean_train_time, std_train_time=std_train_time)
    
    
if __name__ == '__main__':
    main()