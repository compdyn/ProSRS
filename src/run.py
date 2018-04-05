"""
This code runs ProSRS algorithm to solve optimization problems.
"""
import os
from ProSRS import run
from problem import gen_problem
from problem import test_func # TODO: import the problem you want to run (provide the name of the problem class)

def main():
    
    ##################### Parameters #########################
    
    # TODO: modify following parameters as needed
    opt_iter = 20 # number of optimization iterations (total number of iterations for ProSRS = opt_iter+1, where 1 accounts for initial DOE iteration)
    n_proc = 12 # number of processes for parallel optimization (>1, typically = number of processes of a compute node on a cluster)
    resume_opt_iter = None # optimization iteration index to resume from (= "opt_iter" value from last run). If None, then we start from scratch.
    seed = 1 # random seed for reproducibility
    verbose = False # If True, progress of every iteration will be displayed at terminal as well as saved to file "{outdir}/std_out_log_{prob_name}.txt"
    run_mode = 'local' # one of ['local','cluster']. 'local': run on a local machine; 'cluster': run on a cluster
    outdir = '../result' # output directory where results will be saved
    
    ##################### End of Parameters #########################
    
    # sanity check
    assert(run_mode in ['local','cluster'])
    if resume_opt_iter is not None:
        assert(0<=resume_opt_iter<opt_iter)
    # determine whether we initiate serial mode
    serial_mode = True if run_mode == 'local' else False
    # make output directory if it does not exist
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        
    # TODO: generate an optimization problem that you would like to run (change the following line as needed)
    prob = gen_problem(test_func()) # replace 'test_func' with the name of the imported problem class
    
    # run optimization algorithm
    best_loc, best_val = run(prob, n_proc, outdir, opt_iter=opt_iter, seed=seed, verbose=verbose, 
                             serial_mode=serial_mode, resume_opt_iter=resume_opt_iter)
    
    # print optimization results
    print('\nbest point found:')
    print(', '.join(['%s = %g' % (x,v) for x,v in zip(prob.x_var, best_loc)]))
    print('(noisy) function value corresponding to the best point:')
    print('%s = %g' % (prob.y_var, best_val))

if __name__ == '__main__':
    
    main()