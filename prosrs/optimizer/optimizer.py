"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Implement ProSRS algorithm.
"""
import numpy as np
import os, sys
from timeit import default_timer
from pyDOE import lhs
from scipy.spatial.distance import cdist
from pathos.multiprocessing import ProcessingPool as Pool
from ..utility.constants import STATE_NPZ_FILE_TEMP, STATE_PKL_FILE_TEMP
from ..utility.classes import std_out_logger, std_err_logger
from ..utility.functions import eval_func, put_back_box, scale_one_zero, scale_zero_one, eff_npt, boxify, domain_intersect
from .surrogate import RBF_reg


class Optimizer:
    """
    A class that handles optimization using ProSRS algorithm.
    """
    def __init___(self, prob, n_worker, n_proc_master=None, n_iter=None, n_iter_doe=None, 
                  n_restart=2, resume=False, seed=1, seed_func=None, out_dir='out'):
        """
        Constructor.
        
        Args:
            
            prob (`prosrs.Problem` type): Optimization problem.
            
            n_worker (int): Number of workers in the optimization.
                This also determines the number of proposed or evaluated
                points per iteration.
            
            n_proc_master (int or None, optional): Number of parallel processes (cores)
                that will be used in the master. If None, then all the available
                processes (cores) of the master will be used.
                
            n_iter (int or None, optional): Total number of iterations for the optimization.
                If ``int``, the optimization will terminate upon finishing running `n_iter` iterations. 
                If None, then we use number of restarts `n_restart` as the termination condition.
            
            n_iter_doe (int or None, optional): Number of iterations for DOE.
                If None, then we use default value.
                
            n_restart (int, optional): Total number of restarts for the optimization.
                This parameter takes effect only when `n_iter` is None, and 
                the optimization will terminate upon finishing restarting `n_restart` times.
            
            resume (bool, optional): Whether to resume from the last run.
                The information of the last run will be read from the directory `out_dir`.
            
            seed (int or None, optional): Random seed for the optimizer.
                If None, then we do not set random seed for the optimization.
            
            seed_func (callable or None, optional): User-specified function for setting the random seed for evaluations.
                If None, then we use ``numpy.random.seed(seed)`` method to set random seed.
                If callable, it is a function taking ``seed`` as an argument.
                Note: using ``numpy.random.seed`` may not always gaurantee
                the reproducibility. Here we provide an option for users to specify their own routines.
            
            out_dir (str, optional): Output directory.
                All the output files (e.g., optimization status file) will be saved to
                `out_dir` directory.
        """
        # class members (convention: variables wih prefix '_' are constant parameters during optimization).
        self._prob = prob # optimization problem
        self._dim = prob.dim # dimention of optimization problem.
        self._n_worker = n_worker # number of workers.
        self._n_proc_master = n_proc_master # number of parallel processes in the master.
        self._n_iter = n_iter # total number of iterations.
        self._n_restart = n_restart # total number of restarts.
        self._resume = resume # whether to resume from the last run.
        self._seed = seed # random seed for the optimizer.
        self._seed_func = seed_func # function for setting random seed for evaluations.
        self._out_dir = out_dir # output directory.
        self._n_cand_fact = 1000 # number of candidate = self._n_cand_fact * self._dim.
        self._wgt_pat_bd = [0.3, 1.] # the bound of the weight pattern in the SRS method.
        self._normalize_data = True # whether to normalize data when training RBF regression model.
        self._init_gamma = 0. # initial weight exponent in the SRS method (>=0). If zero, then disable weighting in RBF regression.
        self._delta_gamma = 2. # amount of decrease of weight exponent whenever failure occurs.
        self._init_sigma = 0.1 # initial sigma value in the SRS method (controls initial spread of Type II candidate points).
        self._max_n_reduce_sigma = 2 # max number of times of halving self.sigma before zooming in/restart. Critical sigma value = self._init_sigma * 0.5**self._max_n_reduce_sigma.
        self._rho = 0.4 # zoom-in factor. Must be in (0, 1).
        self._init_p = 1. # initial p value in the SRS method (controls initial proportion of Type I candidate points).
        self._init_beta = 0.02 # initial beta value (= initial probability of zooming out).
        self._min_beta = 0.01 # minimum beta value (= minimum probability of zooming out).
        self._alpha = 1. # parameter controlling decreasing rate of p value: p = p*n_{eff}**(-self._alpha/self.dim).
        self._lambda_range = [1e-7, 1e4] # range of regularization parameter in RBF regression, over which we determine regularization constant in L2 regularization.
        self._rbf_kernel = 'multiquadric' # RBF kernel. See `scipy.interpolate.Rbf <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.interpolate.Rbf.html>`.
        self._rbf_poly_deg = 0 # degree of RBF polynomial tail (either 0 or 1). If zero, then no polynomial tail.
        self._n_fold = 5 # number of folds for the cross validation in training a RBF model.
        self._resol = 0.01 # resolution parameter for restart.
        self._use_eff_n_samp = True # whether to use effective number of samples for the dynamics of p value.
        self._max_C_fail = max(2, int(np.ceil(self._dim/float(self._n_worker)))) # maximum number of consecutive failures before halving self.sigma value.
        self._n_iter_doe = int(np.ceil(3/float(self._n_worker))) if n_iter_doe is None else n_iter_doe # number of iterations for DOE.
        self._n_iter_doe = min(self._n_iter, self._n_iter_doe) if self._n_iter is not None else self._n_iter_doe # adjusted for the fact that self._n_iter_doe <= self._n_iter.
        self._n_doe_samp = self._n_iter_doe * self._n_worker # number of DOE samples
        self._n_cand = self._n_cand_fact * self._dim # number of candidate points in SRS method
        self._state_npz_file = os.path.join(self._out_dir, STATE_NPZ_FILE_TEMP % self._prob.name) # file that saves optimizer state (useful for resume)
        self._state_pkl_file = os.path.join(self._out_dir, STATE_PKL_FILE_TEMP % self._prob.name) # file that saves optimizer state (useful for resume)
        self._pool_rbf = Pool() if self._n_proc_master is None else Pool(nodes=self._n_proc_master)
        
        # sanity check
        # TODO: check the type of `self._prob`
        assert(type(self._n_worker) is int and self._n_worker > 0)
        assert(type(self._n_proc_master) is int and self._n_proc_master > 0)
        assert(0 <= self._wgt_pat_bd[0] <= self._wgt_pat_bd[1] <= 1 and len(self._wgt_pat_bd) == 2)
        assert(self._delta_gamma >= 0)
        assert(type(self._max_n_reduce_sigma) is int and self._max_n_reduce_sigma >= 0)
        assert(0 < self._rho < 1)
        assert(0 <= self._init_p <= 1)
        assert(0 <= self._min_beta <= self._init_beta <= 1)
        assert(self._alpha > 0)
        assert(0 < self._lambda_range[0] <= self._lambda_range[1] and len(self._lambda_range) == 2)
        assert(type(self._rbf_poly_deg) is int and self._rbf_poly_deg in [0, 1])
        assert(type(self._n_fold) is int and self._n_fold > 1)
        assert(0 < self._resol < 1)
        assert(type(self._max_C_fail) is int and self._max_C_fail > 0)
        assert(type(self._n_iter_doe) is int and self._n_iter_doe > 0)
        assert(type(self._n_cand) is int and self._n_cand >= self._n_worker)
        if type(self._n_iter) is int:
            assert(self._n_iter > 0)
        else:
            assert(self._n_iter is None), 'n_iter is either an integer or None.'
        assert(type(self._n_restart) is int and self._n_restart > 0)
        
        # create output directory, if not existent
        if not os.path.isdir(self._out_dir):
            os.makedirs(self._out_dir)
        
        # initialize the state of the optimizer
        if not self._resume:
            self.i_iter = 0 # iteration index (how many iterations have been run)
            self.i_restart = 0 # restart index (how many restarts have been initiated)
            self.doe_samp = self.doe() # DOE samples
            self.i_iter_doe = 0 # iteration index during DOE phase (how many DOE iterations have been run in current restart cycle)
            self.t_build_arr = np.zeros(0) # time of building a RBF model for each iteration. If an iteration is DOE, = 0.  
            self.t_srs_arr = np.zeros(0) # time of running SRS method for each iteration. If an iteration is DOE, = 0. 
            self.t_prop_arr = np.zeros(0) # time of proposing new points for each iteration.
            self.t_eval_arr = np.zeros(0) # time of evaluating proposed points for each iteration.
            self.t_update_arr = np.zeros(0) # time of updating the optimizer state for each iteration.
            self.gSRS_pct_arr = np.zeros(0) # pertentage of global SRS (= percentage of Type I candidates = floor(10*p_val)/10.) for each iteration.
            self.zoom_lv_arr = np.zeros(0) # zoom level at the time of proposing new points for each iteration.
            self.x_all = np.zeros((0, self._dim)) # all the evaluated (proposed) points so far in a tree.
            self.y_all = np.zeros(0) # (noisy) y values of `self.x_all`.
            self.seed_all = np.zeros(0) # random seeds for points in `self.x_all`.
            self.best_x = np.ones(self._dim)*np.nan # best point so far.
            self.best_y = np.nan # (noisy) y value of the best point `self.best_x`.
            np.random.seed(self._seed) # set random seed.
            self.random_state = np.random.get_state() # FIXME: check necessity of passing random state within the class.
            self.zoom_lv = 0 # zoom level (zero-based).
            self.act_node_ix = 0 # index of the activate node for the zoom level `self.zoom_lv` (zero-based).
            self.srs_wgt_pat = np.linspace(self._wgt_pat_bd[0], self._wgt_pat_bd[1], self._n_worker) # weight pattern in the SRS method.
            self.tree = self.init_tree() # initialize optimization tree    
        else:
            # load optimizer state from the last run
            self.load_state()
        
    
    def __str__(self, verbose_level=1):
        """
        Display optimizer status.
        
        Args:
            
            verbose_level (int, optional): Verbose level. The larger `verbose` is,
                the more detailed information will be displayed.
        """
    
    def run(self, verbose=True, std_out_file=None, std_err_file=None):
        """
        Run ProSRS algorithm.
        
        Args:
                
            verbose (bool, optional): Whether to verbose while running the algorithm.
            
            std_out_file (str or None, optional): Standard output file path.
                If ``str``, then standard outputs will be directed to the file `std_out_file`.
                If None, then standard outputs will not be saved to a file.
            
            std_err_file (str or None, optional): Standard error file path.
                If ``str``, then standard errors will be directed to the file `std_err_file`.
                If None, then standard errors will not be saved to a file.
            
        """            
        # Log standard outputs and standard errors.
        # Here we write to a new file if we do not resume. Otherwise, we append to the old file.
        if std_out_file is not None:
            if not self._resume:
                if os.path.isfile(std_out_file):
                    os.remove(std_out_file)
            sys.stdout = std_out_logger(std_out_file)    
        if std_err_file is not None:
            if not self._resume:
                if os.path.isfile(std_err_file):
                    os.remove(std_err_file)
            sys.stderr = std_err_logger(std_err_file)
        
        # main loop
        while not self.is_done():
            # propose new points
            new_pt = self.propose(verbose=verbose)
            
            # evaluate proposed points
            new_val = self.eval_pt(new_pt, verbose=verbose)
            
            # update optimizer state with the new evaluations
            self.update(new_pt, new_val, verbose=verbose)
            
            if verbose:
                # TODO: display some summary of results
                pass
            
            # flush standard outputs and standard errors to files
            if std_out_file is not None:
                sys.stdout.terminal.flush()
                sys.stdout.log.flush()
            if std_err_file is not None:
                sys.stderr.terminal.flush()
                sys.stderr.log.flush()
        
        
#        ########################### Main program ############################
#        
#        if resume_iter is None:
#            
#            pass
            
#            ########### Initial sampling (DOE) ################ 
#            
#            np.random.seed(int(seed))
#            
#            t1 = default_timer()
#            
#            n_init_samp = n_proc*init_iter
#            X_all = doe(n_init_samp,func_bd)
#                
#            t2 = default_timer()
#            
#            t_doe = t2-t1
#            
#            ########### Evaluate DOE ################
#            
#            t_eval = np.zeros(init_iter)
#            
#            seed_base = seed+1
#            seed_flat_arr = np.array([])
#            
#            # initialization
#            Y_all = np.zeros(n_init_samp)
#            for k in range(init_iter):
#                ix1,ix2 = k*n_proc,(k+1)*n_proc
#                X = X_all[ix1:ix2]
#                seed_arr = seed_base+np.arange(n_proc)
#                seed_flat_arr = np.append(seed_flat_arr,seed_arr) # update seed_flat_arr for sanity check
#                seed_base += n_proc
#                out_list = [os.path.join(outdir, RESULT_SAMP_FILE_TEMP % (prob_name,k+1,n+1)) \
#                            for n in range(n_proc)]
#                t1 = default_timer()
#                # evaluate function
#                Y = func_eval(obj_func,X,seed_arr,out_list,pool_eval,comm)
#                t2 = default_timer()
#                t_eval[k] = t2-t1
#                # save Y
#                Y_all[ix1:ix2] = Y
#                
#                # save to netcdf file
#                if save_samp:
#                    out_file = os.path.join(outdir, RESULT_GENERAL_SAMP_FILE_TEMP % (prob_name,k+1))
#                    if os.path.isfile(out_file):
#                        os.remove(out_file) # remove file if exists
#                    with netcdf.netcdf_file(out_file,'w') as f:
#                        f.createDimension('var',dim)
#                        f.createDimension('samp',n_proc)
#                        nc_samp = f.createVariable('x','d',('samp','var'))
#                        nc_samp[:] = X
#                        nc_samp = f.createVariable('y','d',('samp',))
#                        nc_samp[:] = Y
#                        nc_seed = f.createVariable('seed','i',('samp',))
#                        nc_seed[:] = seed_arr
#                        nc_wall_time = f.createVariable('tw_eval','d',())
#                        nc_wall_time.assignValue(t_eval[k])
#                        nc_wall_time = f.createVariable('tw_doe','d',())
#                        nc_wall_time.assignValue(t_doe)
#            
#            # sanity check
#            assert(np.all(seed_flat_arr == np.arange(np.min(seed_flat_arr),np.max(seed_flat_arr)+1)))
#        
#        
#            ######### Initializations for optimization ########
#                
#            # count number of full restart
#            n_full_restart = 0
#            # full restart flag
#            full_restart = False
#            # doe samples for restart (useful for resume)
#            X_samp_restart = np.zeros((0,dim))
        
#        else:
#            
#            t1 = default_timer()
#            
#            resume_opt_iter = resume_iter-init_iter
#            resume_opt_iter = int(resume_opt_iter) # convert to integer type if not
#            # remove lock file if exists
#            result_npz_lock_file = result_npz_file+'.lock'
#            if os.path.isfile(result_npz_lock_file):
#                os.remove(result_npz_lock_file)
#            # read experiment conditions from previous trials
#            data = np.load(result_npz_file)
#            assert(resume_opt_iter==data['opt_iter']), \
#            'Please resume from where it ended last time by setting resume_iter = %d' % (data['opt_iter']+init_iter)
#    
#            # sanity check for consistency of experiment conditions
#            assert(init_iter==data['init_iter'] and n_proc==data['n_proc'] and seed==data['seed']
#                   and np.all(func_bd==data['func_bd']) and n_core_node==data['n_core_node']
#                   and n_cand_fact==data['n_cand_fact'] and alpha==data['alpha'] 
#                   and normalize_data==data['normalize_data'] and init_gamma==data['init_gamma']
#                   and delta_gamma==data['delta_gamma'] and max_n_reduce_sigma==data['max_n_reduce_sigma']
#                   and rho==data['rho'] and np.all(init_sigma==data['init_sigma'])
#                   and init_p==data['init_p'] and np.all(wgt_pat_bd==data['wgt_pat_bd'])
#                   and np.all(lambda_range==data['lambda_range'])
#                   and n_fold==data['n_fold'] and resol==data['resol'] and min_beta==data['min_beta']
#                   and init_beta==data['init_beta'] and max_C_fail==data['max_C_fail']
#                   and use_eff_n_samp==data['use_eff_n_samp'] and rbf_kernel==data['rbf_kernel'])
#            
#            # read status variables from previous experiment
#            full_restart = data['full_restart'].item(0)
#            X_samp_restart = data['X_samp_restart']
#            zoom_lv = data['zoom_lv'].item(0)
#            act_node_ix = data['act_node_ix'].item(0)
#            X_all = data['X_all']
#            Y_all = data['Y_all']
#            tree = data['tree'].item(0)
#            n_full_restart = data['n_full_restart'].item(0)
#            seed_base = data['seed_base'].item(0)
#            seed_flat_arr = data['seed_flat_arr']
#            
#            # read optimization results from previous experiment
#            t_build_arr[:resume_opt_iter] = data['t_build_arr']
#            t_prop_arr[:resume_opt_iter] = data['t_prop_arr']
#            t_srs_arr[:resume_opt_iter] = data['t_srs_arr']
#            t_eval_arr[:resume_opt_iter] = data['t_eval_arr']
#            t_update_arr[:resume_opt_iter] = data['t_update_arr']
#            gSRS_pct_arr[:resume_opt_iter] = data['gSRS_pct_arr']
#            zoom_lv_arr[:resume_opt_iter] = data['zoom_lv_arr']
#            
#            # load random state
#            result_pkl_lock_file = result_pkl_file+'.lock'
#            if os.path.isfile(result_pkl_lock_file):
#                os.remove(result_pkl_lock_file)
#            with open(result_pkl_file, 'rb') as f:
#                np.random.set_state(pickle.load(f))
#                
#            t2 = default_timer()
#            t_resume = t2-t1
#            if verbose:
#                print('\ntime to prepare resume = %.2e sec' % t_resume)
        
#        ########### Optimization iterations ################
#        
#        start_iter = 0 if resume_iter is None else resume_opt_iter
#        
#        for k in range(start_iter,opt_iter):
#            
#            if verbose:
#                print('\nStart optimization iteration = %d/%d' % (k+1,opt_iter))
#            
#            tt1 = default_timer()
#            
#            
#                    
#            if not full_restart:
#                    
#                pass
#        
#            else:
                
#                # i.e. do full restart
#                if n_full_restart == 0:
#                    n_init_samp = n_proc*init_iter
#                    X_samp_restart = doe(n_init_samp,func_bd)
#                
#                ix1, ix2 = n_full_restart*n_proc, (n_full_restart+1)*n_proc
#                prop_pt_arr = X_samp_restart[ix1:ix2]
#                
#                gSRS_pct = np.nan
#                pass
            
#            tm1 = default_timer()        
#            # save variables
#            gSRS_pct_arr[k] = gSRS_pct # save gSRS_pct
#            zoom_lv_arr[k] = zoom_lv # save zoom_lv
#            
#            ############ Evaluate proposed points #############
#            
#            seed_arr = seed_base+np.arange(n_proc)
#            seed_flat_arr = np.append(seed_flat_arr,seed_arr) # update seed_flat_arr for sanity check
#            seed_base += n_proc
#            out_list = [os.path.join(outdir, RESULT_SAMP_FILE_TEMP % (prob_name,k+init_iter+1,n+1)) \
#                        for n in range(n_proc)]
#            t1 = default_timer()
#            Y_prop_pt = func_eval(obj_func,prop_pt_arr,seed_arr,out_list,pool_eval,comm)
#            t2 = default_timer()
#            Y_prop_pt = np.array(Y_prop_pt)
#            assert(len(prop_pt_arr) == len(Y_prop_pt) == n_proc)
#            
#            assert(np.all(seed_flat_arr == np.arange(np.min(seed_flat_arr),np.max(seed_flat_arr)+1)))
#            
#            t_eval_arr[k] = t2-t1
#            
#            if verbose:
#                print('time to evaluate points = %.2e sec' % t_eval_arr[k])
            
#            # update node
#            n_X_all = len(X_all)
#            act_node['ix'] = np.append(act_node['ix'], np.arange(n_X_all,n_X_all+n_proc,dtype=int))
#            # update samples
#            X_all = np.vstack((X_all,prop_pt_arr))
#            Y_all = np.append(Y_all,Y_prop_pt)
#                
#            # update state of the current node
#            if not full_restart:
#                    
#                if n_proc > 1 or (n_proc == 1 and srs_wgt_pat[0] == wgt_pat_bd[0]):
#                    if p_val >= 0.1:
#                        # compute p_val
#                        if use_eff_n_samp:
#                            eff_n = eff_npt(X_all[act_node['ix']],act_node['domain'])
#                        else:
#                            eff_n = len(X_all[act_node['ix']])
#                        p_val = p_val*eff_n**(-alpha/float(dim))
#                        
#                    if gSRS_pct == 0: # i.e. pure local SRS
#                        best_Y_prev = np.min(y_node)
#                        best_Y_new = np.min(Y_prop_pt) # minimum of Y values of newly proposed points
#                        if best_Y_prev <= best_Y_new: # failure                
#                            n_fail += 1 # count failure
#                        else:
#                            n_fail = 0
#                        if n_fail == max_C_fail:
#                            n_fail = 0
#                            gamma += delta_gamma
#                            n_reduce_sigma += 1 # update counter
#                    act_node['state']['p'] = p_val
#                    act_node['state']['Cr'] = n_reduce_sigma
#                    act_node['state']['Cf'] = n_fail
#                    act_node['state']['gamma'] = gamma
            
#            # save to netcdf file
#            t1 = default_timer()
#            if save_samp:
#                out_file = os.path.join(outdir, RESULT_GENERAL_SAMP_FILE_TEMP % (prob_name,k+init_iter+1))
#                if os.path.isfile(out_file):
#                    os.remove(out_file) # remove file if exists
#                    
#                with netcdf.netcdf_file(out_file,'w') as f:
#                    f.createDimension('var',dim)
#                    f.createDimension('samp',n_proc)
#                    nc_samp = f.createVariable('x','d',('samp','var'))
#                    nc_samp[:] = prop_pt_arr
#                    nc_samp = f.createVariable('y','d',('samp',))
#                    nc_samp[:] = Y_prop_pt
#                    nc_seed = f.createVariable('seed','i',('samp',))
#                    nc_seed[:] = seed_arr
#                    nc_wall_time = f.createVariable('tw_eval','d',())
#                    nc_wall_time.assignValue(t_eval_arr[k])
#            
#            t2 = default_timer()
#            t_save = t2-t1
#            
#            if verbose:
#                print('time to save samples = %.2e sec' % t_save)
            
#            ############# Prepare for next iteration #############
#            
#            tp1 = default_timer()
#               
#            if not full_restart:
#                
#                pass
                
#                if n_reduce_sigma > max_n_reduce_sigma:
#                    # then we either restart or zoom-in
#                    Y_fit = rbf_mod(X_all[act_node['ix']])
#                    min_ix = np.argmin(Y_fit)
#                    x_star = X_all[act_node['ix']][min_ix]
#                    # suppose we're going to zoom in
#                    child_node_ix = get_child_node(x_star,zoom_lv,tree)
#                    if child_node_ix is None:
#                        # then we create a new child (if zoom in)
#                        fit_lb, fit_ub = zip(*fit_bd)
#                        blen = np.array(fit_ub)-np.array(fit_lb) # bound length for each dimension
#                        assert(np.min(blen)>0)
#                        fit_lb = np.maximum(x_star-rho/2.0*blen,fit_lb)
#                        fit_ub = np.minimum(x_star+rho/2.0*blen,fit_ub)
#                        fit_bd = list(zip(fit_lb,fit_ub)) # the list function is used to ensure compatibility of python3
#                        child_node = {'ix': np.nonzero(get_box_samp(X_all,fit_bd)[0])[0], 
#                                      'domain': fit_bd,
#                                      'parent_ix': act_node_ix,
#                                      'beta': init_beta,
#                                      'state': init_node_state()}
#                    else:
#                        # then we activate an existing child node (if zoom in)
#                        child_node = tree[zoom_lv+1][child_node_ix]
#                        child_node['ix'] = np.nonzero(get_box_samp(X_all,child_node['domain'])[0])[0]
#                    n_samp = len(child_node['ix'])
#                    fit_lb, fit_ub = zip(*child_node['domain'])
#                    blen = np.array(fit_ub)-np.array(fit_lb) # bound length for each dimension
#                    assert(np.min(blen)>0)
#                    
#                    if np.all(blen*n_samp**(-1./dim)<func_blen*resol): # resolution condition
#                        # then we restart
#                        if verbose:
#                            print('Restart for the next iteration!')                        
#                        full_restart = True
#                        zoom_lv = 0
#                        act_node_ix = 0
#                        X_all = np.zeros((0,dim))
#                        Y_all = np.zeros(0)
#                        tree = {zoom_lv: [{'ix': np.arange(0,dtype=int), # indice of samples for the node (with respect to X_all and Y_all)
#                                          'domain': func_bd, # domain of the node
#                                          'parent_ix': None, # parent node index for the upper zoom level (zero-based). If None, there's no parent
#                                          'beta': init_beta, # zoom-out probability
#                                          'state': init_node_state() # state of the node
#                                          }]}
#                    else:
#                        # then we zoom in
#                        act_node['state'] = init_node_state() # reset the state of the current node
#                        zoom_lv += 1
#                        
#                        if child_node_ix is None:
#                            # then we create a new child
#                            if zoom_lv not in tree.keys():
#                                act_node_ix = 0
#                                tree[zoom_lv] = [child_node]    
#                            else:
#                                act_node_ix = len(tree[zoom_lv])
#                                tree[zoom_lv].append(child_node)
#                                
#                            if verbose:
#                                print('Zoom in (create a new child node)!')
#                                
#                        else:
#                            # then activate existing child node
#                            act_node_ix = child_node_ix
#                            # reduce zoom-out probability
#                            child_node['beta'] = max(min_beta,child_node['beta']/2.)
#                            
#                            if verbose:
#                                print('Zoom in (activate an existing child node)!')
#                                
#                if n_proc > 1 or (n_proc == 1 and srs_wgt_pat[0] == wgt_pat_bd[0]):            
#                    if np.random.uniform() < tree[zoom_lv][act_node_ix]['beta'] and zoom_lv > 0 and not full_restart:
#                        # then we zoom out
#                        child_node = tree[zoom_lv][act_node_ix]
#                        act_node_ix = child_node['parent_ix']
#                        zoom_lv -= 1
#                        assert(act_node_ix is not None)
#                        # check that the node after zooming out will contain the current node
#                        assert(intsect_bd(tree[zoom_lv][act_node_ix]['domain'],child_node['domain']) == child_node['domain']) 
#                        
#                        if verbose:
#                            print('Zoom out!')
#            
#            else:
#                # i.e., restart
#                n_full_restart += 1          
#                if n_full_restart == init_iter:
#                    full_restart = False
#                    n_full_restart = 0
#            
#            tm2 = default_timer()
#            t_misc = tm2-tm1-t_eval_arr[k]-t_save
#            
#            if verbose:
#                print('time for miscellaneous tasks (saving variables, etc.) = %.2e sec' % t_misc)
#            
#            # find time to prepare for next iteration
#            tp2 = default_timer()
#            t_update_arr[k] = tp2-tp1
#            
#            # find the time to run optimization algortithm (excluding time to evaluate and save samples)
#            tt2 = default_timer()
#            t_alg = tt2-tt1-t_eval_arr[k]-t_save
#            
#            if verbose:
#                print('time to run optimization algorithm = %.2e sec' % t_alg)
                    
#            ########### Save results ###############
#            
#            t1 = default_timer()
#            
#            # save random state to pickle file for possible resume
#            if os.path.isfile(result_pkl_file):
#                os.remove(result_pkl_file) # remove file if exists 
#            with open(result_pkl_file,'wb') as f:
#                pickle.dump(np.random.get_state(),f)
#                
#            # save to npz file
#            temp_result_npz_file = result_npz_file+TEMP_RESULT_NPZ_FILE_SUFFIX # first save to temporary file to avoid loss of data upon termination
#            np.savez(temp_result_npz_file,
#                     # experiment condition parameters
#                     init_iter=init_iter,opt_iter=k+1,n_proc=n_proc,n_core_node=n_core_node,seed=seed,outdir=outdir,save_samp=save_samp,verbose=verbose,
#                     n_cand_fact=n_cand_fact,use_eff_n_samp=use_eff_n_samp,init_beta=init_beta,
#                     normalize_data=normalize_data,init_gamma=init_gamma,delta_gamma=delta_gamma,
#                     max_n_reduce_sigma=max_n_reduce_sigma,rho=rho,init_sigma=init_sigma,
#                     init_p=init_p,alpha=alpha,wgt_pat_bd=wgt_pat_bd,
#                     lambda_range=lambda_range,
#                     n_fold=n_fold,resol=resol,min_beta=min_beta,rbf_kernel=rbf_kernel,
#                     func_bd=func_bd,max_C_fail=max_C_fail,resume_iter=resume_iter,n_iter=n_iter,
#                     # optimization results
#                     t_build_arr=t_build_arr[:k+1],t_prop_arr=t_prop_arr[:k+1],
#                     t_srs_arr=t_srs_arr[:k+1],
#                     t_eval_arr=t_eval_arr[:k+1],t_update_arr=t_update_arr[:k+1],
#                     gSRS_pct_arr=gSRS_pct_arr[:k+1],zoom_lv_arr=zoom_lv_arr[:k+1],
#                     # state variables
#                     full_restart=full_restart,zoom_lv=zoom_lv,act_node_ix=act_node_ix,
#                     X_all=X_all,Y_all=Y_all,tree=tree,n_full_restart=n_full_restart,
#                     seed_base=seed_base,seed_flat_arr=seed_flat_arr,
#                     X_samp_restart=X_samp_restart)
#            
#            shutil.copy2(temp_result_npz_file,result_npz_file) # overwrite the original one
#            os.remove(temp_result_npz_file) # remove temporary file
#            
#            t2 = default_timer()
#            
#            if verbose:
#                print('time to save results = %.2e sec' % (t2-t1))  
#           
#            # save terminal output to file
#            if std_out_file is not None:
#                sys.stdout.terminal.flush()
#                sys.stdout.log.flush()
#            if std_err_file is not None:
#                sys.stderr.terminal.flush()
#                sys.stderr.log.flush()
        
        
    def doe(self, criterion='maximin'):
        """
        Design of experiments using Latin Hypercube Sampling (LHS).
        
        Args:
            
            criterion (str, optional): Sampling criterion for LHS.
                For details, see `pyDOE documentation <https://pythonhosted.org/pyDOE/randomized.html>`.
                
        Returns:
            
            samp (2d array): LHS samples. Each row is one sample.
        """
        unit_X = lhs(self._dim, samples=self._n_doe_samp, criterion=criterion) # unit_X: 2d array in unit cube
        samp = np.zeros_like(unit_X)
        for i in range(self._dim):
            samp[:, i] = unit_X[:, i]*(self._prob.domain[i][1]-self._prob.domain[i][0])+self._prob.domain[i][0] # scale and shift
            
        return samp
    
    
    def init_tree(self):
        """
        Initialize an optimization tree.
        
        Returns:
            
            tree (dict): Initial tree.
        """
        tree = {self.zoom_lv: [{'ix': np.arange(self._n_doe_samp, dtype=int), # indice of data for the tree node (w.r.t. `self.x_all` or `self.y_all`).
                                'domain': self._prob.domain, # domain of the tree node.
                                'parent_ix': None, # parent node index for the upper zoom level (zero-based). If None, there's no parent.
                                'beta': self._init_beta, # zoom-out probability.
                                'state': self.init_node_state() # state of the tree node.
                                }]}
        return tree
    
        
    def init_node_state(self):
        """
        Initialize the state of a node of the optimization tree.
        
        Returns:
            
            state (dict): Values of state variables.
        """
        
        state = {'p': self._init_p, # p value in the SRS method (controls proportion of Type I candidate points).
                 'Cr': 0, # counter that counts number of times of reducing the sigma value of local SRS.
                 'Cf': 0, # counter that counts number of consecutive failures.
                 'gamma': self._init_gamma # weight exponent parameter of weighted RBF
                 }
        
        return state
    
    
    def propose(self, verbose=True):
        """
        Propose new points for the next iteration.
        
        Args:
            
            verbose (bool, optional): Whether to verbose about proposing new points.
        
        Returns:
            
            new_pt (2d array): Proposed new points. Each row is one point.
        """
        # FIXME: check verbose
        np.random.set_state(self.random_state)
        
        tt1 = default_timer()
        
        if self.i_iter_doe < self._n_iter_doe:
            # i.e., current iteration is in DOE phase
            new_pt = self.doe_samp[self.i_iter_doe*self._n_worker:(self.i_iter_doe+1)*self._n_worker]
            # the following variables will be saved in the `self.update` method
            # so we need to set them appropriately
            self.gSRS_pct = np.nan
            self.t_build = 0.
            self.t_srs = 0.
        else:
            # i.e., current iteration is in true optimization phase
            
            ########### Get activated node ##############
            
            self.act_node = self.tree[self.zoom_lv][self.act_node_ix]
            self.x_node = self.x_all[self.act_node['ix']] # get X data of the node
            self.y_node = self.y_all[self.act_node['ix']] # get Y data of the node
            self.p_val = self.act_node['state']['p']
            self.n_reduce_sigma = self.act_node['state']['Cr']
            self.n_fail = self.act_node['state']['Cf']
            self.gamma = self.act_node['state']['gamma']
            self.gSRS_pct = np.floor(10*self.p_val)/10. # pertentage of global SRS (= percentage of Type I candidates)
            self.sigma = self._init_sigma*0.5**self.n_reduce_sigma
            
            ########### Build RBF surrogate model ##############
                
            t1 = default_timer()
            
            # FIXME: we may not need `opt_sm` eventually
            self.rbf_mod, opt_sm, _, _ = RBF_reg(self.x_node, self.y_node, self._lambda_range,
                                            normalize_data=self._normalize_data, wgt_expon=self.gamma,
                                            n_fold=self._n_fold, kernel=self._rbf_kernel, pool=self._pool_rbf,
                                            poly_deg=self._rbf_poly_deg)                
            t2 = default_timer()
            self.t_build = t2-t1
            
            if verbose:
                print('time to build RBF surrogate = %.2e sec' % self.t_build)
                
            ########### Propose new points using SRS method ############## 
            
            t1 = default_timer()
            
            new_pt = self.SRS()
            
            t2 = default_timer()
            self.t_srs = t2-t1
            
            if verbose:
                print('act_node_ix = %d' % self.act_node_ix)
                print('gamma = %g' % self.gamma)
                print('opt_sm = %.1e' % opt_sm)
                print('gSRS_pct = %g' % self.gSRS_pct)
                print('n_fail = %d' % self.n_fail)
                print('n_reduce_sigma = %d' % self.n_reduce_sigma)
                print('sigma = %g' % self.sigma)
                print('zoom_out_prob = %g' % self.act_node['beta'])
                print('zoom_lv = %d' % self.zoom_lv)
                print('node_domain =')
                print(self.act_node['domain'])
            
        tt2 = default_timer()
        self.t_prop = tt2-tt1
        
        if verbose:
            print('time to propose new points = %.2e sec' % self.t_prop)
            
        self.random_state = np.random.get_state()
        
        return new_pt
        
    
    def SRS(self):
        """
        Propose new points using SRS method.
        
        Returns:
            
            new_pt (2d array): Proposed points.
        """        
        # generate candidate points
        if self.gSRS_pct == 1:            
            # generate candidate points uniformly (global SRS)
            cand_pt = np.zeros((self._n_cand, self._dim))
            for d, bd in enumerate(self.act_node['domain']):
                cand_pt[:, d] = np.random.uniform(low=bd[0], high=bd[1], size=self._n_cand)
        else:
            n_cand_gSRS = int(np.round(self._n_cand*self.gSRS_pct)) # number of candidate points for global SRS
            n_cand_lSRS = self._n_cand-n_cand_gSRS # number of candidate points for local SRS
            assert(n_cand_lSRS > 0) # sanity check
            # generate candidate points uniformly (global SRS)           
            cand_pt_gSRS = np.zeros((n_cand_gSRS, self._dim))
            if n_cand_gSRS > 0:
                for d, bd in enumerate(self.act_node['domain']):
                    cand_pt_gSRS[:, d] = np.random.uniform(low=bd[0], high=bd[1], size=n_cand_gSRS)
            # find x_star
            Y_fit = self.rbf_mod(self.x_node)
            min_ix = np.argmin(Y_fit)
            x_star = self.x_node[min_ix]
            assert(np.all([bd[0] <= x_star[j] <= bd[1] for j,bd in enumerate(self.act_node['domain'])])) # sanity check
            # find step size (i.e. std) for each coordinate of `x_star`
            step_size_arr = np.array([self.sigma*(bd[1]-bd[0]) for bd in self.act_node['domain']])
            assert(np.min(step_size_arr) > 0) # sanity check
            # generate candidate points (Gaussian about x_star, local SRS)
            cand_pt_lSRS = np.random.multivariate_normal(x_star, np.diag(step_size_arr**2), n_cand_lSRS)
            # combine two types of candidate points
            comb_cand_pt = np.vstack((cand_pt_gSRS, cand_pt_lSRS))
            # put candidate points back to the domain, if there's any outside
            uniq_cand_pt, raw_cand_pt = put_back_box(comb_cand_pt, self.act_node['domain'])
            # get candidate points (``len(uniq_cand_pt) < n_worker`` is pathological case, almost never encountered in practice)
            cand_pt = uniq_cand_pt if len(uniq_cand_pt) >= self._n_worker else raw_cand_pt
        
        # select new points from candidate points
        n_cand = len(cand_pt)
        assert(n_cand >= self._n_worker)
        resp_cand = self.rbf_mod(cand_pt)
        resp_score = scale_zero_one(resp_cand) # response score
        # initializations
        new_pt = np.zeros((self._n_worker, self._dim))
        refer_pt = self.x_node.copy() # reference points based on which we compute distance scores
        # select points sequentially
        for j in range(self._n_worker):
            wt = self.srs_wgt_pat[j]
            if len(refer_pt) > 0:
                if j == 0:                                       
                    # distance matrix for `refer_pt` and `cand_pt`
                    dist_mat = cdist(cand_pt, refer_pt)   
                    dist_cand = np.amin(dist_mat, axis=1)                        
                else:
                    # distance to the previously proposed point
                    dist_prop_pt = cdist(cand_pt, new_pt[j-1].reshape((1, -1))).flatten()                    
                    dist_cand = np.minimum(dist_cand, dist_prop_pt)
                dist_score = scale_one_zero(dist_cand) # distance score
            else:
                # pathological case
                dist_score = np.zeros(n_cand) # distance score
            cand_score = resp_score*wt+(1-wt)*dist_score # candidate score        
            assert (np.amax(cand_score)<=1 and np.amin(cand_score)>=0) # sanity check            
            # select the best one based on the score
            min_ix = np.argmin(cand_score)
            new_pt[j] = cand_pt[min_ix]           
            # update variables
            refer_pt = np.vstack((refer_pt, new_pt[j].reshape(1, -1)))
            dist_cand = np.delete(dist_cand, min_ix)
            resp_score = np.delete(resp_score, min_ix)
            cand_pt = np.delete(cand_pt, min_ix, axis=0)
            n_cand -= 1
            
        return new_pt
        
    
    def update(self, new_x, new_y, verbose=True):
        """
        Update the state of the optimizer.
        
        Args:
            
            new_x (2d array): Proposed new points. Each row is one point.
            
            new_y (1d array): (Noisy) values of the points in `new_x`.
            
            verbose (bool, optional): Whether to verbose about updating the state of the optimizer.
        """
        # FIXME: check verbose
        np.random.set_state(self.random_state)
        
        t1 = default_timer()
        
        self.i_iter += 1
        self.x_all = np.vstack((self.x_all, new_x))
        self.y_all = np.append(self.y_all, new_y)
        self.seed_all = np.append(self.seed_all, self.eval_seeds)
        min_ix = np.argmin(self.y_all)
        self.best_x = self.x_all[min_ix]
        self.best_y = self.y_all[min_ix]
        self.t_build_arr = np.append(self.t_build_arr, self.t_build) 
        self.t_srs_arr = np.append(self.t_srs_arr, self.t_srs)
        self.t_prop_arr = np.append(self.t_prop_arr, self.t_prop)
        self.t_eval_arr = np.append(self.t_eval_arr, self.t_eval)
        self.gSRS_pct_arr = np.append(self.gSRS_pct_arr, self.gSRS_pct)
        self.zoom_lv_arr = np.append(self.zoom_lv_arr, self.zoom_lv)
           
        if self.i_iter_doe < self._n_iter_doe: # i.e., current iteration is in DOE phase
            self.i_iter_doe += 1
        else:
            # update weight pattern in SRS method
            if self._n_worker == 1:
                self.srs_wgt_pat = np.array([self._wgt_pat_bd[0]]) if self.srs_wgt_pat[0] == self._wgt_pat_bd[1] \
                    else np.array([self._wgt_pat_bd[1]]) # alternating weights
            # update tree node
            npt = len(self.x_all)
            self.act_node['ix'] = np.append(self.act_node['ix'], np.arange(npt, npt+self._n_worker, dtype=int))
            if self._n_worker > 1 or (self._n_worker == 1 and self.srs_wgt_pat[0] == self._wgt_pat_bd[0]):
                if self.p_val >= 0.1:
                    # compute p_val
                    if self._use_eff_n_samp:
                        eff_n = eff_npt(self.x_all[self.act_node['ix']], self.act_node['domain'])
                    else:
                        eff_n = len(self.x_all[self.act_node['ix']])
                    self.p_val = self.p_val*eff_n**(-self._alpha/float(self._dim))
                    
                if self.gSRS_pct == 0: # i.e. pure local SRS
                    best_Y_prev = np.min(self.y_node)
                    best_Y_new = np.min(new_y) # minimum of Y values of newly proposed points
                    if best_Y_prev <= best_Y_new: # failure                
                        self.n_fail += 1 # count failure
                    else:
                        self.n_fail = 0
                    if self.n_fail == self._max_C_fail:
                        self.n_fail = 0
                        self.gamma -= self._delta_gamma
                        self.n_reduce_sigma += 1 # update counter
                self.act_node['state']['p'] = self.p_val
                self.act_node['state']['Cr'] = self.n_reduce_sigma
                self.act_node['state']['Cf'] = self.n_fail
                self.act_node['state']['gamma'] = self.gamma
            
            if self.n_reduce_sigma > self._max_n_reduce_sigma:
                # then we either restart or zoom-in (i.e., critical state is reached)
                Y_fit = self.rbf_mod(self.x_all[self.act_node['ix']])
                min_ix = np.argmin(Y_fit)
                x_star = self.x_all[self.act_node['ix']][min_ix]
                # suppose we're going to zoom in
                child_node_ix = self.get_child_node(x_star)
                if child_node_ix is None:
                    # then we create a new child (if zoom in)
                    domain_lb, domain_ub = zip(*self.act_node['domain'])
                    blen = np.array(domain_ub)-np.array(domain_lb) # bound length for each dimension
                    assert(np.min(blen)>0)
                    domain_lb = np.maximum(x_star-self._rho/2.*blen, domain_lb)
                    domain_ub = np.minimum(x_star+self._rho/2.*blen, domain_ub)
                    domain = list(zip(domain_lb, domain_ub)) # the list function is used to ensure compatibility of python3
                    child_node = {'ix': np.nonzero(boxify(self.x_all, domain)[0])[0], 
                                  'domain': domain,
                                  'parent_ix': self.act_node_ix,
                                  'beta': self._init_beta,
                                  'state': self.init_node_state()}
                else:
                    # then we activate an existing child node (if zoom in)
                    child_node = self.tree[self.zoom_lv+1][child_node_ix]
                    child_node['ix'] = np.nonzero(boxify(self.x_all, child_node['domain'])[0])[0]
                child_npt = len(child_node['ix'])
                domain_lb, domain_ub = zip(*child_node['domain'])
                blen = np.array(domain_ub)-np.array(domain_lb) # bound length for each dimension
                assert(np.min(blen)>0)
                
                if np.all(blen*child_npt**(-1./self._dim) < (self._prob.domain_ub-self._prob.domain_lb)*self._resol): # resolution condition
                    # then we restart
                    if verbose:
                        print('Restart for the next iteration!')
                    self.i_iter_doe = 0
                    self.doe_samp = self.doe()
                    self.i_restart += 1
                    self.zoom_lv = 0
                    self.act_node_ix = 0
                    self.x_all = np.zeros((0, self._dim))
                    self.y_all = np.zeros(0)
                    self.tree = self.init_tree()
                else:
                    # then we zoom in
                    self.act_node['state'] = self.init_node_state() # reset the state of the current node
                    self.zoom_lv += 1
                    if child_node_ix is None:
                        # then we create a new child
                        if self.zoom_lv not in self.tree.keys():
                            self.act_node_ix = 0
                            self.tree[self.zoom_lv] = [child_node]    
                        else:
                            self.act_node_ix = len(self.tree[self.zoom_lv])
                            self.tree[self.zoom_lv].append(child_node)    
                        if verbose:
                            print('Zoom in (create a new child node)!')       
                    else:
                        # then activate existing child node
                        self.act_node_ix = child_node_ix
                        # reduce zoom-out probability
                        child_node['beta'] = max(self._min_beta, child_node['beta']/2.)
                        if verbose:
                            print('Zoom in (activate an existing child node)!')
            
            if self._n_worker > 1 or (self._n_worker == 1 and self.srs_wgt_pat[0] == self._wgt_pat_bd[0]):
                if np.random.uniform() < self.tree[self.zoom_lv][self.act_node_ix]['beta'] and self.zoom_lv > 0 and self.i_iter_doe >= self._n_iter_doe:
                    # then we zoom out
                    child_node = self.tree[self.zoom_lv][self.act_node_ix]
                    self.act_node_ix = child_node['parent_ix']
                    self.zoom_lv -= 1
                    assert(self.act_node_ix is not None)
                    # check that the node after zooming out will contain the current node
                    assert(domain_intersect(self.tree[self.zoom_lv][self.act_node_ix]['domain'], child_node['domain']) == child_node['domain'])
                    if verbose:
                        print('Zoom out!')
        
        t2 = default_timer()
        self.t_update_arr = np.append(self.t_update_arr, t2-t1)
        
        self.save_state()
        self.random_state = np.random.get_state()
        
    
    def eval_pt(self, x, verbose=True):
        """
        Evaluate proposed points.
        
        Args:
            
            x (2d array): Points to be evaluated. Each row is one point.
            
            verbose (bool, optional): Whether to verbose about the evaluation.
        
        Returns:
            
            y (1d array): Evaluations of points in `x`.
        """
        # FIXME: add verbose
        np.random.set_state(self.random_state)
        
        t1 = default_timer()
        
        self.eval_seeds = self._seed+np.arange(self.i_iter*self._n_worker, (self.i_iter+1)*self._n_worker, dtype=int)
        y = eval_func(self._prob.f, x, n_proc=self._n_worker, seeds=self.eval_seeds.tolist(),
                      seed_func=self._seed_func)
        
        t2 = default_timer()
        self.t_eval = t2-t1
        
        self.random_state = np.random.get_state()
        
        return y
   
     
    def is_done(self):
        """
        Check whether we are done with the optimization.
        
        Returns:
            
            done (bool): Indicator.
        """
        if self._n_iter is None:
            assert(self.i_restart <= self._n_restart+1)
            done = self.i_restart > self._n_restart
        else:
            assert(self.i_iter <= self._n_iter)
            done = self.i_iter == self._n_iter
            
        return done
    
    
    def get_child_node(self, x_star):
        """
        Get the child node for the optimization tree.
        
        Args:
            
            x_star (1d array): Focal point of zoom-in.
            
        Returns:
            
            child_ix (int or None): Selected child node index. 
                If None, then we need to create a new child.
        """
        assert(x_star.ndim == 1)
        # get zoom level of a child node
        n_zoom_child = self.zoom_lv+1
        if n_zoom_child not in self.tree.keys():
            child_ix = None
        else:
            # the indice of candidate child nodes
            child_node_ix_list = [i for i, c in enumerate(self.tree[n_zoom_child]) if all(boxify(x_star.reshape((1, -1)), c['domain'])[0])]
            if len(child_node_ix_list) == 0:
                child_ix = None
            else:
                # find the child node among the candidates, of which the center of the domain is closest to x_star
                dist_list = [np.linalg.norm(np.mean(self.tree[n_zoom_child][i]['domain'])-x_star) for i in child_node_ix_list]
                child_ix = child_node_ix_list[np.argmin(dist_list)]
        
        return child_ix
    
    
    def save_state(self):
        """
        Save the state of the optimizer to files.
        """
        # FIXME: rewrite the following
        
        # save random state to pickle file for possible resume
        if os.path.isfile(result_pkl_file):
            os.remove(result_pkl_file) # remove file if exists 
        with open(result_pkl_file,'wb') as f:
            pickle.dump(np.random.get_state(),f)
            
        # save to npz file
        temp_result_npz_file = result_npz_file+TEMP_RESULT_NPZ_FILE_SUFFIX # first save to temporary file to avoid loss of data upon termination
        np.savez(temp_result_npz_file,
                 # experiment condition parameters
                 init_iter=init_iter,opt_iter=k+1,n_proc=n_proc,n_core_node=n_core_node,seed=seed,outdir=outdir,save_samp=save_samp,verbose=verbose,
                 n_cand_fact=n_cand_fact,use_eff_n_samp=use_eff_n_samp,init_beta=init_beta,
                 normalize_data=normalize_data,init_gamma=init_gamma,delta_gamma=delta_gamma,
                 max_n_reduce_sigma=max_n_reduce_sigma,rho=rho,init_sigma=init_sigma,
                 init_p=init_p,alpha=alpha,wgt_pat_bd=wgt_pat_bd,
                 lambda_range=lambda_range,
                 n_fold=n_fold,resol=resol,min_beta=min_beta,rbf_kernel=rbf_kernel,
                 func_bd=func_bd,max_C_fail=max_C_fail,resume_iter=resume_iter,n_iter=n_iter,
                 # optimization results
                 t_build_arr=t_build_arr[:k+1],t_prop_arr=t_prop_arr[:k+1],
                 t_srs_arr=t_srs_arr[:k+1],
                 t_eval_arr=t_eval_arr[:k+1],t_update_arr=t_update_arr[:k+1],
                 gSRS_pct_arr=gSRS_pct_arr[:k+1],zoom_lv_arr=zoom_lv_arr[:k+1],
                 # state variables
                 full_restart=full_restart,zoom_lv=zoom_lv,act_node_ix=act_node_ix,
                 X_all=X_all,Y_all=Y_all,tree=tree,n_full_restart=n_full_restart,
                 seed_base=seed_base,seed_flat_arr=seed_flat_arr,
                 X_samp_restart=X_samp_restart)
        
        shutil.copy2(temp_result_npz_file,result_npz_file) # overwrite the original one
        os.remove(temp_result_npz_file) # remove temporary file
   
    
    def load_state(self):
        """
        Load the state of the optimizer from files.
        """
        # FIXME: rewrite the following
        
        t1 = default_timer()
            
        resume_opt_iter = resume_iter-init_iter
        resume_opt_iter = int(resume_opt_iter) # convert to integer type if not
        # remove lock file if exists
        result_npz_lock_file = result_npz_file+'.lock'
        if os.path.isfile(result_npz_lock_file):
            os.remove(result_npz_lock_file)
        # read experiment conditions from previous trials
        data = np.load(result_npz_file)
        assert(resume_opt_iter==data['opt_iter']), \
        'Please resume from where it ended last time by setting resume_iter = %d' % (data['opt_iter']+init_iter)

        # sanity check for consistency of experiment conditions
        assert(init_iter==data['init_iter'] and n_proc==data['n_proc'] and seed==data['seed']
               and np.all(func_bd==data['func_bd']) and n_core_node==data['n_core_node']
               and n_cand_fact==data['n_cand_fact'] and alpha==data['alpha'] 
               and normalize_data==data['normalize_data'] and init_gamma==data['init_gamma']
               and delta_gamma==data['delta_gamma'] and max_n_reduce_sigma==data['max_n_reduce_sigma']
               and rho==data['rho'] and np.all(init_sigma==data['init_sigma'])
               and init_p==data['init_p'] and np.all(wgt_pat_bd==data['wgt_pat_bd'])
               and np.all(lambda_range==data['lambda_range'])
               and n_fold==data['n_fold'] and resol==data['resol'] and min_beta==data['min_beta']
               and init_beta==data['init_beta'] and max_C_fail==data['max_C_fail']
               and use_eff_n_samp==data['use_eff_n_samp'] and rbf_kernel==data['rbf_kernel'])
        
        # read status variables from previous experiment
        full_restart = data['full_restart'].item(0)
        X_samp_restart = data['X_samp_restart']
        zoom_lv = data['zoom_lv'].item(0)
        act_node_ix = data['act_node_ix'].item(0)
        X_all = data['X_all']
        Y_all = data['Y_all']
        tree = data['tree'].item(0)
        n_full_restart = data['n_full_restart'].item(0)
        seed_base = data['seed_base'].item(0)
        seed_flat_arr = data['seed_flat_arr']
        
        # read optimization results from previous experiment
        t_build_arr[:resume_opt_iter] = data['t_build_arr']
        t_prop_arr[:resume_opt_iter] = data['t_prop_arr']
        t_srs_arr[:resume_opt_iter] = data['t_srs_arr']
        t_eval_arr[:resume_opt_iter] = data['t_eval_arr']
        t_update_arr[:resume_opt_iter] = data['t_update_arr']
        gSRS_pct_arr[:resume_opt_iter] = data['gSRS_pct_arr']
        zoom_lv_arr[:resume_opt_iter] = data['zoom_lv_arr']
        
        # load random state
        result_pkl_lock_file = result_pkl_file+'.lock'
        if os.path.isfile(result_pkl_lock_file):
            os.remove(result_pkl_lock_file)
        with open(result_pkl_file, 'rb') as f:
            np.random.set_state(pickle.load(f))
            
        t2 = default_timer()
        t_resume = t2-t1
        if verbose:
            print('\ntime to prepare resume = %.2e sec' % t_resume)
            
        
    def visualize(self, fig_paths={'optim_curve': None, 'zoom_level': None, 'time': None}):
        """
        Visualize optimization progress.
        
        Args:
            
            fig_paths (dict, optional): Paths where the plots will be saved.
                The keys of `fig_paths` are the types of plots. The values
                of the keys are paths to be saved. If the value of a key is None 
                or the key does not exist, a plot will be shown but no figure 
                will be saved for the plot. If the value of a key is specified,
                then a plot will be shown, and will be saved to this key value.
        """
    