"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Implement ProSRS algorithm.
"""
import numpy as np
import os, sys
from pyDOE import lhs
from pathos.multiprocessing import ProcessingPool as Pool
from ..utility.constants import STATE_NPZ_FILE_TEMP, STATE_PKL_FILE_TEMP
from ..utility.classes import std_out_logger, std_err_logger
#TODO: change 'processes=' to 'nodes=' for pool initialization.
#TODO: change 'wgt_expon' to 'gamma'. "wgt_expon = - gamma". Change rbf_wgt() and optimizer function accordingly.


class Optimizer:
    """
    A class that handles optimization using ProSRS algorithm.
    """
    def __init___(self, prob, n_worker, n_proc_master=None, n_iter=None, n_restart=2, resume=False, 
                  seed=1, seed_func=None, out_dir='out'):
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
        self._n_iter_doe = int(np.ceil(3/float(self._n_worker))) # default number of iterations for DOE.
        self._n_iter_doe = min(self._n_iter, self._n_iter_doe) if self._n_iter is not None else self._n_iter_doe # adjusted for the fact that self._n_iter_doe <= self._n_iter.
        self._n_doe_samp = self._n_iter_doe * self._n_worker # number of DOE samples
        self._n_cand = self._n_cand_fact * self._dim # number of candidate points in SRS method
        self._state_npz_file = os.path.join(self._out_dir, STATE_NPZ_FILE_TEMP % self._prob.name) # file that saves optimizer state (useful for resume)
        self._state_pkl_file = os.path.join(self._out_dir, STATE_PKL_FILE_TEMP % self._prob.name) # file that saves optimizer state (useful for resume)
        
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
            self.remain_doe_samp = np.zeros((0, self._dim)) # remaining DOE samples
            self.i_iter_doe = 0 # iteration index during DOE phase (how many DOE iterations have been run in current cycle)
            self.t_build_arr = np.zeros(0) # time of building a RBF model for each iteration. If an iteration is DOE, = np.nan.  
            self.t_prop_arr = np.zeros(0) # time of proposing new points for each iteration.
            self.t_eval_arr = np.zeros(0) # time of evaluating proposed points for each iteration.
            self.t_update_arr = np.zeros(0) # time of updating the optimizer state for each iteration.
            self.gSRS_pct_arr = np.zeros(0) # pertentage of global SRS (= percentage of Type I candidates = floor(10*p_val)/10.) for each iteration.
            self.zoom_lv_arr = np.zeros(0) # zoom level at the time of proposing new points for each iteration.
            self.x_all = np.zeros((0, self._dim)) # all the evaluated (proposed) points so far.
            self.y_all = np.zeros(0) # (noisy) y values of `self.x_all`.
            self.seed_all = np.zeros(0) # random seeds for points in `self.x_all`.
            self.best_x = np.ones(self._dim)*np.nan # best point so far.
            self.best_y = np.nan # (noisy) y value of the best point `self.best_x`.
            np.random.seed(self._seed) # set random seed.
            self.random_state = np.random.get_state() # numpy random state.
            self.zoom_lv = 0 # zoom level (zero-based).
            self.act_node_ix = 0 # index of the activate node for the zoom level `self.zoom_lv` (zero-based).
            self.srs_wgt_pat = np.linspace(self._wgt_pat_bd[0], self._wgt_pat_bd[1], self._n_proc_master) # weight pattern in the SRS method.
            # initialize optimization tree
            self.tree = {self.zoom_lv: [{'ix': np.arange(self._n_doe_samp, dtype=int), # indice of samples for the tree node (w.r.t. `self.x_all` or `self.y_all`).
                                         'bd': self._prob.domain, # domain of the tree node.
                                         'parent_ix': None, # parent node index for the upper zoom level (zero-based). If None, there's no parent.
                                         'zp': self._init_beta, # zoom-out probability.
                                         'state': self.init_node_state() # state of the tree node.
                                         }]}
            
            # TODO: possibly initialize more variables hereafter
            
        else:
            # load optimizer state from the last run
            self.load_state()
            
        # TODO: to be continued
        
        
        
        
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
        self._verbose = verbose
        self._std_out_file = std_out_file
        self._std_err_file = std_err_file
            
        # Log standard outputs and standard errors to files
        # Here we write to a new file if we do not resume. Otherwise, we append to the old file.
        if self._std_out_file is not None:
            if not self._resume:
                if os.path.isfile(self._std_out_file):
                    os.remove(self._std_out_file)
            sys.stdout = std_out_logger(self._std_out_file)    
        if self._std_err_file is not None:
            if not self._resume:
                if os.path.isfile(self._std_err_file):
                    os.remove(self._std_err_file)
            sys.stderr = std_err_logger(self._std_err_file)
        
        # main loop
        while not self.is_done():
            
            # TODO: to be continued
            
            # propose new points
            
            if verbose:
                pass 
            
            # evaluate proposed points
            
            if verbose:
                pass
            
            # update optimizer state (also save the state)
            
            if verbose:
                pass
            
        
        
        ########################### Functions ###############################
        
        def propose(gSRS_pct,x_star,rbf_mod,n_reduce_step_size,wgt_pat_arr,fit_bd,X_samp):
            '''
            Propose points using SRS method.
            Input:
                gSRS_pct: float
                x_star: 1d array
                rbf_mod: function
                n_reduce_step_size: int
                wgt_pat_arr: weight pattern array, 1d array
                fit_bd: list of tuples
                X_samp: 2d array
            Output:
                prop_pt_arr: 2d array
            '''
            assert(wgt_pat_arr.ndim == 1 and np.min(wgt_pat_arr) >= 0 and np.max(wgt_pat_arr) <= 1)
            n_prop = len(wgt_pat_arr) # number of proposed points
            
            if gSRS_pct == 1:
                
                #### pure global SRS ####
                
                # generate candidate points uniformly (global SRS)
                cand_pt = np.zeros((n_cand,dim))
                for d,bd in enumerate(fit_bd):
                    cand_pt[:,d] = np.random.uniform(low=bd[0],high=bd[1],size=n_cand)
                prop_pt_arr = SRS(rbf_mod,cand_pt,X_samp,wgt_pat_arr)
                        
            else:
                #### global-local SRS (possibly pure local SRS) ####
                
                # get number of candidate points for both global and local SRS
                n_cand_gSRS = int(np.round(n_cand*gSRS_pct))
                n_cand_lSRS = n_cand-n_cand_gSRS # total number of candidate points for local SRS
                assert (n_cand_lSRS>0) # sanity check
                
                # generate candidate points uniformly (global SRS)           
                cand_pt = np.zeros((n_cand_gSRS,dim))
                if n_cand_gSRS>0:
                    for d,bd in enumerate(fit_bd):
                        cand_pt[:,d] = np.random.uniform(low=bd[0],high=bd[1],size=n_cand_gSRS)
                
                # find step size (i.e. std) for each coordinate of x_star
                sigma = init_sigma*0.5**n_reduce_step_size
                step_size_arr = np.array([sigma*(x[1]-x[0]) for x in fit_bd])
                assert(np.min(step_size_arr)>0) # sanity check
                
                # generate candidate points (gaussian about x_star, local SRS)
                cand_pt_lSRS = np.random.multivariate_normal(x_star,\
                                                             np.diag(step_size_arr**2),n_cand_lSRS)
                # add to candidate points
                cand_pt = np.vstack((cand_pt,cand_pt_lSRS))
                
                # put candidate points back to the domain, if there's any outside
                cand_pt, cand_pt_raw = put_back_box(cand_pt,fit_bd)
                if len(cand_pt) >= n_prop:
                    prop_pt_arr = SRS(rbf_mod,cand_pt,X_samp,wgt_pat_arr)
                else:
                    # this rarely happens, then we use raw candidate point (possibly with duplicate points)
                    prop_pt_arr = SRS(rbf_mod,cand_pt_raw,X_samp,wgt_pat_arr)
                
            return prop_pt_arr
        
        ########################### Main program ############################
        
        if resume_iter is None:
            
            ########### Initial sampling (DOE) ################ 
            
            np.random.seed(int(seed))
            
            t1 = default_timer()
            
            n_init_samp = n_proc*init_iter
            X_all = doe(n_init_samp,func_bd)
                
            t2 = default_timer()
            
            t_doe = t2-t1
            
            ########### Evaluate DOE ################
            
            t_eval = np.zeros(init_iter)
            
            seed_base = seed+1
            seed_flat_arr = np.array([])
            
            # initialization
            Y_all = np.zeros(n_init_samp)
            for k in range(init_iter):
                ix1,ix2 = k*n_proc,(k+1)*n_proc
                X = X_all[ix1:ix2]
                seed_arr = seed_base+np.arange(n_proc)
                seed_flat_arr = np.append(seed_flat_arr,seed_arr) # update seed_flat_arr for sanity check
                seed_base += n_proc
                out_list = [os.path.join(outdir, RESULT_SAMP_FILE_TEMP % (prob_name,k+1,n+1)) \
                            for n in range(n_proc)]
                t1 = default_timer()
                # evaluate function
                Y = func_eval(obj_func,X,seed_arr,out_list,pool_eval,comm)
                t2 = default_timer()
                t_eval[k] = t2-t1
                # save Y
                Y_all[ix1:ix2] = Y
                
                # save to netcdf file
                if save_samp:
                    out_file = os.path.join(outdir, RESULT_GENERAL_SAMP_FILE_TEMP % (prob_name,k+1))
                    if os.path.isfile(out_file):
                        os.remove(out_file) # remove file if exists
                    with netcdf.netcdf_file(out_file,'w') as f:
                        f.createDimension('var',dim)
                        f.createDimension('samp',n_proc)
                        nc_samp = f.createVariable('x','d',('samp','var'))
                        nc_samp[:] = X
                        nc_samp = f.createVariable('y','d',('samp',))
                        nc_samp[:] = Y
                        nc_seed = f.createVariable('seed','i',('samp',))
                        nc_seed[:] = seed_arr
                        nc_wall_time = f.createVariable('tw_eval','d',())
                        nc_wall_time.assignValue(t_eval[k])
                        nc_wall_time = f.createVariable('tw_doe','d',())
                        nc_wall_time.assignValue(t_doe)
            
            # sanity check
            assert(np.all(seed_flat_arr == np.arange(np.min(seed_flat_arr),np.max(seed_flat_arr)+1)))
        
        
            ######### Initializations for optimization ########
                
            # count number of full restart
            n_full_restart = 0
            # full restart flag
            full_restart = False
            # doe samples for restart (useful for resume)
            X_samp_restart = np.zeros((0,dim))
        
        else:
            
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
        
        ########### Optimization iterations ################
        
        start_iter = 0 if resume_iter is None else resume_opt_iter
        
        for k in range(start_iter,opt_iter):
            
            if verbose:
                print('\nStart optimization iteration = %d/%d' % (k+1,opt_iter))
            
            tt1 = default_timer()
            
            ########### Read activated node #######
                
            # get activated node
            act_node = tree[zoom_lv][act_node_ix]
            # get samples of the node
            X_samp = X_all[act_node['ix']]
            Y_samp = Y_all[act_node['ix']]
            # get variables
            gSRS_pct_full = act_node['state']['p']
            n_reduce_step_size = act_node['state']['Cr']
            n_fail = act_node['state']['Cf']
            gamma = act_node['state']['gamma']
            zoom_out_prob = act_node['zp']
            fit_bd = act_node['bd']
                    
            if not full_restart:
                    
                ########### Fit model ##############
                
                t1 = default_timer()
    
                rbf_mod,opt_sm,cv_err,_ = RBF_reg(X_samp,Y_samp,n_core_node,lambda_range,
                                                  normalize_data=normalize_data,gamma=gamma,
                                                  n_fold=n_fold,kernel=rbf_kernel,pool=pool_rbf,
                                                  poly_deg=rbf_poly_deg)                
                t2 = default_timer()

                t_build_arr[k] = t2-t1
                
                if verbose:
                    print('time to build model = %.2e sec' % t_build_arr[k])
                    
                ########### Propose points for next iteration ############## 
                
                t1 = default_timer()
                
                # find gSRS_pct
                gSRS_pct = np.floor(10*gSRS_pct_full)/10.
                assert(0<=gSRS_pct<=1)
                
                if verbose:
                    print('act_node_ix = %d' % act_node_ix)
                    print('gamma = %g' % gamma)
                    print('opt_sm = %.1e' % opt_sm)
                    print('gSRS_pct = %g' % gSRS_pct)
                    print('n_fail = %d' % n_fail)
                    print('n_reduce_step_size = %d' % n_reduce_step_size)
                    print('sigma = %g' % (init_sigma*0.5**n_reduce_step_size))
                    print('zoom_out_prob = %g' % zoom_out_prob)
                    print('zoom_lv = %d' % zoom_lv)
                    print('fit_bd =')
                    print(fit_bd)
                    
                # find x_star
                Y_fit = rbf_mod(X_samp)
                min_ix = np.argmin(Y_fit)
                x_star = X_samp[min_ix]
                
                if n_proc == 1:
                    # prepare for next iteration
                    srs_wgt_pat = np.array([wgt_pat_bd[0]]) if srs_wgt_pat[0] == wgt_pat_bd[1] else np.array([wgt_pat_bd[1]])
                
                prop_pt_arr = propose(gSRS_pct,x_star,rbf_mod,n_reduce_step_size,srs_wgt_pat,fit_bd,X_samp)
                
                assert(np.all([bd[0]<=x_star[j]<=bd[1] for j,bd in enumerate(fit_bd)])) # sanity check
                
                t2 = default_timer()
                
                t_prop_arr[k] = t2-t1
                
                if verbose:
                    print('time to propose points = %.2e sec' % t_prop_arr[k])
        
            else:
                
                # i.e. do full restart
                if n_full_restart == 0:
                    n_init_samp = n_proc*init_iter
                    X_samp_restart = doe(n_init_samp,func_bd)
                
                ix1, ix2 = n_full_restart*n_proc, (n_full_restart+1)*n_proc
                prop_pt_arr = X_samp_restart[ix1:ix2]
                
                gSRS_pct = np.nan
            
            tm1 = default_timer()        
            # save variables
            gSRS_pct_arr[k] = gSRS_pct # save gSRS_pct
            zoom_lv_arr[k] = zoom_lv # save zoom_lv
            
            ############ Evaluate proposed points #############
            
            seed_arr = seed_base+np.arange(n_proc)
            seed_flat_arr = np.append(seed_flat_arr,seed_arr) # update seed_flat_arr for sanity check
            seed_base += n_proc
            out_list = [os.path.join(outdir, RESULT_SAMP_FILE_TEMP % (prob_name,k+init_iter+1,n+1)) \
                        for n in range(n_proc)]
            t1 = default_timer()
            Y_prop_pt = func_eval(obj_func,prop_pt_arr,seed_arr,out_list,pool_eval,comm)
            t2 = default_timer()
            Y_prop_pt = np.array(Y_prop_pt)
            assert(len(prop_pt_arr) == len(Y_prop_pt) == n_proc)
            
            assert(np.all(seed_flat_arr == np.arange(np.min(seed_flat_arr),np.max(seed_flat_arr)+1)))
            
            t_eval_arr[k] = t2-t1
            
            if verbose:
                print('time to evaluate points = %.2e sec' % t_eval_arr[k])
            
            # update node
            n_X_all = len(X_all)
            act_node['ix'] = np.append(act_node['ix'], np.arange(n_X_all,n_X_all+n_proc,dtype=int))
            # update samples
            X_all = np.vstack((X_all,prop_pt_arr))
            Y_all = np.append(Y_all,Y_prop_pt)
                
            # update state of the current node
            if not full_restart:
                    
                if n_proc > 1 or (n_proc == 1 and srs_wgt_pat[0] == wgt_pat_bd[0]):
                    if gSRS_pct_full >= 0.1:
                        # compute gSRS_pct_full
                        if use_eff_n_samp:
                            eff_n = eff_n_samp(X_all[act_node['ix']],act_node['bd'])
                        else:
                            eff_n = len(X_all[act_node['ix']])
                        gSRS_pct_full = gSRS_pct_full*eff_n**(-alpha/float(dim))
                        
                    if gSRS_pct == 0: # i.e. pure local SRS
                        best_Y_prev = np.min(Y_samp)
                        best_Y_new = np.min(Y_prop_pt) # minimum of Y values of newly proposed points
                        if best_Y_prev <= best_Y_new: # failure                
                            n_fail += 1 # count failure
                        else:
                            n_fail = 0
                        if n_fail == max_C_fail:
                            n_fail = 0
                            gamma += delta_gamma
                            n_reduce_step_size += 1 # update counter
                    act_node['state']['p'] = gSRS_pct_full
                    act_node['state']['Cr'] = n_reduce_step_size
                    act_node['state']['Cf'] = n_fail
                    act_node['state']['gamma'] = gamma
            
            # save to netcdf file
            t1 = default_timer()
            if save_samp:
                out_file = os.path.join(outdir, RESULT_GENERAL_SAMP_FILE_TEMP % (prob_name,k+init_iter+1))
                if os.path.isfile(out_file):
                    os.remove(out_file) # remove file if exists
                    
                with netcdf.netcdf_file(out_file,'w') as f:
                    f.createDimension('var',dim)
                    f.createDimension('samp',n_proc)
                    nc_samp = f.createVariable('x','d',('samp','var'))
                    nc_samp[:] = prop_pt_arr
                    nc_samp = f.createVariable('y','d',('samp',))
                    nc_samp[:] = Y_prop_pt
                    nc_seed = f.createVariable('seed','i',('samp',))
                    nc_seed[:] = seed_arr
                    nc_wall_time = f.createVariable('tw_eval','d',())
                    nc_wall_time.assignValue(t_eval_arr[k])
            
            t2 = default_timer()
            t_save = t2-t1
            
            if verbose:
                print('time to save samples = %.2e sec' % t_save)
            
            ############# Prepare for next iteration #############
            
            tp1 = default_timer()
               
            if not full_restart:
                
                if n_reduce_step_size > max_n_reduce_sigma:
                    # then we either restart or zoom-in
                    Y_fit = rbf_mod(X_all[act_node['ix']])
                    min_ix = np.argmin(Y_fit)
                    x_star = X_all[act_node['ix']][min_ix]
                    # suppose we're going to zoom in
                    child_node_ix = get_child_node(x_star,zoom_lv,tree)
                    if child_node_ix is None:
                        # then we create a new child (if zoom in)
                        fit_lb, fit_ub = zip(*fit_bd)
                        blen = np.array(fit_ub)-np.array(fit_lb) # bound length for each dimension
                        assert(np.min(blen)>0)
                        fit_lb = np.maximum(x_star-rho/2.0*blen,fit_lb)
                        fit_ub = np.minimum(x_star+rho/2.0*blen,fit_ub)
                        fit_bd = list(zip(fit_lb,fit_ub)) # the list function is used to ensure compatibility of python3
                        child_node = {'ix': np.nonzero(get_box_samp(X_all,fit_bd)[0])[0], 
                                      'bd': fit_bd,
                                      'parent_ix': act_node_ix,
                                      'zp': init_beta,
                                      'state': init_node_state()}
                    else:
                        # then we activate an existing child node (if zoom in)
                        child_node = tree[zoom_lv+1][child_node_ix]
                        child_node['ix'] = np.nonzero(get_box_samp(X_all,child_node['bd'])[0])[0]
                    n_samp = len(child_node['ix'])
                    fit_lb, fit_ub = zip(*child_node['bd'])
                    blen = np.array(fit_ub)-np.array(fit_lb) # bound length for each dimension
                    assert(np.min(blen)>0)
                    
                    if np.all(blen*n_samp**(-1./dim)<func_blen*resol): # resolution condition
                        # then we restart
                        if verbose:
                            print('Restart for the next iteration!')                        
                        full_restart = True
                        zoom_lv = 0
                        act_node_ix = 0
                        X_all = np.zeros((0,dim))
                        Y_all = np.zeros(0)
                        tree = {zoom_lv: [{'ix': np.arange(0,dtype=int), # indice of samples for the node (with respect to X_all and Y_all)
                                          'bd': func_bd, # domain of the node
                                          'parent_ix': None, # parent node index for the upper zoom level (zero-based). If None, there's no parent
                                          'zp': init_beta, # zoom-out probability
                                          'state': init_node_state() # state of the node
                                          }]}
                    else:
                        # then we zoom in
                        act_node['state'] = init_node_state() # reset the state of the current node
                        zoom_lv += 1
                        
                        if child_node_ix is None:
                            # then we create a new child
                            if zoom_lv not in tree.keys():
                                act_node_ix = 0
                                tree[zoom_lv] = [child_node]    
                            else:
                                act_node_ix = len(tree[zoom_lv])
                                tree[zoom_lv].append(child_node)
                                
                            if verbose:
                                print('Zoom in (create a new child node)!')
                                
                        else:
                            # then activate existing child node
                            act_node_ix = child_node_ix
                            # reduce zoom-out probability
                            child_node['zp'] = max(min_beta,child_node['zp']/2.)
                            
                            if verbose:
                                print('Zoom in (activate an existing child node)!')
                                
                if n_proc > 1 or (n_proc == 1 and srs_wgt_pat[0] == wgt_pat_bd[0]):            
                    if np.random.uniform() < tree[zoom_lv][act_node_ix]['zp'] and zoom_lv > 0 and not full_restart:
                        # then we zoom out
                        child_node = tree[zoom_lv][act_node_ix]
                        act_node_ix = child_node['parent_ix']
                        zoom_lv -= 1
                        assert(act_node_ix is not None)
                        # check that the node after zooming out will contain the current node
                        assert(intsect_bd(tree[zoom_lv][act_node_ix]['bd'],child_node['bd']) == child_node['bd']) 
                        
                        if verbose:
                            print('Zoom out!')
            
            else:
                # i.e., restart
                n_full_restart += 1          
                if n_full_restart == init_iter:
                    full_restart = False
                    n_full_restart = 0
            
            tm2 = default_timer()
            t_misc = tm2-tm1-t_eval_arr[k]-t_save
            
            if verbose:
                print('time for miscellaneous tasks (saving variables, etc.) = %.2e sec' % t_misc)
            
            # find time to prepare for next iteration
            tp2 = default_timer()
            t_update_arr[k] = tp2-tp1
            
            # find the time to run optimization algortithm (excluding time to evaluate and save samples)
            tt2 = default_timer()
            t_alg = tt2-tt1-t_eval_arr[k]-t_save
            
            if verbose:
                print('time to run optimization algorithm = %.2e sec' % t_alg)
                    
            ########### Save results ###############
            
            t1 = default_timer()
            
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
                     t_eval_arr=t_eval_arr[:k+1],t_update_arr=t_update_arr[:k+1],
                     gSRS_pct_arr=gSRS_pct_arr[:k+1],zoom_lv_arr=zoom_lv_arr[:k+1],
                     # state variables
                     full_restart=full_restart,zoom_lv=zoom_lv,act_node_ix=act_node_ix,
                     X_all=X_all,Y_all=Y_all,tree=tree,n_full_restart=n_full_restart,
                     seed_base=seed_base,seed_flat_arr=seed_flat_arr,
                     X_samp_restart=X_samp_restart)
            
            shutil.copy2(temp_result_npz_file,result_npz_file) # overwrite the original one
            os.remove(temp_result_npz_file) # remove temporary file
            
            t2 = default_timer()
            
            if verbose:
                print('time to save results = %.2e sec' % (t2-t1))  
            
            # save terminal output to file
            if std_out_file is not None:
                sys.stdout.terminal.flush()
                sys.stdout.log.flush()
            if std_err_file is not None:
                sys.stderr.terminal.flush()
                sys.stderr.log.flush()
        
        # find best point and its (noisy) function value
        
        
    def doe(self, n_samp=None, criterion='maximin'):
        """
        Design of experiments using Latin Hypercube Sampling (LHS).
        
        Args:
            
            n_samp (int or None, optional): Number of samples in LHS. 
                If None, we use the default value.
            
            criterion (str, optional): Sampling criterion for LHS.
                For details, see `pyDOE documentation <https://pythonhosted.org/pyDOE/randomized.html>`.
                
        Returns:
            
            samp (2d array): LHS samples. Each row is one sample.
        """
        n_samp = self._n_doe_samp if n_samp is None else n_samp
        unit_X = lhs(self._dim, samples=n_samp, criterion=criterion) # unit_X: 2d array in unit cube
        samp = np.zeros_like(unit_X)
        for i in range(self._dim):
            samp[:, i] = unit_X[:, i]*(self._prob.domain[i][1]-self._prob.domain[i][0])+self._prob.domain[i][0] # scale and shift
            
        return samp
    
    
    def init_node_state(self):
        """
        Initialize the state of a node of the optimization tree.
        
        Returns:
            
            state (dict): values of state variables.
        """
        
        state = {'p': self._init_p, # p value in the SRS method (controls proportion of Type I candidate points).
                 'Cr': 0, # counter that counts number of times of reducing the sigma value of local SRS.
                 'Cf': 0, # counter that counts number of consecutive failures.
                 'gamma': self._init_gamma # weight exponent parameter of weighted RBF
                 }
        
        return state
        
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
    
    def save_state(self):
        """
        Save the state of the optimizer to files.
        """
    
    def load_state(self):
        """
        Load the state of the optimizer from files.
        """
        
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
    