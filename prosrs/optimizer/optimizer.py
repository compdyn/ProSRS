"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Implement ProSRS algorithm.
"""
import numpy as np
from pyDOE import lhs
#TODO: change 'processes=' to 'nodes=' for pool initialization.
#TODO: change 'wgt_expon' to 'gamma'. "wgt_expon = - gamma". Change rbf_wgt() and optimizer function accordingly.

class Optimizer:
    """
    A class that handles optimization using ProSRS algorithm.
    """
    def __init___(self, prob, n_worker, n_proc_master=None, seed=1, out_dir='out'):
        """
        Constructor.
        
        Args:
            
            prob (`prosrs.Problem` type): Optimization problem.
            
            n_worker (int): Number of workers in the optimization.
                This also determines the number of proposed or evaluated
                points per iteration.
            
            n_proc_master (int or None, optional): Number of parallel processes (cores)
                that will be used in the master node. If None, then all the available
                processes (cores) of the master will be used.
            
            seed (int or None, optional): Random seed.
                If None, then we do not set random seed for the optimization.
            
            out_dir (str, optional): Output directory.
                All the output files (e.g., optimization status file) will be saved to
                `out_dir` directory.
        """
        # class members (convention: variables wih prefix '_' are constant parameters during optimization).
        self._dim = prob.dim # dimention of optimization problem.
        self._domain = prob.domain # domain of optimization problem.
        self._n_worker = n_worker # number of workers.
        self._n_cand_fact = 1000 # number of candidate = self._n_cand_fact * self._dim.
        self._wgt_pat_bd = [0.3, 1.] # the bound of the weight pattern in the SRS method.
        self._normalize_data = True # whether to normalize data when training RBF regression model.
        self._init_gamma = 0. # initial weight exponent in the SRS method (>=0). If zero, then disable weighting in RBF regression.
        self._delta_gamma = 2. # amount of decrease of weight exponent whenever failure occurs.
        self._init_sigma = 0.1 # initial sigma value in the SRS method (controls initial spread of Type II candidate points).
        self._max_n_reduce_sigma = 2 # max number of times of halving self.sigma before zooming in/restart. Critical sigma value = self._init_sigma * 0.5**self._max_n_reduce_sigma.
        self._rho = 0.4 # zoom-in factor. Must be in (0, 1).
        self._init_p = 1. # initial p value in the SRS method (roughly = initial proportion of Type I candidate points).
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
        # TODO: sanity check for class members
        
        
        
        
        
    def __str__(self, verbose_level=1):
        """
        Display optimizer status.
        
        Args:
            
            verbose_level (int, optional): Verbose level. The larger `verbose` is,
                the more detailed information will be displayed.
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
    
    def run(self, n_iter=None, n_restart=2, resume=False, verbose=True, std_out_file=None, std_err_file=None):
        """
        Run ProSRS algorithm.
        
        Args:
            
            n_iter (int or None, optional): Total number of iterations for the optimization.
                If ``int``, the optimization will terminate upon finishing running `n_iter` iterations. 
                If None, then we use number of restarts `n_restart` as the termination condition.
                
            n_restart (int, optional): Number of restarts for the optimization.
                This parameter takes effect only when `n_iter` is None, and 
                the optimization will terminate upon finishing restarting `n_restart` times.
            
            resume (bool, optional): Whether to resume from the last run.
                The information of the last run will be read from the directory `self.out_dir`.
                
            verbose (bool, optional): Whether to verbose while running the algorithm.
            
            std_out_file (str or None, optional): Name of standard output file name.
                If ``str``, then standard outputs will be directed to the file {self.out_dir}/{std_out_file}.
                If None, then standard outputs will not be saved to a file.
            
            std_err_file (str or None, optional): Name of standard error file name.
                If ``str``, then standard errors will be directed to the file {self.out_dir}/{std_err_file}.
                If None, then standard errors will not be saved to a file.
            
        """
    
        ############## Algorithm parameters (global constants) ####################
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # TODO: continue from here
        
        
        # convert input to correct data type if not done so
        n_iter = int(n_iter)
        init_iter = min(n_iter, self._n_iter_doe)
        opt_iter = n_iter-init_iter # number of iterations for optimization
        n_proc = int(n_proc)
        n_core_node = int(n_core_node)
        seed = int(seed)
        assert(seed >= 0 and n_core_node > 0 and n_proc > 0)
        
        # get parameters of optimization problem
        dim = prob.dim
        func_bd = prob.domain
        obj_func = prob.object_func # objective function (corrupted with noise)
        prob_name = prob.prob_name
        
        # get file and directory names
        result_npz_file = os.path.join(outdir, RESULT_NPZ_FILE_TEMP % prob_name)
        result_pkl_file = os.path.join(outdir, RESULT_PKL_FILE_TEMP % prob_name)
        
        # get lower and upper bounds for function
        func_lb, func_ub = zip(*func_bd)
        func_lb, func_ub = np.array(func_lb), np.array(func_ub) # convert to numpy array
        assert(np.all(func_lb<func_ub)) # sanity check
        func_blen = func_ub-func_lb # function bound length for each dimension
        # get number of candidate points
        n_cand = int(n_cand_fact*dim)        
        assert(type(max_C_fail) == int and max_C_fail > 0)
        assert(n_cand_fact*dim >= n_proc), 'number of candidate points needs to be no less than n_proc'
        
        # get pool of processes for parallel computing
        pool_eval = Pool(processes=n_proc) # for function evaluations
        pool_rbf = Pool(processes=n_core_node) # for training rbf surrogate
            
        # sanity check
        assert(delta_gamma >= 0 and max_n_reduce_sigma>=0 and 0<=init_p<=1)  
        assert(0<rho<1 and resol>0)
        assert(0<=min_beta<=init_beta<=1)
        assert(n_fold > 1)
        assert(rbf_poly_deg in [0,1])
        assert(min(wgt_pat_bd)>=0 and max(wgt_pat_bd)<=1 and len(wgt_pat_bd)==2)
        
        # check if the output directory exists
        assert(os.path.isdir(outdir))
            
        # log standard outputs/errors to file
        if std_out_file is not None:
            orig_std_out = sys.stdout
            log_file = os.path.join(outdir, std_out_file)
            if resume_iter is None:
                if os.path.isfile(log_file):
                    os.remove(log_file)
            sys.stdout = std_out_logger(log_file)    
        if std_err_file is not None:
            orig_std_err = sys.stderr
            log_file = os.path.join(outdir, std_err_file)
            if resume_iter is None:
                if os.path.isfile(log_file):
                    os.remove(log_file)
            sys.stderr = std_err_logger(log_file)
        
        ########################### Functions ###############################
        
        def init_state():
            '''
            Initialize the state of a node
            Output:
                state: dictionary of state variables
            '''
            
            state = {'p': init_p, # global SRS percentage (full, without rounding to 0.1 resolution)
                     'Cr': 0, # counter that counts number of times of reducing step size of local SRS
                     'Cf': 0, # counter that counts number of consecutive failures
                     'w': init_gamma # weight exponent parameter of weighted RBF
                     }
            return state
        
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
        
        # initializations
        t_build_mod, t_build_mod_cpu = np.zeros(opt_iter), np.zeros(opt_iter) 
        t_prop_pt, t_prop_pt_cpu = np.zeros(opt_iter), np.zeros(opt_iter)  
        t_eval_prop, t_eval_prop_cpu = np.zeros(opt_iter), np.zeros(opt_iter) 
        t_opt, t_opt_cpu = np.zeros(opt_iter), np.zeros(opt_iter)
        t_prep, t_prep_cpu = np.zeros(opt_iter), np.zeros(opt_iter)
        opt_sm_arr = np.nan*np.ones(opt_iter) # save optimal smooth parameter
        cv_err_arr = [None]*opt_iter # save cv error for each iteration
        gSRS_pct_arr = np.zeros(opt_iter) # save gSRS_pct in each iteration
        n_zoom_arr = np.zeros(opt_iter) # save n_zoom in each iteration
        best_loc_it = np.zeros((init_iter+opt_iter,dim)) # save best points for each iteration
        best_val_it = np.zeros(init_iter+opt_iter) # save Y value for the best point for each iteration
        best_seed_it = np.zeros(init_iter+opt_iter,dtype=int) # save random seed corresponding to the best point for each iteration
        
        if resume_iter is None:
            
            ########### Initial sampling (DOE) ################ 
            
            np.random.seed(int(seed))
            
            t1 = default_timer()
            t1_c = clock()
            
            n_init_samp = n_proc*init_iter
            X_all = doe(n_init_samp,func_bd)
                
            t2 = default_timer()
            t2_c = clock()
            
            t_doe = t2-t1
            t_doe_cpu = t2_c-t1_c
            
            ########### Evaluate DOE ################
            
            t_eval = np.zeros(init_iter)
            t_eval_cpu = np.zeros(init_iter)
            
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
                t1, t1_c = default_timer(), clock()
                # evaluate function
                Y = func_eval(obj_func,X,seed_arr,out_list,pool_eval,comm)
                t2, t2_c = default_timer(), clock()
                t_eval[k], t_eval_cpu[k] = t2-t1, t2_c-t1_c
                # get best value and best location for each iteration
                min_ix = np.argmin(Y)
                best_loc_it[k],best_val_it[k] = X[min_ix],Y[min_ix]
                best_seed_it[k] = seed_arr[min_ix]
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
                        nc_cpu_time = f.createVariable('tc_eval','d',())
                        nc_cpu_time.assignValue(t_eval_cpu[k])
                        nc_wall_time = f.createVariable('tw_doe','d',())
                        nc_wall_time.assignValue(t_doe)
                        nc_cpu_time = f.createVariable('tc_doe','d',())
                        nc_cpu_time.assignValue(t_doe_cpu)
            
            # sanity check
            assert(np.all(seed_flat_arr == np.arange(np.min(seed_flat_arr),np.max(seed_flat_arr)+1)))
        
        
            ######### Initializations for optimization ########
                
            # zoom level indicator (zero-based)
            n_zoom = 0
            # activate node index (for zoom level = n_zoom, zero-based)
            act_node_ix = 0
            # count number of full restart
            n_full_restart = 0
            # full restart flag
            full_restart = False
            # running weight for SRS (useful only when n_proc = 1)
            run_wgt_SRS = wgt_pat_bd[0]
            # doe samples for restart (useful for resume)
            X_samp_restart = np.zeros((0,dim))
            
            # optimization tree
            tree = {n_zoom: [{'ix': np.arange(n_init_samp,dtype=int), # indice of samples of the node (with respect to X_all and Y_all)
                              'bd': func_bd, # domain of the node
                              'parent_ix': None, # parent node index for the upper zoom level (zero-based). If None, there's no parent
                              'zp': init_beta, # zoom-out probability
                              'state': init_state() # state of the node
                              }]}
        
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
            run_wgt_SRS = data['run_wgt_SRS'].item(0)
            X_samp_restart = data['X_samp_restart']
            n_zoom = data['n_zoom'].item(0)
            act_node_ix = data['act_node_ix'].item(0)
            X_all = data['X_all']
            Y_all = data['Y_all']
            tree = data['tree'].item(0)
            n_full_restart = data['n_full_restart'].item(0)
            seed_base = data['seed_base'].item(0)
            seed_flat_arr = data['seed_flat_arr']
            
            # read optimization results from previous experiment
            t_build_mod[:resume_opt_iter] = data['t_build_mod']
            t_build_mod_cpu[:resume_opt_iter] = data['t_build_mod_cpu']
            t_prop_pt[:resume_opt_iter] = data['t_prop_pt']
            t_prop_pt_cpu[:resume_opt_iter] = data['t_prop_pt_cpu']
            t_eval_prop[:resume_opt_iter] = data['t_eval_prop']
            t_eval_prop_cpu[:resume_opt_iter] = data['t_eval_prop_cpu']
            t_opt[:resume_opt_iter] = data['t_opt']
            t_opt_cpu[:resume_opt_iter] = data['t_opt_cpu']
            t_prep[:resume_opt_iter] = data['t_prep']
            t_prep_cpu[:resume_opt_iter] = data['t_prep_cpu']
            opt_sm_arr[:resume_opt_iter] = data['opt_sm_arr']
            cv_err_arr[:resume_opt_iter] = data['cv_err_arr']
            gSRS_pct_arr[:resume_opt_iter] = data['gSRS_pct_arr']
            n_zoom_arr[:resume_opt_iter] = data['n_zoom_arr']
            best_loc_it[:init_iter+resume_opt_iter] = data['best_loc_it']
            best_val_it[:init_iter+resume_opt_iter] = data['best_val_it']
            best_seed_it[:init_iter+resume_opt_iter] = data['best_seed_it']
            
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
            tt1_c = clock()
            
            ########### Read activated node #######
                
            # get activated node
            act_node = tree[n_zoom][act_node_ix]
            # get samples of the node
            X_samp = X_all[act_node['ix']]
            Y_samp = Y_all[act_node['ix']]
            # get variables
            gSRS_pct_full = act_node['state']['p']
            n_reduce_step_size = act_node['state']['Cr']
            n_fail = act_node['state']['Cf']
            gamma = act_node['state']['w']
            zoom_out_prob = act_node['zp']
            fit_bd = act_node['bd']
                    
            if not full_restart:
                    
                ########### Fit model ##############
                
                t1 = default_timer()
                t1_c = clock()
    
                rbf_mod,opt_sm,cv_err,_ = RBF_reg(X_samp,Y_samp,n_core_node,lambda_range,
                                                  normalize_data=normalize_data,gamma=gamma,
                                                  n_fold=n_fold,kernel=rbf_kernel,pool=pool_rbf,
                                                  poly_deg=rbf_poly_deg)
                # save parameters
                opt_sm_arr[k] = opt_sm
                cv_err_arr[k] = cv_err
                
                t2 = default_timer()
                t2_c = clock()
                t_build_mod[k] = t2-t1
                t_build_mod_cpu[k] = t2_c-t1_c
                
                if verbose:
                    print('time to build model = %.2e sec' % t_build_mod[k])
                    
                ########### Propose points for next iteration ############## 
                
                t1 = default_timer()
                t1_c = clock()
                
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
                    print('n_zoom = %d' % n_zoom)
                    print('fit_bd =')
                    print(fit_bd)
                    
                # find x_star
                Y_fit = rbf_mod(X_samp)
                min_ix = np.argmin(Y_fit)
                x_star = X_samp[min_ix]
                
                if n_proc > 1:
                    wgt_pat_arr = np.linspace(wgt_pat_bd[0],wgt_pat_bd[1],n_proc)
                else:
                    assert(run_wgt_SRS in wgt_pat_bd)
                    wgt_pat_arr = np.array([run_wgt_SRS])
                    # prepare for next iteration
                    run_wgt_SRS = wgt_pat_bd[0] if run_wgt_SRS == wgt_pat_bd[1] else wgt_pat_bd[1]
                
                prop_pt_arr = propose(gSRS_pct,x_star,rbf_mod,n_reduce_step_size,wgt_pat_arr,fit_bd,X_samp)
                
                assert(np.all([bd[0]<=x_star[j]<=bd[1] for j,bd in enumerate(fit_bd)])) # sanity check
                
                t2 = default_timer()
                t2_c = clock()
                
                t_prop_pt[k] = t2-t1
                t_prop_pt_cpu[k] = t2_c-t1_c
                
                if verbose:
                    print('time to propose points = %.2e sec' % t_prop_pt[k])
        
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
            n_zoom_arr[k] = n_zoom # save n_zoom
            
            ############ Evaluate proposed points #############
            
            seed_arr = seed_base+np.arange(n_proc)
            seed_flat_arr = np.append(seed_flat_arr,seed_arr) # update seed_flat_arr for sanity check
            seed_base += n_proc
            out_list = [os.path.join(outdir, RESULT_SAMP_FILE_TEMP % (prob_name,k+init_iter+1,n+1)) \
                        for n in range(n_proc)]
            t1, t1_c = default_timer(), clock()
            Y_prop_pt = func_eval(obj_func,prop_pt_arr,seed_arr,out_list,pool_eval,comm)
            t2, t2_c = default_timer(), clock()
            Y_prop_pt = np.array(Y_prop_pt)
            assert(len(prop_pt_arr) == len(Y_prop_pt) == n_proc)
            
            assert(np.all(seed_flat_arr == np.arange(np.min(seed_flat_arr),np.max(seed_flat_arr)+1)))
            
            t_eval_prop[k] = t2-t1
            t_eval_prop_cpu[k] = t2_c-t1_c
            
            if verbose:
                print('time to evaluate points = %.2e sec' % t_eval_prop[k])
            
            # update node
            n_X_all = len(X_all)
            act_node['ix'] = np.append(act_node['ix'], np.arange(n_X_all,n_X_all+n_proc,dtype=int))
            # update samples
            X_all = np.vstack((X_all,prop_pt_arr))
            Y_all = np.append(Y_all,Y_prop_pt)
                
            # update state of the current node
            if not full_restart:
                    
                if n_proc > 1 or (n_proc == 1 and run_wgt_SRS == wgt_pat_bd[0]):
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
                    act_node['state']['w'] = gamma
            
            # save to netcdf file
            t1, t1_c = default_timer(), clock()
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
                    nc_wall_time.assignValue(t_eval_prop[k])
                    nc_cpu_time = f.createVariable('tc_eval','d',())
                    nc_cpu_time.assignValue(t_eval_prop_cpu[k])
            
            t2, t2_c = default_timer(), clock()
            t_save, t_save_c = t2-t1, t2_c-t1_c
            
            if verbose:
                print('time to save samples = %.2e sec' % t_save)
            
            ############### Output results #################
            
            # get the best point for current iteration
            min_ix = np.argmin(Y_prop_pt)
            best_loc_it[k+init_iter],best_val_it[k+init_iter] = prop_pt_arr[min_ix],Y_prop_pt[min_ix]
            best_seed_it[k+init_iter] = seed_arr[min_ix]
            
            ############# Prepare for next iteration #############
            
            tp1, tp1_c = default_timer(), clock()
               
            if not full_restart:
                
                if n_reduce_step_size > max_n_reduce_sigma:
                    # then we either restart or zoom-in
                    Y_fit = rbf_mod(X_all[act_node['ix']])
                    min_ix = np.argmin(Y_fit)
                    x_star = X_all[act_node['ix']][min_ix]
                    # suppose we're going to zoom in
                    child_node_ix = get_child_node(x_star,n_zoom,tree)
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
                                      'state': init_state()}
                    else:
                        # then we activate an existing child node (if zoom in)
                        child_node = tree[n_zoom+1][child_node_ix]
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
                        n_zoom = 0
                        act_node_ix = 0
                        X_all = np.zeros((0,dim))
                        Y_all = np.zeros(0)
                        tree = {n_zoom: [{'ix': np.arange(0,dtype=int), # indice of samples for the node (with respect to X_all and Y_all)
                                          'bd': func_bd, # domain of the node
                                          'parent_ix': None, # parent node index for the upper zoom level (zero-based). If None, there's no parent
                                          'zp': init_beta, # zoom-out probability
                                          'state': init_state() # state of the node
                                          }]}
                    else:
                        # then we zoom in
                        act_node['state'] = init_state() # reset the state of the current node
                        n_zoom += 1
                        
                        if child_node_ix is None:
                            # then we create a new child
                            if n_zoom not in tree.keys():
                                act_node_ix = 0
                                tree[n_zoom] = [child_node]    
                            else:
                                act_node_ix = len(tree[n_zoom])
                                tree[n_zoom].append(child_node)
                                
                            if verbose:
                                print('Zoom in (create a new child node)!')
                                
                        else:
                            # then activate existing child node
                            act_node_ix = child_node_ix
                            # reduce zoom-out probability
                            child_node['zp'] = max(min_beta,child_node['zp']/2.)
                            
                            if verbose:
                                print('Zoom in (activate an existing child node)!')
                                
                if n_proc > 1 or (n_proc == 1 and run_wgt_SRS == wgt_pat_bd[0]):            
                    if np.random.uniform() < tree[n_zoom][act_node_ix]['zp'] and n_zoom > 0 and not full_restart:
                        # then we zoom out
                        child_node = tree[n_zoom][act_node_ix]
                        act_node_ix = child_node['parent_ix']
                        n_zoom -= 1
                        assert(act_node_ix is not None)
                        # check that the node after zooming out will contain the current node
                        assert(intsect_bd(tree[n_zoom][act_node_ix]['bd'],child_node['bd']) == child_node['bd']) 
                        
                        if verbose:
                            print('Zoom out!')
            
            else:
                # i.e., restart
                n_full_restart += 1          
                if n_full_restart == init_iter:
                    full_restart = False
                    n_full_restart = 0
            
            tm2 = default_timer()
            t_misc = tm2-tm1-t_eval_prop[k]-t_save
            
            if verbose:
                print('time for miscellaneous tasks (saving variables, etc.) = %.2e sec' % t_misc)
            
            # find time to prepare for next iteration
            tp2, tp2_c = default_timer(), clock()
            t_prep[k] = tp2-tp1
            t_prep_cpu[k] = tp2_c-tp1_c
            
            # find the time to run optimization algortithm (excluding time to evaluate and save samples)
            tt2 = default_timer()
            tt2_c = clock()
            t_opt[k] = tt2-tt1-t_eval_prop[k]-t_save
            t_opt_cpu[k] = tt2_c-tt1_c-t_eval_prop_cpu[k]-t_save_c
            
            if verbose:
                print('time to run optimization algorithm = %.2e sec' % t_opt[k])
                    
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
                     t_build_mod=t_build_mod[:k+1],t_build_mod_cpu=t_build_mod_cpu[:k+1],t_prop_pt=t_prop_pt[:k+1],
                     t_prop_pt_cpu=t_prop_pt_cpu[:k+1],t_eval_prop=t_eval_prop[:k+1],t_eval_prop_cpu=t_eval_prop_cpu[:k+1],
                     t_opt=t_opt[:k+1],t_opt_cpu=t_opt_cpu[:k+1],t_prep=t_prep[:k+1],t_prep_cpu=t_prep_cpu[:k+1],
                     opt_sm_arr=opt_sm_arr[:k+1],cv_err_arr=cv_err_arr[:k+1],
                     gSRS_pct_arr=gSRS_pct_arr[:k+1],n_zoom_arr=n_zoom_arr[:k+1],
                     best_loc_it=best_loc_it[:init_iter+k+1],
                     best_val_it=best_val_it[:init_iter+k+1], best_seed_it=best_seed_it[:init_iter+k+1],
                     # state variables
                     full_restart=full_restart,n_zoom=n_zoom,act_node_ix=act_node_ix,
                     X_all=X_all,Y_all=Y_all,tree=tree,n_full_restart=n_full_restart,
                     seed_base=seed_base,seed_flat_arr=seed_flat_arr,run_wgt_SRS=run_wgt_SRS,
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
              
        # reset stdout and stderr
        if std_out_file is not None:
            sys.stdout = orig_std_out # set back to original stdout (i.e. to console only)
        if std_err_file is not None:
            sys.stderr = orig_std_err # set back to original stderr (i.e. to console only)
        
        # find best point and its (noisy) function value
        min_ix = np.argmin(best_val_it)
        best_loc = best_loc_it[min_ix]
        best_val = best_val_it[min_ix]
        
        return best_loc, best_val
        
        
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
        n_samp = int(np.ceil(3/float(self._n_worker)))*self._n_worker if n_samp is None else n_samp
        unit_X = lhs(self._dim, samples=n_samp, criterion=criterion) # unit_X: 2d array in unit cube
        samp = np.zeros_like(unit_X)
        for i in range(self._dim):
            samp[:, i] = unit_X[:, i]*(self._domain[i][1]-self._domain[i][0])+self._domain[i][0] # scale and shift
            
        return samp
    