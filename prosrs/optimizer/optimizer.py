"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Implement ProSRS algorithm.
"""
import numpy as np
import os, sys, pickle, shutil, warnings
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from timeit import default_timer
from pyDOE import lhs
from scipy.spatial.distance import cdist
from ..utility.constants import STATE_NPZ_FILE_TEMP, STATE_PKL_FILE_TEMP, STATE_NPZ_TEMP_FILE_TEMP, STATE_PKL_TEMP_FILE_TEMP
from ..utility.classes import std_out_logger, std_err_logger
from ..utility.functions import eval_func, put_back_box, scale_one_zero, scale_zero_one, eff_npt, boxify, domain_intersect
from .surrogate import RBF_reg


class Optimizer:
    """
    A class that handles optimization using ProSRS algorithm.
    """
    def __init__(self, prob, n_worker, n_iter=None, n_iter_doe=None, n_restart=2, resume=False, 
                 seed=1, seed_func=None, parallel_training=False, out_dir='out'):
        """
        Constructor.
        
        Args:
            
            prob (`prosrs.Problem` type): Optimization problem.
            
            n_worker (int): Number of workers for the optimization.
                This also determines the number of proposed or evaluated
                points per iteration.
                
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
            
            parallel_training (bool, optional): Whether to train a RBF surrogate in parallel.
                If True, then we will use all the available processes (cores) during training.
                We use `pathos.multiprocessing` module for parallelism. Our tests have
                shown that depending on the machine and the optimization function,
                sometimes parallelism may cause memory issues. So we disable it by default.
                
            out_dir (str, optional): Output directory.
                All the output files (e.g., optimization status file) will be saved to
                `out_dir` directory.
        """
        # class members (convention: variables wih prefix '_' are constant parameters during optimization).
        self._prob = prob # optimization problem
        self._dim = prob.dim # dimention of optimization problem.
        self._n_worker = n_worker # number of workers.
        self._n_iter = n_iter # total number of iterations.
        self._n_restart = n_restart # total number of restarts.
        self._resume = resume # whether to resume from the last run.
        self._seed = seed # random seed for the optimizer.
        self._seed_func = seed_func # function for setting random seed for evaluations.
        self._parallel_training = parallel_training # whether to use parallelism for training RBF models.
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
        self._use_eff_npt = True # whether to use effective number of points for the dynamics of p value.
        self._max_C_fail = max(2, int(np.ceil(self._dim/float(self._n_worker)))) # maximum number of consecutive failures before halving self.sigma value.
        self._n_iter_doe = int(np.ceil(3/float(self._n_worker))) if n_iter_doe is None else n_iter_doe # number of iterations for DOE.
        self._n_iter_doe = min(self._n_iter, self._n_iter_doe) if self._n_iter is not None else self._n_iter_doe # adjusted for the fact that self._n_iter_doe <= self._n_iter.
        self._n_doe_samp = self._n_iter_doe * self._n_worker # number of DOE samples
        self._n_cand = self._n_cand_fact * self._dim # number of candidate points in SRS method
        self._pool_rbf = Pool(nodes=mp.cpu_count()) if self._parallel_training else None # parallel pool used for RBF regression
        self._state_npz_file = os.path.join(self._out_dir, STATE_NPZ_FILE_TEMP % self._prob.name) # file that saves optimizer state (needed for resume)
        self._state_pkl_file = os.path.join(self._out_dir, STATE_PKL_FILE_TEMP % self._prob.name) # file that saves optimizer state (needed for resume)
        self._state_npz_lock_file = self._state_npz_file+'.lock' # a lock file that may be generated in some system, which prevents reading data from `self._state_npz_file`.
        self._state_pkl_lock_file = self._state_pkl_file+'.lock' # a lock file that may be generated in some system, which prevents reading data from `self._state_pkl_file`.
        self._state_npz_temp_file = os.path.join(self._out_dir, STATE_NPZ_TEMP_FILE_TEMP % self._prob.name) # a temporary file that holds data for `self._state_npz_file`.
        self._state_pkl_temp_file = os.path.join(self._out_dir, STATE_PKL_TEMP_FILE_TEMP % self._prob.name) # a temporary file that holds data for `self._state_pkl_file`.
        
        # sanity check
        assert(type(self._n_worker) is int and self._n_worker > 0)
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
            np.random.seed(self._seed) # set random seed.
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
            self.x_tree = np.zeros((0, self._dim)) # all the evaluated (proposed) points so far in a tree.
            self.y_tree = np.zeros(0) # (noisy) y values of `self.x_tree`.
            self.x_all = np.zeros((0, self._dim)) # all the evaluated (proposed) points so far in the course of optimization.
            self.y_all = np.zeros(0) # (noisy) y values of `self.x_all`.
            self.seed_all = np.zeros(0) # random seeds for points in `self.x_all`.
            self.best_x = np.ones(self._dim)*np.nan # best point so far.
            self.best_y = np.nan # (noisy) y value of the best point `self.best_x`.
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
    
    def run(self, std_out_file=None, std_err_file=None, verbose=True):
        """
        Run ProSRS algorithm.
        
        Args:
            
            std_out_file (str or None, optional): Standard output file path.
                If ``str``, then standard outputs will be directed to the file `std_out_file`.
                If None, then standard outputs will not be saved to a file.
            
            std_err_file (str or None, optional): Standard error file path.
                If ``str``, then standard errors will be directed to the file `std_err_file`.
                If None, then standard errors will not be saved to a file.
                
            verbose (bool, optional): Whether to verbose while running the algorithm.     
        """ 
        # FIXME: check verbose           
        # log standard outputs and standard errors.
        # here we write to a new file if we do not resume. Otherwise, we append to the old file.
        if std_out_file is not None:
            orig_std_out = sys.stdout
            if not self._resume:
                if os.path.isfile(std_out_file):
                    os.remove(std_out_file)
            sys.stdout = std_out_logger(std_out_file)    
        if std_err_file is not None:
            orig_std_err = sys.stderr
            if not self._resume:
                if os.path.isfile(std_err_file):
                    os.remove(std_err_file)
            sys.stderr = std_err_logger(std_err_file)
        
        # main loop
        while not self.is_done():
            
            if verbose:
                print('\nIteration = %d' % (self.i_iter+1))
                
            # propose new points
            new_pt = self.propose(verbose=verbose)        
            # evaluate proposed points
            new_val = self.eval_pt(new_pt, verbose=verbose)
                
            # update optimizer state with the new evaluations
            self.update(new_pt, new_val, verbose=verbose)
            
            # flush standard outputs and standard errors to files
            if std_out_file is not None:
                sys.stdout.terminal.flush()
                sys.stdout.log.flush()
            if std_err_file is not None:
                sys.stderr.terminal.flush()
                sys.stderr.log.flush()
        
        # reset stdout and stderr
        if std_out_file is not None:
            sys.stdout = orig_std_out 
        if std_err_file is not None:
            sys.stderr = orig_std_err
        
        
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
        tree = {self.zoom_lv: [{'ix': np.arange(self._n_doe_samp, dtype=int), # indice of data for the tree node (w.r.t. `self.x_tree` or `self.y_tree`).
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
            self.x_node = self.x_tree[self.act_node['ix']] # get X data of the node
            self.y_node = self.y_tree[self.act_node['ix']] # get Y data of the node
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
                                                 n_fold=self._n_fold, kernel=self._rbf_kernel, 
                                                 poly_deg=self._rbf_poly_deg, pool=self._pool_rbf)                
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
            refer_pt = np.vstack((refer_pt, new_pt[j].reshape((1, -1))))
            dist_cand = np.delete(dist_cand, min_ix)
            resp_score = np.delete(resp_score, min_ix)
            cand_pt = np.delete(cand_pt, min_ix, axis=0)
            n_cand -= 1
        
        return new_pt
        
    
    def eval_pt(self, x, verbose=True):
        """
        Evaluate proposed points.
        
        Args:
            
            x (2d array): Points to be evaluated. Each row is one point.
            
            verbose (bool, optional): Whether to verbose about the evaluation.
        
        Returns:
            
            y (1d array): Evaluations of points in `x`.
        """
        # FIXME: check verbose
        
        t1 = default_timer()
        
        self.eval_seeds = self._seed+1+np.arange(self.i_iter*self._n_worker, (self.i_iter+1)*self._n_worker, dtype=int)
        y = eval_func(self._prob.f, x, n_proc=self._n_worker, seeds=self.eval_seeds.tolist(),
                      seed_func=self._seed_func)
        
        t2 = default_timer()
        self.t_eval = t2-t1
        
        if verbose:
            print('time to evaluate points = %.2e sec' % self.t_eval)
        
        return y
    
    
    def update(self, new_x, new_y, verbose=True):
        """
        Update the state of the optimizer.
        
        Args:
            
            new_x (2d array): Proposed new points. Each row is one point.
            
            new_y (1d array): (Noisy) values of the points in `new_x`.
            
            verbose (bool, optional): Whether to verbose about updating the state of the optimizer.
        """
        # FIXME: check verbose        
        t1 = default_timer()
        
        self.i_iter += 1
        self.x_tree = np.vstack((self.x_tree, new_x))
        self.y_tree = np.append(self.y_tree, new_y)
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
            npt = len(self.x_tree)
            self.act_node['ix'] = np.append(self.act_node['ix'], np.arange(npt-self._n_worker, npt, dtype=int))
            if self._n_worker > 1 or (self._n_worker == 1 and self.srs_wgt_pat[0] == self._wgt_pat_bd[0]):
                if self.p_val >= 0.1:
                    # compute p_val
                    if self._use_eff_npt:
                        eff_n = eff_npt(self.x_tree[self.act_node['ix']], self.act_node['domain'])
                    else:
                        eff_n = len(self.x_tree[self.act_node['ix']])
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
                Y_fit = self.rbf_mod(self.x_tree[self.act_node['ix']])
                min_ix = np.argmin(Y_fit)
                x_star = self.x_tree[self.act_node['ix']][min_ix]
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
                    child_node = {'ix': np.nonzero(boxify(self.x_tree, domain)[0])[0], 
                                  'domain': domain,
                                  'parent_ix': self.act_node_ix,
                                  'beta': self._init_beta,
                                  'state': self.init_node_state()}
                else:
                    # then we activate an existing child node (if zoom in)
                    child_node = self.tree[self.zoom_lv+1][child_node_ix]
                    child_node['ix'] = np.nonzero(boxify(self.x_tree, child_node['domain'])[0])[0]
                child_npt = len(child_node['ix'])
                domain_lb, domain_ub = zip(*child_node['domain'])
                blen = np.array(domain_ub)-np.array(domain_lb) # bound length for each dimension
                assert(np.min(blen)>0)
                
                if np.all(blen*child_npt**(-1./self._dim) < (self._prob.domain_ub-self._prob.domain_lb)*self._resol): # resolution condition
                    # then we restart
                    if verbose:
                        print('Restart for the next iteration!')
                    self.i_iter_doe = 0
                    
                    # FIXME: debugging only (to recover the original, uncomment the following)
#                    self.doe_samp = self.doe()
                    
                    self.i_restart += 1
                    self.zoom_lv = 0
                    self.act_node_ix = 0
                    self.x_tree = np.zeros((0, self._dim))
                    self.y_tree = np.zeros(0)
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
                
                # FIXME: debugging only (remove the following to recover the original)
                elif self.i_iter_doe == 0:
                    self.doe_samp = self.doe()
                    
                    
        
        t2 = default_timer()
        t_update = t2-t1
        self.t_update_arr = np.append(self.t_update_arr, t_update)
        if verbose:
            print('time to update optimizer = %.2e sec' % t_update)
            
        self.save_state()
   
     
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
        # save state to pickle file
        with open(self._state_pkl_temp_file, 'wb') as f:
            pickle.dump(np.random.get_state(), f) # first save to temporary file, preventing data loss due to termination during execution of `pickl.dump`
        shutil.copy2(self._state_pkl_temp_file, self._state_pkl_file) # create a new or overwrite the old `self._state_pkl_file`
        os.remove(self._state_pkl_temp_file) # remove temporary file
        
        # save state to npz file
        np.savez(self._state_npz_temp_file,
                 # constant parameters
                 _dim=self._dim, _n_worker=self._n_worker, _n_iter=self._n_iter, _n_restart=self._n_restart, 
                 _resume=self._resume, _seed=self._seed, _parallel_training=self._parallel_training, _out_dir=self._out_dir, 
                 _n_cand_fact=self._n_cand_fact, _wgt_pat_bd=self._wgt_pat_bd, _normalize_data=self._normalize_data,
                 _init_gamma=self._init_gamma, _delta_gamma=self._delta_gamma, _init_sigma=self._init_sigma, 
                 _max_n_reduce_sigma=self._max_n_reduce_sigma, _rho=self._rho, _init_p=self._init_p,
                 _init_beta=self._init_beta, _min_beta=self._min_beta, _alpha=self._alpha, _lambda_range=self._lambda_range,
                 _rbf_kernel=self._rbf_kernel, _rbf_poly_deg=self._rbf_poly_deg, _n_fold=self._n_fold, _resol=self._resol,
                 _use_eff_npt=self._use_eff_npt, _max_C_fail=self._max_C_fail, _n_iter_doe=self._n_iter_doe,
                 _n_doe_samp=self._n_doe_samp, _n_cand=self._n_cand, _state_npz_file=self._state_npz_file,
                 _state_pkl_file=self._state_pkl_file, _state_npz_lock_file=self._state_npz_lock_file,
                 _state_pkl_lock_file=self._state_pkl_lock_file, _state_npz_temp_file=self._state_npz_temp_file,
                 _state_pkl_temp_file=self._state_pkl_temp_file,
                 # state variables
                 i_iter=self.i_iter, i_restart=self.i_restart, doe_samp=self.doe_samp, i_iter_doe=self.i_iter_doe,
                 t_build_arr=self.t_build_arr, t_srs_arr=self.t_srs_arr, t_prop_arr=self.t_prop_arr,
                 t_eval_arr=self.t_eval_arr, t_update_arr=self.t_update_arr, gSRS_pct_arr=self.gSRS_pct_arr,
                 zoom_lv_arr=self.zoom_lv_arr, x_tree=self.x_tree, y_tree=self.y_tree, x_all=self.x_all,
                 y_all=self.y_all, seed_all=self.seed_all, best_x=self.best_x, best_y=self.best_y, 
                 zoom_lv=self.zoom_lv, act_node_ix=self.act_node_ix, srs_wgt_pat=self.srs_wgt_pat, tree=self.tree)
        
        shutil.copy2(self._state_npz_temp_file, self._state_npz_file)
        os.remove(self._state_npz_temp_file) # remove temporary file
   
    
    def load_state(self):
        """
        Load the state of the optimizer from files.
        """        
        # load state data from pkl file
        if os.path.isfile(self._state_pkl_lock_file):
            os.remove(self._state_pkl_lock_file) # remove lock file, if there's any
        with open(self._state_pkl_file, 'rb') as f:
            np.random.set_state(pickle.load(f))    
        # load state data from npz file
        if os.path.isfile(self._state_npz_lock_file):
            os.remove(self._state_npz_lock_file) # remove lock file, if there's any
        data = np.load(self._state_npz_file)
        # check consistency
        assert(self._dim==data['_dim'] and self._n_worker==data['_n_worker'] and self._n_cand_fact==data['_n_cand_fact']
               and np.all(self._wgt_pat_bd==data['_wgt_pat_bd']) and self._normalize_data==data['_normalize_data']
               and self._init_gamma==data['_init_gamma'] and self._delta_gamma==data['_delta_gamma']
               and self._init_sigma==data['_init_sigma'] and self._max_n_reduce_sigma==data['_max_n_reduce_sigma']
               and self._rho==data['_rho'] and self._init_p==data['_init_p'] and self._init_beta==data['_init_beta']
               and self._min_beta==data['_min_beta'] and self._alpha==data['_alpha']
               and np.all(self._lambda_range==data['_lambda_range']) and np.all(self._rbf_kernel==data['_rbf_kernel'])
               and self._rbf_poly_deg==data['_rbf_poly_deg'] and self._n_fold==data['_n_fold'] and self._resol==data['_resol']
               and self._use_eff_npt==data['_use_eff_npt'] and self._max_C_fail==data['_max_C_fail']
               and self._n_cand==data['_n_cand']), 'To resume, experiment condition needs to be consistent with that of the last run.'
        # read state variables
        self.i_iter = data['i_iter'].item(0)
        self.i_restart = data['self.i_restart'].item(0)
        self.doe_samp = data['doe_samp']
        self.i_iter_doe = data['i_iter_doe'].item(0)
        self.t_build_arr = data['t_build_arr']
        self.t_srs_arr = data['t_srs_arr']
        self.t_prop_arr = data['t_prop_arr']
        self.t_eval_arr = data['t_eval_arr']
        self.t_update_arr = data['t_update_arr']
        self.gSRS_pct_arr = data['gSRS_pct_arr']
        self.zoom_lv_arr = data['zoom_lv_arr']
        self.x_tree = data['x_tree']
        self.y_tree = data['y_tree']
        self.x_all = data['x_all']
        self.y_all = data['y_all']
        self.seed_all = data['seed_all']
        self.best_x = data['best_x']
        self.best_y = data['best_y'].item(0)
        self.zoom_lv = data['zoom_lv'].item(0)
        self.act_node_ix = data['act_node_ix'].item(0)
        self.srs_wgt_pat = data['srs_wgt_pat']
        self.tree = data['tree'].item(0)        
            
        
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
        # TODO
    
    
    def posterior_eval(self, n_top=0.1, n_repeat=10, n_worker=None, seed=1, seed_func=None, verbose=True):
        """
        Posterior Monte Carlo evaluations for selecting the true best point.
        
        Args:
            
            n_top (float or int, optional): Proportion/number of top points to be evaluated.
                If ``float``, then `n_top` in (0, 1) is the proportion of top points
                among all the evaluated points. If ``int``, then `n_top`, a positive
                integer, is the number of top points to be evaluated.
            
            n_repeat (int, optional): Number of Monte Carlo repeats for the evaluations.
            
            n_worker (int or None, optional): Number of workers for the evaluations.
                Used for parallel computing. If None, then we use the default value = 
                `self._n_worker`. Here we assume parallelism across multiple cores
                in a machine and using Python package ``pathos``.
            
            seed (int, optional): (Base) random seed for Monte Carlo evaluations.
                For a point with point index i (zero-based) and repeat index j 
                (zero-based), the seed for the point = `seed`+i*`n_repeat`+j.
            
            seed_func (callable or None, optional): User-specified function for setting the random seed.
                If None, then we use ``numpy.random.seed`` method to set random seed.
                If callable, it is a function taking ``seed`` as an argument.
                Note: using ``numpy.random.seed`` may not always gaurantee
                the reproducibility. Here we provide an option for users to specify their own routines.
            
            verbose (bool, optional): Whether to verbose about posterior evaluations.
        
        Returns:
            
            top_pt (2d array): Evaluated top points. 
                Each row is one point.
            
            top_mean_val (1d array): Mean estimates of function values of `top_pt`.
                In essense, these are estimates for the true expectation ``self._prob.F(top_pt)``.
            
            top_std_val (1d array): Standard deviation estimates of function values of top_pt`.
                In essence, these are std estimates for the random noises in ``self._prob.f``
                at `top_pt`.
                
            top_val (2d array): (Noisy) function values at `top_pt`. 
                Each row is `n_repeat` Monte Carlo evaluations of one point
                (i.e., ``top_eval.shape[1] = n_repeat``).
        """
        # FIXME: add verbose
        assert(type(n_top) in [float, int])
        assert(type(n_repeat) is int and n_repeat > 0)
        assert(type(seed) is int and seed >= 0)
        n_all = len(self.x_all)
        assert(n_all > 0), 'No points have been evaluated yet. Run the optimization algorithm first.'
        if type(n_top) is float:
            assert(0 < n_top < 1)
            n_top = max(1, int(round(n_top*n_all)))
        else:
            assert(n_top > 0)
            if n_top > n_all:
                warnings.warn('Specified n_top is larger than the number of evaluations. Reduce n_top to the number of evaluations.')
                n_top = n_all
        n_worker = self._n_worker if n_worker is None else n_worker
        assert(type(n_worker) is int and n_worker > 0)
        # get top points and evaluate them
        sort_ix = np.argsort(self.y_all)  
        top_pt = self.x_all[sort_ix][:n_top]
        top_pt_mc = top_pt[np.outer(range(n_top), np.ones(n_repeat, dtype=int)).flatten()] # duplicate each point in `top_pt` for `n_repeat` times
        top_seed_mc = seed+np.arange(n_top*n_repeat, dtype=int)
        assert(len(top_pt_mc) == len(top_seed_mc))
        top_val_mc = eval_func(self._prob.f, top_pt_mc, n_proc=n_worker, 
                               seeds=top_seed_mc.tolist(), seed_func=seed_func)
        top_val = top_val_mc.reshape((n_top, n_repeat))
        top_mean_val = np.mean(top_val, axis=1)
        top_std_val = np.std(top_val, axis=1, ddof=1)
        
        return top_pt, top_mean_val, top_std_val, top_val
        
