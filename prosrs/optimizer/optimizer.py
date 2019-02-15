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
        # class members (convention: variables wih prefix '_' are constant parameters during optimization)
        self._dim = prob.dim
        self._domain = prob.domain
        self._n_worker = n_worker
    
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
    
    def run(self, n_iter=None, n_restart=2, resume=False, verbose=True):
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
        """
        
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
    