"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Define utility functions.
"""
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from timeit import default_timer


def eval_func(func, pts, n_proc=1, seeds=None, seed_func=None, save_files=None):
    """
    Evaluate a function in serial/parallel.
    
    Here we assume that parallelism takes place across multiple cores of a machine,
    and the Python package `pathos <https://pypi.org/project/pathos/0.2.0>` has 
    been installed.
    
    Args:
        
        func (callable): Function to be evaluated, of which the input is 1d array,
            the output is a scalar.
        
        pts (2d array): Points to be evaluated. Each point is one row of `pts`.
        
        n_proc (int, optional): Number of processes used for evaluation. If
            = 1, we evaluate `pts` in serial.
            
        seeds (list or None, optional): Random seed for each point in `pts`.
            If None, then no random seed will be set for any evaluation.
        
        seed_func (callable or None, optional): User-specified function for setting the random seed.
            If None, then we use ``numpy.random.seed`` method to set random seed.
            If callable, it is a function taking ``seed`` as an argument.
            Note: using ``numpy.random.seed`` may not always gaurantee
            the reproducibility. Here we provide an option for users to specify their own routines.
        
        save_files (list or None): Specify ``.npz`` file where each evaluation will be saved.
            If None, then no evaluation will be saved. It may be helpful to save evaluations
            for debugging purposes.
    
    Returns:
        
        vals (1d array): Function values at the points `pts`.
    
    Raises:
        
        ValueError: If n_proc is less than 1.
    """ 
    def eval_wrapper(arg):
        """
        A wrapper for evaluating the function `func`.
        
        Args:
            
            arg (tuple): Function argument = (pt, seed, save_file) with
            
                pt (1d array): A point to be evaluated.
            
                seed (int or None): Random seed for the evaluation. 
                    If None, then no random seed will be set for the evaluation.
            
                save_file (str or None): Specify ``.npz`` file where the evaluation will be saved.
                    If None, then the evaluation will not be saved.
        
        Returns:
            
            val (float): Function value at the point `pt`.
        """
        # parse argument
        pt, seed, save_file = arg
        
        if seed is not None:
            # set random seed
            if seed_func is None:
                np.random.seed(seed)
            else:
                assert(callable(seed_func)), 'seed_func is not callable.'
                seed_func(seed)
        
        t1 = default_timer()
        val = func(pt)
        t2 = default_timer()
        eval_time = t2-t1 # elapsed time for the evaluation
        
        if save_file is not None:
            np.savez(save_file, pt=pt, val=val, seed=seed, eval_time=eval_time)
        
        return val
        
    n_pts = pts.shape[0]
    seeds = [None]*n_pts if seeds is None else seeds
    save_files = [None]*n_pts if save_files is None else save_files
    # sanity check
    assert(len(seeds) == len(save_files) == n_pts)
    assert(callable(func)), 'func is not callable.'
    
    if n_proc == 1:
        vals = np.array([eval_wrapper(x) for x in zip(pts, seeds, save_files)])
    elif n_proc > 1:
        assert(type(n_proc) is int)
        pool = Pool(nodes=n_proc)
        vals = np.array(pool.map(eval_wrapper, list(zip(pts, seeds, save_files))))
    else:
        raise ValueError('Invalid n_proc value.')
    # sanity check
    assert(vals.ndim == 1), 'vals should be 1d numpy array.'
       
    return vals


def unique_row(array):
    """
    Sort a 2d array row-wise and get unique rows of it.
    
    Note: experiments show that this function produces results faster than ``np.unique`` method,
    especially when the number of rows of the array is large.
    
    Args:
        
        array (2d array): Array to get unique rows from
    
    Returns:
    
        unique_arr (2d array): Sorted array of unique rows. If `array` is empty (i.e.,
            ``array.shape = (0,0)``), then the empty `array` will be returned.
    """
    # sanity check
    assert(array.ndim == 2)
    
    if array.size > 0: # i.e. non-empty        
        
        sorted_arr =  array[np.lexsort(array.T),:]
        # get unique rows
        row_ix = np.append([True], np.any(np.diff(sorted_arr, axis=0), axis=1))
        unique_arr = sorted_arr[row_ix]
        
    return unique_arr
 
