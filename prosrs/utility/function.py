"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Defines utility functions.
"""
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

def eval_func(func, pts, n_proc=1):
    """
    Evaluate function in serial/parallel.
    
    Here we assume that parallelism takes place across multiple cores of a machine,
    and the Python package `pathos <https://pypi.org/project/pathos/0.2.0>` has 
    been installed.
    
    Args:
        
        func (callable): Function to be evaluated, of which the input is 1d array,
            the output is a scalar.
        
        pts (2d array): Points to be evaluated. Each point is one row of `pts`.
        
        n_proc (int, optional): number of processes used for evaluation. If
            = 1, we evaluate `pts` in serial.
    
    Returns:
        
        vals (1d array): Function values at the points `pts`.
    
    Raises:
        
        ValueError: If n_proc is less than 1.
    """
    assert(callable(func)), 'func is not callable.'
    if n_proc == 1:
        vals = np.array([func(x) for x in pts])  
    elif n_proc > 1:
        pool = Pool(processes=int(n_proc))
        vals = np.array(pool.map(func, pts))
    else:
        raise ValueError('Invalid n_proc value.')
    assert(vals.ndim == 1), 'vals should be 1d numpy array.'
       
    return vals