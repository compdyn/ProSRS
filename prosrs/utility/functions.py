"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Define utility functions.
"""
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from timeit import default_timer
from bisect import bisect_left


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
    else:
        unique_arr = array
        
    return unique_arr

 
def put_back_box(pt, domain):
    """
    Put points back to a box-shaped domain, if there are any outside the domain.
    Any outside points are relocated to their nearest points in the domain.
    
    Args:
        
        pt (2d array): Points to be put back.
        
        domain (list of tuples): Box-shaped domain.
                For example, `domain` = [(0, 1), (0, 2)] means defining a 2D domain: 
                with first coordinate in [0, 1] and second coordinate in [0, 2].
        
    Returns:
        
        uniq_pt (2d array): Unique points after being put back (any duplicated points are squashed).
        
        raw_pt (2d array): Original points after being put back (with possible duplicate points).
    """
    n_pt, dim = pt.shape
    assert(len(domain)==dim)
    
    for j in range(dim):
        min_bd = domain[j][0]
        max_bd = domain[j][1]
        assert(max_bd > min_bd)
        if j == 0:
            ind = np.logical_and(pt[:, j] > min_bd,pt[:, j] < max_bd) # indicator that shows if a number is inside the range
        else:
            ind = np.logical_or(ind, np.logical_and(pt[:, j] > min_bd,pt[:, j] < max_bd))
        pt[:, j] = np.maximum(min_bd, np.minimum(max_bd, pt[:, j]))       
    pt_1 = pt[ind]
    pt_2 = pt[np.logical_not(ind)]
    raw_pt = np.vstack((pt_1, pt_2))
    pt_2 = unique_row(pt_2) # find unique points
    uniq_pt = np.vstack((pt_1, pt_2))
    
    return uniq_pt, raw_pt


def scale_zero_one(arr):
    """
    Scale an array to the range [0, 1] with minimum corresponding to 0 
    and maximum to 1.
    
    Args:
        
        arr (array): Array to be scaled.
        
    Returns:
        
        scale_arr (array): Scaled array (same shape as `arr`).
    """
    max_arr, min_arr = np.amax(arr), np.amin(arr)
    if max_arr != min_arr:
        scale_arr = (arr-min_arr)/(max_arr-min_arr)
    else:
        scale_arr = np.ones_like(arr)
        
    return scale_arr


def scale_one_zero(arr):
    """
    Scale an array to the range [0, 1] with minimum corresponding to 1 
    and maximum to 0.
    
    Args:
        
        arr (array): Array to be scaled.
        
    Returns:
        
        scale_arr (array): Scaled array (same shape as `arr`).
    """         
    max_arr, min_arr = np.amax(arr), np.amin(arr)
    if max_arr != min_arr:
        scale_arr = (max_arr-arr)/(max_arr-min_arr)
    else:
        scale_arr = np.ones_like(arr)
        
    return scale_arr


def eff_npt(pt, domain):
    """
    Get effective number of points within in a box-shaped domain.
    The effective number ``n_{eff}`` is computed by first uniformly partitioning the domain
    into cells with the number of cells per dimension being equal to ``ceil(n^{1/d})``, 
    where ``n`` is the number of points. Then ``n_{eff}`` is the number of cells that 
    are occupied by at least one point.
    
    Args:
        
        pt (2d array): Points in the domain.
        
        domain (list of tuples): Box-shaped domain.
            For example, `domain` = [(0, 1), (0, 2)] means defining a 2D domain: 
            with first coordinate in [0, 1] and second coordinate in [0, 2].
                
    Returns:
        
        n_eff (int): Effective number of points.
    """
    npt, dim = pt.shape
    assert(dim == len(domain))
    m = int(np.ceil(npt**(1./dim)))
    assert(m>1)
    # find grid for each axis
    grid_arr = np.zeros((dim, m+1))
    for j, axis_bd in enumerate(domain):
        grid_arr[j] = np.linspace(axis_bd[0], axis_bd[1], num=m+1)
    # find location of each sample (save in string)
    loc_samp = ['']*npt
    for i in range(npt):
        for j, grid in enumerate(grid_arr):
            assert(grid[0] <= pt[i, j] <=grid[-1])
            ix = bisect_left(grid, pt[i, j])
            assert(ix < m+1)
            loc = 0 if ix == 0 else ix-1
            loc_samp[i] += str(loc)
    # find number of samples with unique location
    n_eff = len(set(loc_samp))        
    assert(n_eff <= npt)
    
    return n_eff


def boxify(pt, domain):
    """
    Check whether each point is in the box-shaped domain.

    Args:
        
        pt (2d array): Points.
        
        domain (list of tuples): Box-shaped domain.
            For example, `domain` = [(0, 1), (0, 2)] means defining a 2D domain: 
            with first coordinate in [0, 1] and second coordinate in [0, 2].
    Returns:
        
        in_box_ix (1d array of bool): Indicate whether each point in `pt` is in the box.
        
        out_box_ix (1d array of bool): Indicate whether each point in `pt` is out of the box.
    """
    npt, dim = pt.shape
    for i in range(dim):
        assert(domain[i][0] < domain[i][1])
        ix = np.logical_and(domain[i][0] <= pt[:, i], domain[i][1] >= pt[:, i])
        if i == 0:
            in_box_ix = ix # indicates whether the points are in the box
        else:
            in_box_ix = np.logical_and(in_box_ix,ix)
    out_box_ix = np.logical_not(in_box_ix)
    
    return in_box_ix, out_box_ix


def domain_intersect(domain1, domain2):
    """
    Find intersection of two domains.
    
    Args:
        
        domain1 (list of tuples): First box-shaped domain.
            For example, `domain1` = [(0, 1), (0, 2)] means defining a 2D domain: 
            with first coordinate in [0, 1] and second coordinate in [0, 2].
        
        domain2 (list of tuples): Second box-shaped domain.
            For example, `domain2` = [(0, 1), (0, 2)] means defining a 2D domain: 
            with first coordinate in [0, 1] and second coordinate in [0, 2].
            
    Returns:
        
        inter_domain (list of tuples or None): Intersection of `domain1` and `domain2`. 
            If None, then the intersection is empty.
    """
    lb1, ub1 = zip(*domain1)
    lb2, ub2 = zip(*domain2)
    lb1, ub1 = np.array(lb1), np.array(ub1)
    lb2, ub2 = np.array(lb2), np.array(ub2)
    assert(np.all(ub1>=lb1) and np.all(ub2>=lb2))
    assert(lb1.shape==ub1.shape==lb2.shape==ub2.shape)
    inter_lb = np.maximum(lb1,lb2)
    inter_ub = np.minimum(ub1,ub2)
    if np.all(inter_ub>inter_lb):
        inter_domain = list(zip(inter_lb, inter_ub))
    else:
        inter_domain = None
        
    return inter_domain