"""
Copyright (C) 2016-2018 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

This code implements Progressive Stochastic Response Surface (ProSRS) algorithm.
"""

from __future__ import division, absolute_import

import sys
import numpy as np
from sklearn.model_selection import KFold
from pyDOE import lhs
import os
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from timeit import default_timer
from time import clock
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from scipy.io import netcdf
from scipy import linalg
from scipy.special import xlogy
import pickle
import bisect
import warnings
import shutil

########################### Global Constants ####################

# output file template
RESULT_NPZ_FILE_TEMP = 'optim_result_%s.npz'
RESULT_PKL_FILE_TEMP = 'optim_result_%s.pkl'
RESULT_GENERAL_SAMP_FILE_TEMP = 'samp_%s_t%d.nc'
RESULT_SAMP_FILE_TEMP = 'samp_%s_t%d_p%d.nc'
TEMP_RESULT_NPZ_FILE_SUFFIX = '.temp.npz'
# MPI constants
MPI_tag1 = 77 # MPI message tag for the first communication between master and worker
MPI_tag2 = 133 # MPI message tag for the second communication between master and worker

###################### Classes & Functions #######################

class Rbf(object):
    """
    This class implements regularized RBF regression.
    The class is modified based on scipy.interpolate.Rbf (scipy version: 0.19.0)
    
    Copyright (c) 2016-2018 Chenchao Shou
    
    rbf - Radial basis functions for interpolation/smoothing scattered Nd data.
    Written by John Travers <jtravs@gmail.com>, February 2007
    Based closely on Matlab code by Alex Chirokov
    Additional, large, improvements by Robert Hetland
    Some additional alterations by Travis Oliphant
    Permission to use, modify, and distribute this software is given under the
    terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
    this distribution for specifics.
    NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
    Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
    Copyright (c) 2007, John Travers <jtravs@gmail.com>

    A class for radial basis function approximation/interpolation of
    n-dimensional scattered data using L2 regularization.
    Parameters
    ----------
    *args : arrays
        x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
        and d is the array of values at the nodes
    function : str, optional
        The radial basis function, based on the radius, r, given by the norm
        (default is Euclidean distance); the default is 'multiquadric'::
            'multiquadric': sqrt((r/self.epsilon)**2 + 1)
            'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
            'gaussian': exp(-(r/self.epsilon)**2)
            'linear': r
            'cubic': r**3
            'quintic': r**5
            'thin_plate': r**2 * log(r)
    epsilon : float, optional
        Adjustable constant for gaussian or multiquadrics functions
        - defaults to approximate average distance between nodes (which is
        a good start).
    smooth : float, optional
        Values greater than zero increase the smoothness of the
        approximation.  0 is for interpolation (default), the function will
        always go through the nodal points in this case.
    norm : callable, optional
        A function that returns the 'distance' between two points, with
        inputs as arrays of positions (x, y, z, ...), and an output as an
        array of distance.  E.g, the default::
            def euclidean_norm(x1, x2):
                return sqrt( ((x1 - x2)**2).sum(axis=0) )
        which is called with x1=x1[ndims,newaxis,:] and
        x2=x2[ndims,:,newaxis] such that the result is a matrix of the
        distances from each point in x1 to each point in x2.
    use_scipy_rbf: boolean, optional
        Whether to use scipy rbf. Default: False
    
    """

    def _euclidean_norm(self, x1, x2):
        return np.sqrt(((x1 - x2)**2).sum(axis=0))

    def _h_multiquadric(self, r):
        return np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_inverse_multiquadric(self, r):
        return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_gaussian(self, r):
        return np.exp(-(1.0/self.epsilon*r)**2)

    def _h_linear(self, r):
        return r

    def _h_cubic(self, r):
        return r**3

    def _h_quintic(self, r):
        return r**5

    def _h_thin_plate(self, r):
        return xlogy(r**2, r)

    # Setup self._function and do smoke test on initial r
    def _init_function(self, r):
        if isinstance(self.function, str):
            self.function = self.function.lower()
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            func_name = "_h_" + self.function
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
            else:
                functionlist = [x[3:] for x in dir(self) if x.startswith('_h_')]
                raise ValueError("function must be a callable or one of " +
                                     ", ".join(functionlist))
            self._function = getattr(self, "_h_"+self.function)

        a0 = self._function(r)
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of the same shape")
        return a0

    def __init__(self, *args, **kwargs):
        self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten()
                           for a in args[:-1]])
        self.N = self.xi.shape[-1]
        self.di = np.asarray(args[-1]).flatten()

        if not all([x.size == self.di.size for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop('norm', self._euclidean_norm)
        r = self._call_norm(self.xi, self.xi)
        self.epsilon = kwargs.pop('epsilon', None)
        if self.epsilon is None:
            ximax = np.amax(self.xi, axis=1)
            ximin = np.amin(self.xi, axis=1)
            edges = ximax-ximin
            edges = edges[np.nonzero(edges)]
            self.epsilon = np.power(np.prod(edges)/self.N, 1.0/edges.size)
        self.smooth = kwargs.pop('smooth', 0.0)
        self.function = kwargs.pop('function', 'multiquadric')        
        
        assert(self.smooth > 0),'smooth parameter needs to be positive'
        self.use_scipy_rbf = kwargs.pop('use_scipy_rbf',False)
        self.wgt = kwargs.pop('wgt',np.ones(self.N)) # weight vector        
        assert ((len(self.wgt) == self.N) and np.min(self.wgt)>=0),'invalid weight vector'
        W = np.diag(self.wgt) # construct weight matrix
        self.deg = kwargs.pop('deg',0) # degree of polynomials (if = 0, then no polynomial tail)
        assert(self.deg in [0,1]), 'currently we only support degree up to 1'
        
        with warnings.catch_warnings():
            # We ignore possible condition number warnings when doing linalg.solve.
            warnings.simplefilter("ignore")
            # attach anything left in kwargs to self
            #  for use by any user-callable function or
            #  to save on the object returned.
            for item, value in kwargs.items():
                setattr(self, item, value)
            if self.use_scipy_rbf:
                self.A = self._init_function(r) - np.eye(self.N)*self.smooth
                self.nodes = linalg.solve(self.A, self.di)
            else:
                phi = self._init_function(r)
                assert(phi.shape == (self.N, self.N))
                if self.deg == 1:
                   phi = np.hstack((phi,self.xi.T))
                   phi = np.hstack((phi,np.ones((self.N,1))))
                self.A = np.dot(np.dot(phi.T,W),phi)+np.eye(phi.shape[1])*self.smooth
                self.b = np.dot(np.dot(phi.T,W),self.di)
                # since smooth parameter is positive, A must be positive definite.
                # so we can set sym_pos=True when solving with linalg.solve
                self.nodes = linalg.solve(self.A,self.b,sym_pos=True,lower=True)

    def _call_norm(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = x1[np.newaxis, :]
        if len(x2.shape) == 1:
            x2 = x2[np.newaxis, :]
        x1 = x1[..., :, np.newaxis]
        x2 = x2[..., np.newaxis, :]
        return self.norm(x1, x2)

    def __call__(self, *args):
        args = [np.asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        shp = args[0].shape
        xa = np.asarray([a.flatten() for a in args], dtype=np.float_)        
        r = self._call_norm(xa, self.xi) 
        if self.deg == 0:
            y_val = np.dot(self._function(r), self.nodes).reshape(shp)
        elif self.deg == 1:
            nx = xa.shape[1] # number of points to be evaluated
            phi = self._function(r)
            assert(phi.shape == (nx, self.N) and self.xi.shape[0] == xa.shape[0])
            phi = np.hstack((phi,xa.T))
            phi = np.hstack((phi,np.ones((nx,1))))
            y_val = np.dot(phi, self.nodes).reshape(shp)
        return y_val
    
# RBF regression
def RBF_reg(X,Y,n_proc,sm_range,normalize_data=True,wgt_expon=0.0,use_scipy_rbf=False,
            log_opt=True,n_fold=5,n_min_sm=10,kernel='multiquadric',pool=None,poly_deg=0):
    '''
    Input:
    X: x values of data points (2d array, different rows are different points)
    Y: y values of data points (1d array)
    n_proc: number of processors (cores)
    sm_range: range of smooth parameter ([low,upper]) or a single smooth parameter in a list
    normalize_data: boolean. Whether to normalize data before training
    wgt_expon: scalar, weight exponent used in building weighted rbf, useful ONLY when use_scipy_rbf = False
    log_opt: whether to optimize the smooth parameter on a log scale (boolean)
    n_fold: number of folds for cross validation
    n_min_sm: minimum number of points for selecting the optimal smoothing parameter (Must be > 0)
    kernel: RBF kernel (see scipy.interpolate.Rbf for details)
    pool: pool of workers (used for parallel computing)
    poly_deg: int, degree of rbf polynomial tail. If = 0, then no polynomial tail, either 0 or 1,
              useful ONLY when use_scipy_rbf = False
    
    Output:
    rbf_mod: RBF regression model
    opt_sm: optimal smooth parameter
    cv_err_arr: cv error corresponding to each smooth parameter
    rbf_mod_raw: raw RBF object
    '''
    # sanity check
    assert(len(X) == len(Y) > 0)
    assert(len(sm_range) in [1,2])
    assert(n_fold > 1)
    
    n_uniq = len(unique_row(X)) # find number of unique X
    if n_uniq > 1:
        # then we build RBF surrogate
        if normalize_data:
            # normalize X and Y to [0,1]
            X_scaler = preprocessing.MinMaxScaler()
            Y_scaler = preprocessing.MinMaxScaler()
            X = X_scaler.fit_transform(X)
            Y = Y_scaler.fit_transform(Y.reshape((-1,1))).flatten()
        else:
            X_scaler = Y_scaler = None 
            
        if len(sm_range) == 2:
            sm_lw, sm_up = sm_range
            assert(sm_lw < sm_up), 'invalid sm_range'
            assert(n_proc % 1 ==0 and n_proc >0), 'invalid n_proc'    
            
            n_iter = int(np.ceil(n_min_sm/float(n_proc))) # number of iterations
            n_req = n_proc*n_iter # total number of requested points
            assert(n_req>=n_min_sm>0) # sanity check
            # find candidate smooth parameters
            if log_opt:
                smooth = np.logspace(np.log10(sm_lw),np.log10(sm_up),n_req)
            else:
                smooth = np.logspace(sm_lw,sm_up,n_req)
            
            if pool is None:
                # then we compute in serial 
                cv_err_arr = [CV_smooth(s,X,Y,n_fold,kernel,use_scipy_rbf,wgt_expon,poly_deg) for s in smooth]
            else:
                # then we compute in parallel
                CV_smooth_partial = partial(CV_smooth,X=X,Y=Y,n_fold=n_fold,kernel=kernel,
                                            use_scipy_rbf=use_scipy_rbf,wgt_expon=wgt_expon,poly_deg=poly_deg)        
                cv_err_arr = pool.map(CV_smooth_partial,smooth,chunksize=n_iter)
            # find smooth parameter that has smallest cv error
            opt_ix = np.argmin(cv_err_arr)
            opt_sm = smooth[opt_ix]
            
        else:
            # i.e., we specify the opt_sm
            opt_sm = sm_range[0]
            cv_err_arr = None
            
        # then we build RBF surrogate with the optimal smooth parameter using all the data
        XY_data = np.vstack((X.T,Y.reshape((1,-1))))
        if not use_scipy_rbf:
            wgt = gen_rbf_wgt(Y,wgt_expon)
            rbf_mod_raw = Rbf(*XY_data,wgt=wgt,function=kernel,smooth=opt_sm,deg=poly_deg)        
        else:
            rbf_mod_raw = Rbf(*XY_data,use_scipy_rbf=use_scipy_rbf,function=kernel,smooth=opt_sm)
        rbf_mod = gen_normalize_rbf(rbf_mod_raw,X_scaler,Y_scaler)
        
    else:
        # pathological case: X overlaps. Then the model is simply the mean of y values.
        opt_sm, cv_err_arr, rbf_mod_raw = np.nan, None, None
        def rbf_mod(x):
            '''
            Input:
            x: 2d array (different rows are different points) or 1d array
            
            Output:
            y: 1d array
            '''
            if x.ndim == 1:
                x = x.reshape((1,-1)) # convert to 2d if it is 1d
            nx = len(x)
            y = np.mean(Y)*np.ones(nx)
            
            return y
        
    return rbf_mod, opt_sm, cv_err_arr, rbf_mod_raw

# generate normalized RBF model (essentially a decorator)
def gen_normalize_rbf(rbf_mod,X_scaler,Y_scaler):
    '''
    Input:
      rbf_mod: Rbf model returned from Rbf function
      X_scaler: scaler object for X (if None, then no normalization will be performed on X)
      Y_scaler: scaler object for Y (if None, then no normalization will be performed on Y)
    Output:
      norm_rbf_mod: normalized Rbf model
    '''
    def norm_rbf_mod(X):
        '''
        Input:
        X: 2d array (different rows are different points) or 1d array
        
        Output:
        Y: 1d array
        '''
        if X.ndim == 1:
            X = X.reshape((1,-1)) # convert to 2d if it is 1d
        if X_scaler is not None:
            X = X_scaler.transform(X)
        Y = rbf_mod(*X.T)
        if Y_scaler is not None:
            Y = Y_scaler.inverse_transform(Y.reshape((-1,1))).flatten()
        return Y
    return norm_rbf_mod
    
# generate weight vector for weighted rbf
def gen_rbf_wgt(Y,wgt_expon):
    '''
    Input:
    Y: 1d array, y values of data
    wgt_expon: scalar, weight exponent
    
    Output:
    wgt: 1d array, weight vector
    '''
    assert(wgt_expon>=0) # sanity check
    # normalize Y and find the weight
    min_Y, max_Y = np.min(Y), np.max(Y)
    if min_Y == max_Y:
        norm_Y = np.zeros(Y.size)
    else:
        norm_Y = (Y-min_Y)/(max_Y-min_Y)
    assert (np.min(norm_Y)>=0 and np.max(norm_Y)<=1) # sanity check
    wgt = np.exp(-wgt_expon*norm_Y)
    return wgt
       
# cross validated error for a specific smooth parameter       
def CV_smooth(smooth,X,Y,n_fold,kernel,use_scipy_rbf,wgt_expon,poly_deg):
     '''
     Input:     
     smooth: smoothing parameter
     X: x values of data points (2d array, different rows are different points)
     Y: y values of data points (1d array)
     n_fold: number of folds for cross validation
     kernel: RBF kernel (see scipy.interpolate.Rbf for details)
     use_scipy_rbf: whether to use scipy rbf
     wgt_expon: scalar, weight exponent used in building weighted rbf, useful ONLY when use_scipy_rbf = False
     poly_deg: int, degree of rbf polynomial tail. If = 0, then no polynomial tail, either 0 or 1, 
               useful ONLY when use_scipy_rbf = False
     
     Output:
     cv_err: cross validated error
     '''
     nX = len(X)
     n_fold = min(nX, n_fold)
     assert(n_fold > 1) # sanity check
     
     kf = KFold(n_splits=n_fold)
     test_err = np.zeros(n_fold)
     for f,index in enumerate([j for j in kf.split(np.ones(nX))]):
        train_ix, test_ix = index
        X_train, X_test = X[train_ix], X[test_ix]
        Y_train, Y_test = Y[train_ix], Y[test_ix]
        # get number of unique X_train
        n_uniq = len(unique_row(X_train))
        if n_uniq > 1:
            # then we build RBF model
            XY_data = np.vstack((X_train.T,Y_train.reshape((1,-1))))
            if not use_scipy_rbf:
                wgt = gen_rbf_wgt(Y_train,wgt_expon)
                rbf_mod = Rbf(*XY_data,wgt=wgt,function=kernel,smooth=smooth,deg=poly_deg)
            else:
                rbf_mod = Rbf(*XY_data,use_scipy_rbf=use_scipy_rbf,function=kernel,smooth=smooth)
            test_err[f],_ = test_RBF_reg(X_test,Y_test,rbf_mod)
        else:
            # pathological case: X_train overlaps. Then the model is simply the mean of y values.
            Y_pred = np.mean(Y_train)
            error = Y_pred - Y_test
            test_err[f] = np.sqrt(np.mean(error**2)) # RMSE
     cv_err = np.mean(test_err)
     return cv_err

# test performance of RBF model (i.e. RMSE of rbf_mod(X) with repect to Y)
def test_RBF_reg(X,Y,rbf_mod):
    '''
    Input:
    X: x values of data points (2d array, different rows are different points)
    Y: y values of data points (1d array)
    rbf_mod: RBF model
    
    Output:    
    RMSE: root mean square error
    error: fit error = rbf_mod(X)-Y
    '''
    error = rbf_mod(*X.T)-Y
    RMSE = np.sqrt(np.mean(error**2))
    return RMSE,error

# get unique rows of a 2d array
def unique_row(data):
    '''
    Input:
    data: 2d numpy array
    
    Output:
    2d array data with unique rows
    
    (reference: http://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array)
    
    Note: experiments on OSX show this method is somewhat faster than np.unique method especially for large number of points.
    '''
    if data.size > 0: # i.e. non-empty        
        # Perform lex sort and get sorted data
        sorted_idx = np.lexsort(data.T)
        sorted_data =  data[sorted_idx,:]
        
        # Get unique row mask
        row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
        
        # Get unique rows
        return sorted_data[row_mask] 
    else:
        return data  

def put_back_box(pt,bound):
    '''
    Input:
        pt: points, 2d array: (n_pt, dim)
        bound: box bound, list of bound tuples for each dimension
    Output:
        new_pt: points after putting back to the box (and removing duplicate points)
        raw_pt: points after putting back to the box (without removing duplicate points)
    '''
    n_pt, dim = pt.shape
    assert(len(bound)==dim)
    for j in range(dim):
        min_bd = bound[j][0]
        max_bd = bound[j][1]
        if j == 0:
            ind = np.logical_and(pt[:,j]>min_bd,pt[:,j]<max_bd) # indicator that shows if a number is inside the range
        else:
            ind = np.logical_or(ind,np.logical_and(pt[:,j]>min_bd,pt[:,j]<max_bd))
        pt[:,j] = np.maximum(min_bd,np.minimum(max_bd,pt[:,j]))       
    pt_1 = pt[ind]
    pt_2 = pt[np.logical_not(ind)]
    raw_pt = np.vstack((pt_1,pt_2))
    pt_2 = unique_row(pt_2) # find unique points in X_samp_2
    # get unique points
    new_pt = np.vstack((pt_1,pt_2))
    return new_pt, raw_pt

def scale_zero_one(arr):
    '''
    Scale a 1d array to the range [0,1] with minimum corresponding to 0 
    and maximum to 1.
    
    Input:
        arr: 1d array
    Output:
        scale_arr: 1d array
    '''
    max_arr, min_arr = np.amax(arr), np.amin(arr)
    if max_arr != min_arr:
        scale_arr = (arr-min_arr)/(max_arr-min_arr)
    else:
        scale_arr = np.ones_like(arr)
        
    return scale_arr

def scale_one_zero(arr):
    '''
    Scale a 1d array to the range [0,1] with minimum corresponding to 1
    and maximum to 0.
    
    Input:
        arr: 1d array
    Output:
        scale_arr: 1d array
    '''            
    max_arr, min_arr = np.amax(arr), np.amin(arr)
    if max_arr != min_arr:
        scale_arr = (max_arr-arr)/(max_arr-min_arr)
    else:
        scale_arr = np.ones_like(arr)
        
    return scale_arr
            
def SRS(fit_mod, cand_pt, cmp_pt, wgt_pat_arr):
    '''
    Propose points using stochastic response surface method (SRS)
    Input:
        fit_mod: response model, a function that takes 2d array X and outputs 1d array Y
        cand_pt: candidate points, 2d array
        cmp_pt: points to compare when determining the distance score, 2d array
        wgt_pat_arr: weight pattern, 1d array (length equal to number of proposed point)
    Output:
        prop_pt_arr: proposed points array, 2d array   
    '''
    assert(wgt_pat_arr.ndim == 1 and np.min(wgt_pat_arr) >= 0 and np.max(wgt_pat_arr) <= 1)
    n_prop = len(wgt_pat_arr)
    n_cd, dim = cand_pt.shape
    assert(n_cd>=n_prop)
    
    # find response score
    resp_cand = fit_mod(cand_pt)
    resp_score = scale_zero_one(resp_cand)
    # initializations
    prop_pt_arr = np.zeros((n_prop,dim))
    # propose points sequentially
    for j in range(n_prop):
        wt = wgt_pat_arr[j]
        if len(cmp_pt) > 0:
            if j == 0:                                       
                # distance matrix for cmp_pt and cand_pt
                dist_mat = cdist(cand_pt,cmp_pt)   
                dist_cand = np.amin(dist_mat,axis=1)                        
            else:
                # distance to the previously proposed point
                dist_prop_pt = cdist(cand_pt,prop_pt_arr[j-1].reshape((1,-1))).flatten()                    
                dist_cand = np.minimum(dist_cand,dist_prop_pt)
            dist_score = scale_one_zero(dist_cand)
        else:
            # i.e., no points to compare (pathological case)
            dist_score = np.zeros(n_cd)
        cand_score = resp_score*wt+(1-wt)*dist_score            
        assert (np.max(cand_score)<=1 and np.min(cand_score)>=0) # sanity check            
        # select from candidate points
        min_ix = np.argmin(cand_score)
        prop_pt = cand_pt[min_ix]
        prop_pt_arr[j] = prop_pt            
        # add proposed point to cmp_pt
        cmp_pt = np.vstack((cmp_pt,prop_pt.reshape(1,-1)))
        # remove proposed point from cand_pt and resp_score and dist_score
        dist_cand = np.delete(dist_cand,min_ix)
        resp_score = np.delete(resp_score,min_ix)
        cand_pt = np.delete(cand_pt,min_ix,axis=0)
        # update number of candidates and points to be compared
        n_cd -= 1
    
    return prop_pt_arr

def func_eval(f,x_list,seed_list,out_list,pool,comm):
    '''
    Evaluate functions in parallel.
    Input:
        f: evaluation function
        x_list: list of x (1d array) or 2d array for each evaluation
        seed_list: list of seed (int) or 1d array for each evaluation
        out_list: list of output netcdf file (string) for each evaluation
        pool: Pool object. Useful only if comm is None
        comm: MPI communicator instance or None. If None, use multiple cores of a node for evaluation.
    Output:
        y_list: list, function evaluations for each x in x_list
    '''
    # sanity check
    nx = len(x_list)
    assert(nx==len(seed_list)==len(out_list))
    
    x_seed_f_out_list = [(x,seed,f,out) for x,seed,out in zip(x_list,seed_list,out_list)]
    if comm is None:
        y_list = pool.map(eval_wrapper,x_seed_f_out_list)
    else:
        assert(comm.rank == 0), 'only root process can call func_eval'
        assert(comm.size >= nx), 'number of available nodes should be no less than number of evaluations'
        # send data to remaining processes
        for r in range(1,nx):
            data = {'x': x_list[r], 'seed': seed_list[r], 'out': out_list[r]}
            comm.send(data, dest=r, tag=MPI_tag1)
        # evaluate the first data point
        y_list = []
        y_list.append(eval_wrapper([x_list[0], seed_list[0], f, out_list[0]]))
        # get results from remaining processes
        for r in range(1,nx):
            data = np.ones(1)*np.nan
            comm.Recv(data, source=r, tag=MPI_tag2)
            y = data.item(0)
            assert(not np.isnan(y))
            y_list.append(y)
    
    return y_list
       
def eval_wrapper(x_seed_f_out):
    '''
    Evaluate function wrapper.
    Input:
        x_seed_f_out: list or tuple, [x, seed, f, out]
    Output:
        y: float, f(x) with specified random seed
    '''
    # whether to preserve random state upon calling this function
    preserve_random_state = True
    
    # parse inputs
    x, seed, f, nc_file_out = x_seed_f_out
    
    if preserve_random_state:
        state = np.random.get_state() # save random state
    
    assert(x.shape == (len(x),) and seed>0 and seed%1 == 0) # sanity check
    t1, t1_c = default_timer(), clock()
    y = f(x, seed=seed) # output is a scalar
    t2, t2_c = default_timer(), clock()
    t_eval, t_eval_cpu = t2-t1, t2_c-t1_c # evaluation time
    # save evaluation to file
    if os.path.isfile(nc_file_out):
        os.remove(nc_file_out) # remove file if exists
    with netcdf.netcdf_file(nc_file_out,'w') as g:
        g.createDimension('var',len(x))
        nc_x = g.createVariable('x','d',('var',))
        nc_x[:] = x
        nc_y = g.createVariable('y','d',())
        nc_y.assignValue(y)
        nc_seed = g.createVariable('seed','i',())
        nc_seed.assignValue(seed)
        nc_t = g.createVariable('tw_eval','d',())
        nc_t.assignValue(t_eval)
        nc_t = g.createVariable('tc_eval','d',())
        nc_t.assignValue(t_eval_cpu)
            
    if preserve_random_state:
        np.random.set_state(state) # set random state (ensure the random state does not change through calling this function)
    
    return y

def doe(n_samp,bd):
    '''
    Generate a design of experiments
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

def get_box_samp(samp,bd):
    '''
    Get in-box and out-of-box samples given the bound.
    Input:
        samp: 2d array, total samples
        bd: list of tuples, bound
    Output:
        in_box_ix: 1d array of boolean, indicate whether it is an in-box sample
        out_box_ix: 1d array of boolean, indicate whether it is an out-of-box sample
    '''
    n_samp,dim = samp.shape
    for i in range(dim):
        assert(bd[i][0]<bd[i][1])
        ix = np.logical_and(bd[i][0]<=samp[:,i],bd[i][1]>=samp[:,i])
        if i == 0:
            in_box_ix = ix # indicates whether the points are in the box
        else:
            in_box_ix = np.logical_and(in_box_ix,ix)
    out_box_ix = np.logical_not(in_box_ix)
    
    return in_box_ix,out_box_ix
 
def eff_n_samp(samp, bd):
    '''
    Get effective number of samples within in the bound
    Input:
        samp: 2d array, samples
        bd: list of tuples, bound
    Output:
        n_eff: int, effective number of samples
    '''
    n_samp, dim = samp.shape
    assert(dim == len(bd))
    m = int(np.ceil(n_samp**(1./dim)))
    assert(m>1)
    # find grid for each axis
    grid_arr = np.zeros((dim,m+1))
    for j,axis_bd in enumerate(bd):
        grid_arr[j] = np.linspace(axis_bd[0],axis_bd[1],num=m+1)
    # find location of each sample (save in string)
    loc_samp = ['']*n_samp
    for i in range(n_samp):
        for j,grid in enumerate(grid_arr):
            assert(grid[0]<=samp[i,j]<=grid[-1])
            ix = bisect.bisect_left(grid,samp[i,j])
            assert(ix<m+1)
            loc = 0 if ix == 0 else ix-1
            loc_samp[i] += str(loc)
    # find number of samples with unique location
    n_eff = len(set(loc_samp))        
    assert(n_eff<=n_samp)
    
    return n_eff

def intsect_bd(bd1,bd2):
    '''
    Find intersection of two bounds
    Input:
        bd1: list of tuples
        bd2: list of tuples
    Output:
        inter_bd: list of tuples or None. If None, then the intersection is empty
    '''
    lb1, ub1 = zip(*bd1)
    lb2, ub2 = zip(*bd2)
    lb1, ub1 = np.array(lb1), np.array(ub1)
    lb2, ub2 = np.array(lb2), np.array(ub2)
    assert(np.all(ub1>=lb1) and np.all(ub2>=lb2))
    assert(lb1.shape==ub1.shape==lb2.shape==ub2.shape)
    inter_lb = np.maximum(lb1,lb2)
    inter_ub = np.minimum(ub1,ub2)
    if np.all(inter_ub>inter_lb):
        inter_bd = list(zip(inter_lb,inter_ub))
    else:
        inter_bd = None
    return inter_bd

def get_child_node(x_star, n_zoom, tree):
    '''
    Get the child node.
    Input:
        x_star: 1d array. Focal point of zoom-in.
        n_zoom: int. Zoom level of current parent node.
        tree: dictionary. Optimization tree.
    Output:
        child_ix: None or int. Selected child node index. If None, then we need to create a new child.
    '''
    assert(x_star.ndim == 1)
    # get zoom level of a child node
    n_zoom_child = n_zoom+1
    if n_zoom_child not in tree.keys():
        child_ix = None
    else:
        # the indice of candidate child nodes
        child_node_ix_list = [i for i,c in enumerate(tree[n_zoom_child]) if all(get_box_samp(x_star.reshape(1,-1),c['bd'])[0])]
        if len(child_node_ix_list) == 0:
            child_ix = None
        else:
            # find the child node among the candidates, of which the center of the domain is closest to x_star
            dist_list = [np.linalg.norm(np.mean(tree[n_zoom_child][i]['bd'])-x_star) for i in child_node_ix_list]
            child_ix = child_node_ix_list[np.argmin(dist_list)]
    
    return child_ix

def ncread(nc_file, var, isScalar=False):
    """
    Read variable from a netcdf file
    Input:
      - nc_file: full path of netcdf file
      - var: variable string
      - isScalar: whether is a scalar
    Output:
      - variable value (scalar or numpy array)    
    """
    with netcdf.netcdf_file(nc_file,'r',mmap=False) as f: # mmap = False is to get rid of possible runtime warning in some scipy version
        nc_var = f.variables[var]
        if isScalar:
            val = nc_var.getValue()
        else:
            val = nc_var[:].copy()
    return val
        
class std_out_logger(object):
    '''
    Write standard output to a file.
    Reference: http://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    '''
    def __init__(self,log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        # This is needed for python 3 compatibility.
        pass

class std_err_logger(object):
    '''
    Write standard error to a file.
    Reference: http://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    '''
    def __init__(self,log_file):
        self.terminal = sys.stderr
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        # This is needed for python 3 compatibility.
        pass        

def run(prob, n_iter, n_proc, n_core_node, comm, outdir, init_iter=None, seed=1, save_samp=False, 
        verbose=False, resume_iter=None):
    
    """
    Run ProSRS optimization algorithm.
    
    Input:
        
        prob: instance of optim_prob class, optimization problem.
        
        n_iter: integer, total number of iterations including both DOE iterations and optimization iteration.
        
        n_proc: integer, number of evaluations per iteration for parallel optimization.
        
        n_core_node: integer, number of cores in one node
        
        comm: instance or None, MPI communicator. If None, then we do parallel optimization within one node
        
        outdir: string, path of output directory that saves optimization results.
        
        init_iter: integer or None, initial number of iterations for DOE. If None, then we use default value.
        
        seed: unsigned integer, random seed.
        
        save_samp: boolean, whether to save samples for each iteration.
        
        verbose: boolean, whether to verbose about algorithm.
    
        resume_iter: integer or None, starting iteration for resume (= 'n_iter' in last run). If None, then no resume.
    
    Output:
        
        best_loc: 1d array, location of the best point.
        
        best_val: float, (noisy) evaluation of the best point.
    """
    if comm is None or comm.rank == 0:
        
        try:
    
            ############## Algorithm parameters (global constants) ####################
            
            n_cand_fact = 1000 # number of candidate = n_cand_fact * d, where d is dimension of the problem
            wgt_pat_bd = [0.3,1.] # weight pattern bound
            normalize_data = True # whether to normalize data when training radial basis function
            init_wgt_expon = 0. # initial weight exponent for local SRS (>=0) if == 0, then essentially no weighting involved when fitting with rbf
            wgt_expon_delta = 2. # increment of weight exponent whenever failure occurs
            max_reduce_step_size = 2 # max number of reducing step size for pure local SRS before zooming in
            box_fact = 0.4 # box factor determines the shrinkage ratio of box. Must be in (0, 1)
            init_step_size_fact_list = [0.1] # list of initial step size factors for local SRS
            init_gSRS_pct = 1. # initial global SRS percentage
            init_zoom_out_prob = 0.02 # initial probability of zooming out
            min_zoom_out_prob = 0.01 # minimum probability of zooming out
            alpha = 1. # parameter for controling global SRS percentage dynamics
            log_std_out = True # whether to log standard output to a file
            log_std_err = True # whether to log standard error to a file
            sm_range = [1e-7,1e4] # range of smooth parameters
            rbf_poly_deg = 0 # degree for rbf polynomial tail, [0,1]. If 0, then no polynomial tail for rbf.
            n_fold = 5 # number of folds for cross validation when training radial basis function
            log_opt = True # whether do optimization of smooth parameter in log space
            resol = 0.01 # resolution of solution
            rbf_kernel = 'multiquadric' # default: multiquadric
            use_eff_n_samp = True # whether to use effective number of samples for the dynamics of gSRS_pct_full
            max_C_fail = None # maximum number of consecutive failures before reducing step size. If None, then we use default
            
            # convert input to correct data type if not done so
            if init_iter is None:
                init_iter = int(np.ceil(3/float(n_proc))) # default value
            else:
                init_iter = int(init_iter)
            n_iter = int(n_iter)
            assert(n_iter > init_iter), 'total iterations should be larger than iterations of DOE = %d. Please set n_iter > %d.' % (init_iter,init_iter)
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
            # get maximum number of consecutive failures
            if max_C_fail is None:
                max_C_fail = max(2,int(np.ceil(dim/float(n_proc))))
            assert(type(max_C_fail) == int and max_C_fail > 0)
            assert(n_cand_fact*dim >= n_proc), 'number of candidate points needs to be no less than n_proc'
            
            # get pool of processes for parallel computing
            pool_eval = Pool(processes=n_proc) # for function evaluations
            pool_rbf = Pool(processes=n_core_node) # for training rbf surrogate
                
            # sanity check
            assert(wgt_expon_delta >= 0 and max_reduce_step_size>=0 and 0<=init_gSRS_pct<=1)  
            assert(0<box_fact<1 and resol>0)
            assert(0<=min_zoom_out_prob<=init_zoom_out_prob<=1)
            assert(n_fold > 1)
            assert(rbf_poly_deg in [0,1])
            assert(min(wgt_pat_bd)>=0 and max(wgt_pat_bd)<=1 and len(wgt_pat_bd)==2)
            
            # check if the output directory exists
            assert(os.path.isdir(outdir))
                
            # log file (write standard output to a file)
            if log_std_out:
                orig_std_out = sys.stdout
                log_file = os.path.join(outdir, 'std_output_log_%s.txt' % prob_name)
                if resume_iter is None:
                    if os.path.isfile(log_file):
                        os.remove(log_file)
                sys.stdout = std_out_logger(log_file)    
            if log_std_err:
                orig_std_err = sys.stderr
                log_file = os.path.join(outdir, 'std_error_log_%s.txt' % prob_name)
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
                
                state = {'p': init_gSRS_pct, # global SRS percentage (full, without rounding to 0.1 resolution)
                         'Cr': 0, # counter that counts number of times of reducing step size of local SRS
                         'Cf': 0, # counter that counts number of consecutive failures
                         'w': init_wgt_expon # weight exponent parameter of weighted RBF
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
                    n_cand_lSRS_tot = n_cand-n_cand_gSRS # total number of candidate points for local SRS
                    assert (n_cand_lSRS_tot>0) # sanity check
                    n_lSRS = len(init_step_size_fact_list)
                    n_cand_lSRS_list = [int(round(n_cand_lSRS_tot/float(n_lSRS)))]*n_lSRS
                    n_cand_lSRS_list[-1] = n_cand_lSRS_tot-sum(n_cand_lSRS_list[:-1]) # to ensure the sum = n_lSRS
                    assert(sum(n_cand_lSRS_list) == n_cand_lSRS_tot and min(n_cand_lSRS_list) > 0)
                    
                    # generate candidate points uniformly (global SRS)           
                    cand_pt = np.zeros((n_cand_gSRS,dim))
                    if n_cand_gSRS>0:
                        for d,bd in enumerate(fit_bd):
                            cand_pt[:,d] = np.random.uniform(low=bd[0],high=bd[1],size=n_cand_gSRS)
                    
                    for init_step_size_fact,n_cand_lSRS in zip(init_step_size_fact_list,n_cand_lSRS_list):
                    
                        # find step size (i.e. std) for each coordinate of x_star
                        step_size_fact = init_step_size_fact*0.5**n_reduce_step_size
                        step_size_arr = np.array([step_size_fact*(x[1]-x[0]) for x in fit_bd])
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
                                  'zp': init_zoom_out_prob, # zoom-out probability
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
                       and normalize_data==data['normalize_data'] and init_wgt_expon==data['init_wgt_expon']
                       and wgt_expon_delta==data['wgt_expon_delta'] and max_reduce_step_size==data['max_reduce_step_size']
                       and box_fact==data['box_fact'] and np.all(init_step_size_fact_list==data['init_step_size_fact_list'])
                       and init_gSRS_pct==data['init_gSRS_pct'] and np.all(wgt_pat_bd==data['wgt_pat_bd'])
                       and np.all(sm_range==data['sm_range']) and log_opt==data['log_opt']
                       and n_fold==data['n_fold'] and resol==data['resol'] and min_zoom_out_prob==data['min_zoom_out_prob']
                       and init_zoom_out_prob==data['init_zoom_out_prob'] and max_C_fail==data['max_C_fail']
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
                wgt_expon = act_node['state']['w']
                zoom_out_prob = act_node['zp']
                fit_bd = act_node['bd']
                        
                if not full_restart:
                        
                    ########### Fit model ##############
                    
                    t1 = default_timer()
                    t1_c = clock()
        
                    rbf_mod,opt_sm,cv_err,_ = RBF_reg(X_samp,Y_samp,n_core_node,sm_range,
                                                      normalize_data=normalize_data,wgt_expon=wgt_expon,
                                                      log_opt=log_opt,n_fold=n_fold,kernel=rbf_kernel,pool=pool_rbf,
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
                        print('wgt_expon = %g' % wgt_expon)
                        print('opt_sm = %.1e' % opt_sm)
                        print('gSRS_pct = %g' % gSRS_pct)
                        print('n_fail = %d' % n_fail)
                        print('n_reduce_step_size = %d' % n_reduce_step_size)
                        step_size_fact_arr = np.array(init_step_size_fact_list)*0.5**n_reduce_step_size
                        print('step_size_fact_list = %s' % str(step_size_fact_arr.tolist()))
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
                                wgt_expon += wgt_expon_delta
                                n_reduce_step_size += 1 # update counter
                        act_node['state']['p'] = gSRS_pct_full
                        act_node['state']['Cr'] = n_reduce_step_size
                        act_node['state']['Cf'] = n_fail
                        act_node['state']['w'] = wgt_expon
                
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
                    
                    if n_reduce_step_size > max_reduce_step_size:
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
                            fit_lb = np.maximum(x_star-box_fact/2.0*blen,fit_lb)
                            fit_ub = np.minimum(x_star+box_fact/2.0*blen,fit_ub)
                            fit_bd = list(zip(fit_lb,fit_ub)) # the list function is used to ensure compatibility of python3
                            child_node = {'ix': np.nonzero(get_box_samp(X_all,fit_bd)[0])[0], 
                                          'bd': fit_bd,
                                          'parent_ix': act_node_ix,
                                          'zp': init_zoom_out_prob,
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
                            tree = {n_zoom: [{'ix': np.arange(0,dtype=int), # indice of samples for the node (with respect to X_all_lf and Y_all_lf)
                                              'bd': func_bd, # domain of the node
                                              'parent_ix': None, # parent node index for the upper zoom level (zero-based). If None, there's no parent
                                              'zp': init_zoom_out_prob, # zoom-out probability
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
                                child_node['zp'] = max(min_zoom_out_prob,child_node['zp']/2.)
                                
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
                         n_cand_fact=n_cand_fact,use_eff_n_samp=use_eff_n_samp,init_zoom_out_prob=init_zoom_out_prob,
                         normalize_data=normalize_data,init_wgt_expon=init_wgt_expon,wgt_expon_delta=wgt_expon_delta,
                         max_reduce_step_size=max_reduce_step_size,box_fact=box_fact,init_step_size_fact_list=init_step_size_fact_list,
                         init_gSRS_pct=init_gSRS_pct,alpha=alpha,wgt_pat_bd=wgt_pat_bd,
                         log_std_out=log_std_out,log_std_err=log_std_err,sm_range=sm_range,log_opt=log_opt,
                         n_fold=n_fold,resol=resol,min_zoom_out_prob=min_zoom_out_prob,rbf_kernel=rbf_kernel,
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
                if log_std_out:
                    sys.stdout.terminal.flush()
                    sys.stdout.log.flush()
                if log_std_err:
                    sys.stderr.terminal.flush()
                    sys.stderr.log.flush()
                  
            # reset stdout and stderr
            if log_std_out:
                sys.stdout = orig_std_out # set back to original stdout (i.e. to console only)
            if log_std_err:
                sys.stderr = orig_std_err # set back to original stderr (i.e. to console only)
            
            # find best point and its (noisy) function value
            min_ix = np.argmin(best_val_it)
            best_loc = best_loc_it[min_ix]
            best_val = best_val_it[min_ix]
            
            return best_loc, best_val
        
        except:
            # error handling
            if comm.rank == 0:
                comm.Abort() # kill all the other processes so that the program does not run forever
    
    else:
        # i.e., comm is not None and comm.rank > 0
        try:
            n_iter_worker = n_iter if resume_iter is None else n_iter-resume_iter # number of iterations for each worker
            for _ in range(n_iter_worker):
                data = comm.recv(source=0, tag=MPI_tag1) # data is a dictionary
                # unpack message
                x = data['x'] # 1d array
                seed = data['seed'] # int
                out = data['out'] # str
                # function evaluation 
                y = eval_wrapper([x,seed,prob.object_func,out])
                # send y value to root
                comm.Send(y, dest=0, tag=MPI_tag2)
        
        except:
            # error handling
            comm.Abort() # kill all the other processes so that the program does not run forever