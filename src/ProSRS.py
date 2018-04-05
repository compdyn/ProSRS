"""
This code implements Progressive Stochastic Response Surface (ProSRS) algorithm.

Copyright (C) 2016-2018 Chenchao Shou
"""

from __future__ import division, absolute_import

import sys
import numpy as np
from sklearn.model_selection import KFold
from pyDOE import lhs
import os
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn import preprocessing
from timeit import default_timer
from time import clock
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from scipy.io import netcdf
from scipy import linalg
from scipy.special import xlogy
import pickle
import matplotlib as mpl
import bisect
import matplotlib.pyplot as plt
import warnings
import shutil
# set figure properties
mpl.style.use('classic')
#mpl.rc('text', usetex = True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#mpl.rc('font', family = "serif", serif = ["Computer Modern Roman"])

########################### Global Constants ####################

# output file template
RESULT_NPZ_FILE_TEMP = 'optim_result_%s.npz'
RESULT_PKL_FILE_TEMP = 'optim_result_%s.pkl'
RESULT_GENERAL_SAMP_FILE_TEMP = 'samp_%s_t%d.nc'
RESULT_SAMP_FILE_TEMP = 'samp_%s_t%d_p%d.nc'
RESULT_LF_SAMP_FILE_TEMP = 'samp_%s_t%d_p%d_lf.nc'
RESULT_HF_SAMP_FILE_TEMP = 'samp_%s_t%d_p%d_hf.nc'
# file constant
TEMP_RESULT_NPZ_FILE_SUFFIX = '.temp.npz'

###################### Classes & Functions #######################

class Rbf(object):
    """
    This class implements regularized RBF regression.
    The class is modified based on scipy.interpolate.Rbf (scipy version: 0.19.0)
    
    Copyright (C) 2016-2017 Chenchao Shou
    
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
        
        if self.smooth < 0:
            raise ValueError('Smooth parameter cannot be negative')
        self.use_scipy_rbf = kwargs.pop('use_scipy_rbf',False)
        self.wgt = kwargs.pop('wgt',np.ones(self.N)) # weight vector        
        assert ((len(self.wgt) == self.N) and np.min(self.wgt)>=0),'invalid weight vector'
        W = np.diag(self.wgt) # construct weight matrix
        
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
                self.A = np.dot(np.dot(phi.T,W),phi)+np.eye(self.N)*self.smooth
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
        return np.dot(self._function(r), self.nodes).reshape(shp)
        
    def gradient(self, x):
        '''
        find the gradient at x.
        
        Input:
          - x: 1d numpy array
        Output:
          - grad: 1d numpy array
        '''
        assert (x.shape == (self.xi.shape[0],)), 'Incorrect shape of input to derivative of Rbf'
        grad = None
        if self.function == 'multiquadric':
            r = self._call_norm(x.reshape((-1,1)), self.xi)
            phi = self._function(r)            
            phi = phi.flatten()            
            W = self.nodes/phi
            grad = (np.sum(W)*x-np.dot(self.xi,W))/(self.epsilon**2)
        return grad

# RBF regression
def RBF_reg(X,Y,n_proc,sm_range,normalize_data=True,wgt_expon=0.0,use_scipy_rbf=False,
            log_opt=True,n_fold=5,n_pt_min=10,kernel='multiquadric',verbose=False,pool=None):
    '''
    Input:
    X: x values of data points (2d array, different rows are different points)
    Y: y values of data points (1d array)
    n_proc: number of processors (cores)
    sm_range: range of smooth parameter ([low,upper])
    normalize_data: boolean. Whether to normalize data before training
    wgt_expon: scalar, weight exponent used in building weighted rbf, useful ONLY when use_scipy_rbf = False
    log_opt: whether to optimize the smooth parameter on a log scale (boolean)
    n_fold: number of folds for cross validation
    n_pt_min: minimum number of points in selecting the optimal smoothing parameter (Must be > 0)
    kernel: RBF kernel (see scipy.interpolate.Rbf for details)
    verbose: whether to plot cross validated error versus smooth parameter
    pool: pool of workers (used for parallel computing)
    
    Output:
    rbf_mod: RBF regression model
    opt_sm: optimal smooth parameter
    cv_err_arr: cv error corresponding to each smooth parameter
    rbf_mod_raw: raw RBF object
    '''
    nX = len(X)
    assert(nX == len(Y) and nX > 0) # sanity check
    
    if nX > 2:
        # then we build RBF surrogate using cross validation
        if normalize_data:
            # normalize X and Y to [0,1]
            X_scaler = preprocessing.MinMaxScaler()
            Y_scaler = preprocessing.MinMaxScaler()
            X = X_scaler.fit_transform(X)
            Y = Y_scaler.fit_transform(Y.reshape((-1,1))).flatten()
        else:
            X_scaler = Y_scaler = None
            
        if len(sm_range) == 2:
            assert(sm_range[0] < sm_range[1]), 'invalid input to RBF_reg() function'
            assert(n_proc % 1 ==0 and n_proc >0), 'invalid input to RBF_reg() function'    
            
            n_iter = int(np.ceil(n_pt_min/(1.*n_proc))) # number of iterations
            n_req = n_proc*n_iter # total number of requested points
            assert(n_req>=n_pt_min) # sanity check
            sm_lw, sm_up = sm_range
            # find candidate smooth parameters
            if log_opt:
                smooth = np.logspace(np.log10(sm_lw),np.log10(sm_up),n_req)
            else:
                smooth = np.logspace(sm_lw,sm_up,n_req)
            if pool is None:
                # then we compute in serial 
                cv_err_arr = [CV_smooth(s,X,Y,n_fold,kernel,use_scipy_rbf,wgt_expon) for s in smooth]
            else:
                # then we compute in parallel
                CV_smooth_partial = partial(CV_smooth,X=X,Y=Y,n_fold=n_fold,kernel=kernel,use_scipy_rbf=use_scipy_rbf,\
                                    wgt_expon=wgt_expon)        
                cv_err_arr = pool.map(CV_smooth_partial,smooth,chunksize=n_iter)
        n_uniq = len(unique_row(X)) # find number of unique X
        if n_uniq > 1:
            # then we build RBF model with optimal smooth parameter using all the data
            if len(sm_range) == 2:
                # find smooth parameter that has smallest cv error
                opt_ix = np.argmin(cv_err_arr)
                opt_sm = smooth[opt_ix]
            elif len(sm_range) == 1:
                opt_sm = sm_range[0]
                cv_err_arr = None
            else:
                raise ValueError('invalid sm_range')
            XY_data = np.vstack((X.T,Y.reshape((1,-1))))
            if not use_scipy_rbf:
                wgt = gen_rbf_wgt(Y,wgt_expon)
                rbf_mod_raw = Rbf(*XY_data,wgt=wgt,function=kernel,smooth=opt_sm)        
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
            
    else:
        # pathological case, where number of samples = 1 or 2
        opt_sm, cv_err_arr, rbf_mod_raw = np.nan, None, None
        # the model simply is a constant, equal to the mean of Y values of samples
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
def CV_smooth(smooth,X,Y,n_fold,kernel,use_scipy_rbf,wgt_expon):
     '''
     Input:     
     smooth: smoothing parameter
     X: x values of data points (2d array, different rows are different points)
     Y: y values of data points (1d array)
     n_fold: number of folds for cross validation
     kernel: RBF kernel (see scipy.interpolate.Rbf for details)
     use_scipy_rbf: whether to use scipy rbf
     wgt_expon: scalar, weight exponent used in building weighted rbf, useful ONLY when use_scipy_rbf = False
     
     Output:
     cv_err: cross validated error
     '''
     nX = len(X)
     assert(nX > 2), 'number of samples must be > 2 in order to do cross validation'
     n_fold = min(nX, n_fold) # number of folds should be at least number of points
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
                rbf_mod = Rbf(*XY_data,wgt=wgt,function=kernel,smooth=smooth)
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
        
# get unique row up to specified tolerance
def uniq_row_with_tol_X(data,tol):
    assert(data.shape[1]==len(tol)+1)
    uniq_row = []
    for i,row in enumerate(data):
        if i == 0:
            uniq_row.append(row)
        else:
            uniq = True
            for ur in uniq_row:
                if np.all(np.absolute(ur[:-1]-row[:-1])<tol):
                    # i.e. all the dimensions are within the box defined by tolerance
                    uniq = False
                    break
            if uniq:
                uniq_row.append(row)
    return np.array(uniq_row)

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
            
def SRS(fit_mod, cand_pt, cmp_pt, n_prop, wgt_pat_bd):
    '''
    Propose points using stochastic response surface method (SRS)
    Input:
        fit_mod: response model, a function that takes 2d array X and outputs 1d array Y
        cand_pt: candidate points, 2d array
        cmp_pt: points to compare when determining the distance score, 2d array
        n_prop: number of points to propose, int
        wgt_pat_bd: weight pattern bound, list
    Output:
        prop_pt_arr: proposed points array, 2d array   
    '''
    n_cd, dim = cand_pt.shape
    assert(n_cd>=n_prop)
    # find response score
    resp_cand = fit_mod(cand_pt)
    resp_score = scale_zero_one(resp_cand)
    # initializations
    prop_pt_arr = np.zeros((n_prop,dim))
    assert(len(wgt_pat_bd)==2)
    wgt_pat_arr = np.linspace(wgt_pat_bd[0],wgt_pat_bd[1],n_prop)
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

def func_eval(f_list_list,x_list_list,seed_list_list,out_list_list,pool):
    '''
    Evaluate functions in serial/parallel.
    Input:
        f_list_list: list of list of functions for evaluation of each processor.
        x_list_list: list of [list of x (1d array)] or [x (2d array)] for evaluation of each processor.
        seed_list_list: list of [list of seed (int)] or [seed (1d array)] for evaluation of each processor.
        out_list_list: list of [list of output netcdf file (string)] for evaluation of each processor
        pool: None or Pool object. If None, then we evaluate functions in serial.
    Output:
        y_list_list: list of list of function evaluations
    '''
    # sanity check
    assert(len(f_list_list)==len(x_list_list)==len(seed_list_list)==len(out_list_list))
    
    x_seed_f_out_list = zip(x_list_list,seed_list_list,f_list_list,out_list_list)
    if pool is None:
        y_list_list = [eval_wrapper(x_seed_f_out) for x_seed_f_out in x_seed_f_out_list]      
    else:
        y_list_list = pool.map(eval_wrapper,x_seed_f_out_list)
    
    return y_list_list
        
def eval_wrapper(x_seed_f_out):
    '''
    Evaluate function wrapper.
    Input:
        x_seed_f_out: list or tuple, [x_list, seed_list, f_list, out_list]
    Output:
        y_list: list, list of f(x) with specified random seed
    '''
    # whether to preserve random state upon calling this function
    preserve_random_state = True
    
    # parse inputs
    x_list, seed_list, f_list, out_list = x_seed_f_out
    
    assert(len(x_list)==len(seed_list)==len(f_list)==len(out_list))
    assert(len(set(out_list))==len(out_list)) # check uniqueness
    
    if preserve_random_state:
        state = np.random.get_state() # save random state
    
    y_list = []
    for x,seed,f,nc_file_out in zip(x_list,seed_list,f_list,out_list):
        assert(x.shape == (len(x),) and seed>0 and seed%1 == 0) # sanity check
        np.random.seed(seed) # set seed for reproducibility
        t1, t1_c = default_timer(), clock()
        y = f(x) # output is a numpy array with one element
        t2, t2_c = default_timer(), clock()
        t_eval, t_eval_cpu = t2-t1, t2_c-t1_c # evaluation time
        y = y.item(0)
        y_list.append(y)
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
    
    return y_list

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
    
def sub_doe(samp,n_sub):
    '''
    Subsampling a doe.
    Input:
        samp: 2d array, doe samples
        n_sub: int, number of subsamples
    Output:
        sub_samp: 2d array, subsamples
    '''
    n_samp,dim = samp.shape
    assert(n_sub>0 and n_sub%1 ==0 and n_samp>=n_sub)
    
    if n_sub == n_samp:
        sub_samp = samp.copy()
    else:
        ix = np.random.choice(n_samp)
        sub_samp = [samp[ix]] # initialization
        samp = np.delete(samp,ix,axis=0) # delete it from samples
        for i in range(n_sub-1):
            sub_samp_arr = np.array(sub_samp)
            dist_mat = cdist(samp,sub_samp_arr)
            dist_samp = np.amin(dist_mat,axis=1) 
            assert(dist_samp.shape == (len(samp),))
            max_ix = np.argmax(dist_samp)
            sub_samp.append(samp[max_ix])
            samp = np.delete(samp,max_ix,axis=0) # delete it from samples
        assert(len(samp)==n_samp-n_sub)
        sub_samp = np.array(sub_samp)
    
    return sub_samp

def get_box_samp(samp,bd):
    '''
    Get in-box and out-of-box samples given the bound.
    Input:
        samp: 2d array, total samples
        bd: list of tuples, bound
    Output:
        in_box_ix: 1d array of boolean, indicate whether it is an in-box sample
        out_bod_ix: 1d array of boolean, indicate whether it is an out-of-box sample
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

def subset_arr_ix(arr,sub_arr):
    '''
    Return index of subset array within in an array.
    Input:
        arr: 2d array
        sub_arr: 2d array
    Output:
        sub_ix: 1d array of int
    '''
    sub_ix = np.array([],dtype=int)
    for i,xs in enumerate(sub_arr):
        ix = np.all(arr==xs,axis=1)
        if len(sub_ix) > 0:   
            ix[sub_ix] = False # should exclude any selected point (think about duplicate points)
        assert(np.any(ix)) # sanity check if sub_arr is a subset of arr
        match_ix = np.argmax(ix)
        sub_ix = np.append(sub_ix,match_ix)
    # sanity check
    assert(np.all(arr[sub_ix]==sub_arr))
    assert(len(set(sub_ix)) == len(sub_ix)) # check uniqueness
    
    return sub_ix

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

def run(prob, n_proc, outdir, init_iter=1, opt_iter=30, seed=1, save_samp=True, verbose=False, 
        t_eval_func=None, max_t_opt_ratio=1e-1, debug=False, serial_mode=False, resume_opt_iter=None):
    
    """
    Run ProSRS optimization algorithm.
    
    Input:
        
        prob: instance of optim_prob class, optimization problem.
        
        n_proc: integer, number of processes for parallel optimization.
        
        outdir: string, path of output directory.
        
        init_iter: integer, initial number of iterations for DOE.
        
        opt_iter: integer, number of optimization iterations.
        
        seed: unsigned integer, random seed.
        
        save_samp: boolean, whether to save samples for each iteration. To resume ProSRS, save_samp needs to be True.
        
        verbose: boolean, whether to verbose about algorithm.
        
        t_eval_func: float or None, time (in sec) to evaluate (high fidelity) objective function. If None, then we measure it online. Useful only for multifidelity.
        
        max_t_opt_ratio: float, maximum proportion of time spent in running optimization algorithm for each iteration. Useful only for multifidelity.
        
        debug: boolean, whether to initiate debugging mode.
        
        serial_mode: boolean, whether to initiate serial mode. If True, then building surrogates and evaluating functions are both done in serial.
    
        resume_opt_iter: None or integer, optimization iteration index to resume from (0-based). If None, no resume (i.e., start from scratch).
    
    Output:
        
        best_loc: 1d array, location of the best point.
        
        best_val: float, (noisy) evaluation of the best point.
    """   
    
    ############## Algorithm parameters (global constants) ####################
    
    n_cand_fact = 1000 # number of candidate = n_cand_fact * d, where d is dimension of the problem
    wgt_pat_bd = [0.3,1.] # weight pattern bound
    normalize_data = True # whether to normalize data when training radial basis function
    init_wgt_expon = 0.0 # initial weight exponent for local SRS (>=0) if == 0, then essentially no weighting involved when fitting with rbf
    wgt_expon_delta = 2.0 # increment of weight exponent whenever failure occurs
    max_reduce_step_size = 2 # max number of reducing step size for pure local SRS before zooming in
    box_fact = 0.4 # box factor determines the shrinkage ratio of box. Must be in (0, 1)
    dynam_box_fact = False # whether to make box_fact dynamic with zoom level
    init_step_size_fact = 0.1 # initial step size factor
    init_gSRS_pct = 1.0 # initial global SRS percentage
    round_gSRS_pct_full = True # whether to round gSRS_pct_full
    alpha_base = 1. # base value of alpha, parameter for controling global SRS percentage dynamics
    alpha_amp = 5. # amplitude of alpha, parameter for controling global SRS percentage dynamics
    alpha_crit_err = 0.4 # critical x_star error for alpha, parameter for controling global SRS percentage dynamics
    log_std_out = None # whether to log standard output to a file. If None, log_std_out = verbose or debug (default).
    log_std_err = False # whether to log standard error to a file
    sm_range = [1e-7,1e4] # range of smooth parameters
    log_opt = True # whether do optimization of smooth parameter in log space
    n_fold = 5 # number of folds for cross validation
    resol = 0.01 # resolution of solution
    min_zoom_in = 2 # minimum number of zoom-in
    max_zoom_out = 1 # maximum number of zoom-out per level
    x_star_std_factor = 3. # multiplicative factor for standard deviation of local SRS around x_star. If = 0, then it's hard zoom-in
    max_C_expand_fit = 3 # maximum number of consecutive fit bound expansion before zooming out. If = np.inf, then essentially no zoom-out.
    rbf_kernel = 'multiquadric' # default: multiquadric
    soft_func_bd = False # whether to make function boundary soft
    use_eff_n_samp = True # whether to use effective number of samples for the dynamics of gSRS_pct_full
    lf_test_alpha = 0.05 # significance level of hypothesis test for judging accuracy of low fidelity model
    min_lf_test_samp = 10 # minimum number of (the most recent) samples for the hypothesis test for judging accuracy of low fidelity model
    doe_check_lf_acc = True # whether to check low-fidelity model accuracy after DOE
    mf_x_star_by_fit = True # whether to find x_star by fit value for multi fidelity optimization
    max_C_fail = 2 # maximum number of consecutive failures before reducing step size
    max_C_lf_fail = 2 # maximum number of consecutive failures for low-fidelity accuracy test
    max_C_lf_success = 2 # maximum number of consecutive successes for low-fidelity accuracy test
    visual_debug = False # whether to debugging with visualization. If False, then we only reproduce the result.
    
    # convert input to correct data type if not done so
    init_iter = int(init_iter)
    opt_iter = int(opt_iter)
    n_proc = int(n_proc)
    
    # get parameters of optimization problem
    dim = prob.dim
    func_bd = prob.domain
    obj_func_list = prob.object_func_list # list of objective function
    eff_list = prob.eff_list # list of efficiency for each task
    prob_name = prob.prob_name
    
    # get file and directory names
    result_npz_file = os.path.join(outdir, RESULT_NPZ_FILE_TEMP % prob_name)
    result_pkl_file = os.path.join(outdir, RESULT_PKL_FILE_TEMP % prob_name)
    outdir_debug = outdir+'_debug'
    
    # get plot parameter (for debugging)
    main_title_fs = 13 # figure title font size for the plot
    title_fs = 11 # axis title font size for the plot
    label_fs = 16 # label font size for the plot
    
    # get lower and upper bounds for function
    func_lb, func_ub = zip(*func_bd)
    func_lb, func_ub = np.array(func_lb), np.array(func_ub) # convert to numpy array
    assert(np.all(func_lb<func_ub)) # sanity check
    func_blen = func_ub-func_lb # function bound length for each dimension
    cand_density = (n_cand_fact*dim)**(1./dim) # candidate sample density per dimension
    # get number of candidate points
    n_cand = int(n_cand_fact*dim)
    # get minimum number of processors for high fidelity optimization
    min_n_proc_hf = int(np.ceil(n_proc*0.5))
    
    if serial_mode:
        pool_eval = pool_rbf = None
    else:
        # get pool of processes for parallel computing
        pool_eval = Pool(processes=n_proc) # for function evaluations
        pool_rbf = Pool(processes=n_proc)
        
    # sanity check
    assert(n_proc>1)
    assert(t_eval_func is None or t_eval_func>0)
    assert(0<max_t_opt_ratio<1)
    assert(max_C_lf_fail > 0 and max_C_lf_success > 0)
    assert(wgt_expon_delta >= 0 and max_reduce_step_size>=0 and init_gSRS_pct>=0 and init_gSRS_pct<=1)  
    assert(0<box_fact<1 and resol>0 and min_zoom_in>0 and x_star_std_factor>=0 and max_C_expand_fit>0)
    
    # check if the output directory exists
    assert(os.path.isdir(outdir))
        
    # log file (write standard output to a file)
    log_std_out = verbose or debug if log_std_out is None else log_std_out
    if log_std_out:
        orig_std_out = sys.stdout
        log_file = os.path.join(outdir, 'std_output_log_%s.txt' % prob_name)
        if resume_opt_iter is None:
            if os.path.isfile(log_file):
                os.remove(log_file)
        sys.stdout = std_out_logger(log_file)    
    if log_std_err:
        orig_std_err = sys.stderr
        log_file = os.path.join(outdir, 'std_error_log_%s.txt' % prob_name)
        if resume_opt_iter is None:
            if os.path.isfile(log_file):
                os.remove(log_file)
        sys.stderr = std_err_logger(log_file)
    
    ########################### Functions ###############################
    
    def propose(gSRS_pct,x_star,rbf_mod,step_size_fact,n_zoom,n_prop,
                status_var_list,seq,intr_iter,n_iter,X_samp_hf=None):
        '''
        Propose points using ProSRS method.
        Input:
            gSRS_pct: float
            x_star: 1d array
            rbf_mod: function
            step_size_fact: float
            n_zoom: int
            n_prop: int
            status_var_list: list of status variables
            seq: None or int, used for debugging
            intr_iter: int, intrinsic iteration, used for debugging
            n_iter: int, current number of iterations, used for debugging
            X_samp_hf: None or 2d array, used for debugging
        Output:
            prop_pt_arr: 2d array
            expand_status: int
            expand_mag: float
            status_var_list: list of updated status variables
            seq: None or int
        '''
        # unpack status variables
        fit_bd,X_samp,Y_samp,n_expand,X_samp_out,Y_samp_out = status_var_list
        
        expand_status,expand_mag = 0,0
        
        if gSRS_pct == 1:
            
            #### pure global SRS ####
            
            # generate candidate points uniformly (global SRS)
            cand_pt = np.zeros((n_cand,dim))
            for d,bd in enumerate(fit_bd):
                cand_pt[:,d] = np.random.uniform(low=bd[0],high=bd[1],size=n_cand)
            prop_pt_arr = SRS(rbf_mod,cand_pt,X_samp,n_prop,wgt_pat_bd)
                    
        else:
            #### global-local SRS (possibly pure local SRS) ####
                
            # find step size (i.e. std) for each coordinate of x_star
            step_size_arr = np.array([step_size_fact*(x[1]-x[0]) for x in fit_bd])
            assert(np.min(step_size_arr)>0) # sanity check
            
            # expand fit_bd
            if n_zoom > 0 or soft_func_bd:
                fit_lb, fit_ub = zip(*fit_bd)
                fit_lb, fit_ub = np.array(fit_lb), np.array(fit_ub) # convert to numpy array
                assert(np.all(fit_lb<=x_star) and np.all(x_star<=fit_ub)) # sanity check
                if soft_func_bd:
                    x_star_lw_bd = x_star-step_size_arr*x_star_std_factor
                    x_star_up_bd = x_star+step_size_arr*x_star_std_factor
                else:
                    x_star_lw_bd = np.maximum(x_star-step_size_arr*x_star_std_factor,func_lb)
                    x_star_up_bd = np.minimum(x_star+step_size_arr*x_star_std_factor,func_ub)
                
                if debug and visual_debug and dim in [1,2]:
                    assert(seq is not None and seq>=0)
                    assert(intr_iter is not None and intr_iter>=0)
                    seq += 1
                    nx = 100
                    plt.figure(0)
                    plt.clf()
                    if dim == 2:
                        x1_arr = np.linspace(fit_bd[0][0],fit_bd[0][1],nx)
                        x2_arr = np.linspace(fit_bd[1][0],fit_bd[1][1],nx)
                        x1_resol, x2_resol = x1_arr[1]-x1_arr[0], x2_arr[1]-x2_arr[0]
                        X1, X2 = np.meshgrid(x1_arr, x2_arr)
                        pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                        # fit function contour with evaluated samples
                        fit_val = rbf_mod(pts)
                        if len(set(fit_val)) > 1:
                            plt.contour(X1,X2,fit_val.reshape((nx,nx)),20)
                            plt.colorbar()
                        if X_samp_hf is not None:
                            plt.plot(X_samp[:,0],X_samp[:,1],'o',color='gray') # low fidelity samples
                            plt.plot(X_samp_hf[:,0],X_samp_hf[:,1],'ko')
                        else:    
                            plt.plot(X_samp[:,0],X_samp[:,1],'ko')
                        # x_star
                        plt.plot(x_star[0],x_star[1],'rx',mew=1) # mew: marker edge width
                        # x_star bound
                        bd1, bd2 = x_star_lw_bd.copy(), x_star_up_bd.copy()
                        plt.plot([bd1[0],bd1[0],bd2[0],bd2[0],bd1[0]],[bd1[1],bd2[1],bd2[1],bd1[1],bd1[1]],
                                 color='gray',ls='--')
                        plt.xlabel(prob.x_var[0], fontsize=label_fs)
                        plt.ylabel(prob.x_var[1], fontsize=label_fs)
                        
                        
                    elif dim == 1:
                        pts = np.linspace(fit_bd[0][0], fit_bd[0][1], nx).reshape((-1,1))
                        x_resol = pts[1,0]-pts[0,0]
                        fit_val = rbf_mod(pts)
                        plt.plot(pts,fit_val,'b-')
                        if X_samp_hf is not None:
                            plt.plot(X_samp,rbf_mod(X_samp),'o',color='gray') # low fidelity samples
                            plt.plot(X_samp_hf,rbf_mod(X_samp_hf),'ko')
                        else:    
                            plt.plot(X_samp,rbf_mod(X_samp),'ko')
                        # x_star
                        plt.plot(x_star,rbf_mod(x_star.reshape((-1,1))),'rx',mew=1) # mew: marker edge width
                        # x_star bound
                        plt.axvline(x=x_star_lw_bd,color='gray',ls='--')
                        plt.axvline(x=x_star_up_bd,color='gray',ls='--')
                        plt.xlabel(prob.x_var[0], fontsize=label_fs)
                        plt.ylabel(prob.y_var, fontsize=label_fs)
                    
                    ax = plt.gca()
                    plt.text(0.5, 1.03, 'Iteration %d: check expansion\n' % intr_iter,
                             horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                             fontweight='bold')
                    plt.title('gSRS_p = %g, expand = %d, zoom = %d' % (gSRS_pct, n_expand, n_zoom), 
                              fontsize=title_fs, y = 1.01)
                    plt.savefig('%s/%s_iter%d_seq%d_check_expand.pdf' % (outdir_debug,prob_name,n_iter,seq))
                    plt.close()
                        
                if np.any(x_star_lw_bd<fit_lb) or np.any(x_star_up_bd>fit_ub):
                    n_expand += 1 # update counter
                    expand_status = 1
                    dist_cent = np.linalg.norm((fit_ub-fit_lb)/2.) # corner distance to center
                    dist_x_star = np.linalg.norm(x_star-(fit_ub+fit_lb)/2.) # distance from x_star to center
                    expand_mag = dist_x_star/dist_cent
                    assert(0<=expand_mag_arr[k]<=1+1e-10),'dist_center = %.3e, dist_x_star = %.3e' % (dist_cent, dist_x_star) # sanity check
                    if verbose:
                        print('Expand fit bound!')
                    fit_lb = np.minimum(x_star_lw_bd,fit_lb)
                    fit_ub = np.maximum(x_star_up_bd,fit_ub)
                    fit_bd = list(zip(fit_lb,fit_ub)) # new fit bound
                    fit_bd_save[n_zoom] = fit_bd # update fit_bd_save
                    # update in-box and out-of-box samples
                    tot_X_samp = np.vstack((X_samp,X_samp_out))
                    tot_Y_samp = np.append(Y_samp,Y_samp_out)
                    in_box_ix, out_box_ix = get_box_samp(tot_X_samp,fit_bd)
                    X_samp, X_samp_out = tot_X_samp[in_box_ix], tot_X_samp[out_box_ix]
                    Y_samp, Y_samp_out = tot_Y_samp[in_box_ix], tot_Y_samp[out_box_ix]  

                    if debug and dim in [1,2] and visual_debug:
                        seq += 1
                        plt.figure(0)
                        plt.clf()
                        if dim == 2:
                            nx1 = int(round((fit_bd[0][1]-fit_bd[0][0])/x1_resol))+1
                            nx2 = int(round((fit_bd[1][1]-fit_bd[1][0])/x2_resol))+1
                            x1_arr = np.linspace(fit_bd[0][0],fit_bd[0][1],nx1)
                            x2_arr = np.linspace(fit_bd[1][0],fit_bd[1][1],nx2)
                            X1, X2 = np.meshgrid(x1_arr, x2_arr)
                            pts = np.hstack((X1.reshape(nx1*nx2,1),X2.reshape(nx1*nx2,1)))
                            # fit function contour with evaluated samples
                            fit_val = rbf_mod(pts)
                            if len(set(fit_val)) > 1:
                                plt.contour(X1,X2,fit_val.reshape((nx2,nx1)),20)
                                plt.colorbar()
                            if X_samp_hf is not None:
                                plt.plot(X_samp[:,0],X_samp[:,1],'o',color='gray') # low fidelity samples
                                plt.plot(X_samp_hf[:,0],X_samp_hf[:,1],'ko')
                            else:    
                                plt.plot(X_samp[:,0],X_samp[:,1],'ko')
                            # x_star
                            plt.plot(x_star[0],x_star[1],'rx',mew=1) # mew: marker edge width
                            # x_star bound
                            bd1, bd2 = x_star_lw_bd.copy(), x_star_up_bd.copy()
                            plt.plot([bd1[0],bd1[0],bd2[0],bd2[0],bd1[0]],[bd1[1],bd2[1],bd2[1],bd1[1],bd1[1]],
                                     color='gray',ls='--')
                            plt.xlabel(prob.x_var[0], fontsize=label_fs)
                            plt.ylabel(prob.x_var[1], fontsize=label_fs)
                        
                        elif dim == 1:
                            nx = int(round((fit_bd[0][1]-fit_bd[0][0])/x_resol))+1
                            pts = np.linspace(fit_bd[0][0], fit_bd[0][1], nx).reshape((-1,1))
                            fit_val = rbf_mod(pts)
                            plt.plot(pts,fit_val,'b-')
                            if X_samp_hf is not None:
                                plt.plot(X_samp,rbf_mod(X_samp),'o',color='gray') # low fidelity samples
                                plt.plot(X_samp_hf,rbf_mod(X_samp_hf),'ko')
                            else:    
                                plt.plot(X_samp,rbf_mod(X_samp),'ko')
                            # x_star
                            plt.plot(x_star,rbf_mod(x_star.reshape((-1,1))),'rx',mew=1) # mew: marker edge width
                            # x_star bound
                            plt.axvline(x=x_star_lw_bd,color='gray',ls='--')
                            plt.axvline(x=x_star_up_bd,color='gray',ls='--')
                            plt.xlabel(prob.x_var[0], fontsize=label_fs)
                            plt.ylabel(prob.y_var, fontsize=label_fs)
                                
                        ax = plt.gca()
                        plt.text(0.5, 1.03, 'Iteration %d: expand domain\n' % intr_iter,
                                 horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                                 fontweight='bold')
                        plt.title('gSRS_p = %g, expand = %d, zoom = %d' % (gSRS_pct, n_expand, n_zoom), 
                                  fontsize=title_fs, y = 1.01)
                        plt.savefig('%s/%s_iter%d_seq%d_expand.pdf' % (outdir_debug,prob_name,n_iter,seq))
                        plt.close()
                        
                else:
                    n_expand = 0 # reset counter

            n_cand_gSRS = int(np.round(n_cand*gSRS_pct))
            n_cand_lSRS = n_cand-n_cand_gSRS
            assert (n_cand_lSRS>0) # sanity check
            
            # generate candidate points uniformly (global SRS)           
            cand_pt_gSRS = np.zeros((n_cand_gSRS,dim))
            if n_cand_gSRS>0:
                for d,bd in enumerate(fit_bd):
                    cand_pt_gSRS[:,d] = np.random.uniform(low=bd[0],high=bd[1],size=n_cand_gSRS)
                        
            # generate candidate points (gaussian about x_star, local SRS)
            cand_pt_lSRS = np.random.multivariate_normal(x_star,\
                                                         np.diag(step_size_arr**2),n_cand_lSRS)
            # combine two types of candidate points
            cand_pt = np.vstack((cand_pt_gSRS,cand_pt_lSRS))
            cand_pt, cand_pt_raw = put_back_box(cand_pt,fit_bd)
            if len(cand_pt) >= n_prop:
                prop_pt_arr = SRS(rbf_mod,cand_pt,X_samp,n_prop,wgt_pat_bd)
            else:
                # this rarely happens, then we use raw candidate point (possibly with duplicate points)
                prop_pt_arr = SRS(rbf_mod,cand_pt_raw,X_samp,n_prop,wgt_pat_bd)
            
        # pack status variables
        status_var_list = [fit_bd,X_samp,Y_samp,n_expand,X_samp_out,Y_samp_out]
            
        return prop_pt_arr,expand_status,expand_mag,status_var_list,seq
    
    
    def prepare(gSRS_pct,best_Y_new,best_Y_prev,x_star,status_var_list):
        '''
        Prepare for next iteration.
        Input:
            gSRS_pct: float
            best_Y_new: float
            best_Y_prev: float
            x_star: 1d array or None
            status_var_list: list of status variables
        Output:
            status_var_list: list of updated status variables
        '''
        # unpack status variables
        gSRS_pct_full,n_expand,n_zoom_out,n_zoom,fit_bd_save,\
        fit_bd,X_samp,Y_samp,X_samp_out,Y_samp_out,wgt_expon,\
        n_reduce_step_size,step_size_fact,n_fail,full_restart,n_full_restart,\
        x_star_list = status_var_list
        
        if not full_restart:
            
            if n_expand == max_C_expand_fit and n_zoom_out[n_zoom] < max_zoom_out and n_zoom > 0:
                # then zoom-out
                n_zoom_out[n_zoom] += 1 # update counter
                n_zoom -= 1 # update zoom level
                assert(n_zoom >= 0) # sanity check
                fit_save_lb, fit_save_ub = zip(*fit_bd_save[n_zoom])
                fit_save_lb, fit_save_ub = np.array(fit_save_lb), np.array(fit_save_ub) # convert to numpy array
                fit_lb, fit_ub = zip(*fit_bd)
                fit_lb, fit_ub = np.array(fit_lb), np.array(fit_ub) # convert to numpy array
                fit_lb = np.minimum(fit_save_lb,fit_lb)
                fit_ub = np.maximum(fit_save_ub,fit_ub)
                fit_bd = list(zip(fit_lb,fit_ub)) # new fit bound
                fit_bd_save[n_zoom] = fit_bd # update fit_bd_save
                # update in-box and out-of-box samples
                tot_X_samp = np.vstack((X_samp,X_samp_out))
                tot_Y_samp = np.append(Y_samp,Y_samp_out)
                in_box_ix, out_box_ix = get_box_samp(tot_X_samp,fit_bd)
                X_samp, X_samp_out = tot_X_samp[in_box_ix], tot_X_samp[out_box_ix]
                Y_samp, Y_samp_out = tot_Y_samp[in_box_ix], tot_Y_samp[out_box_ix]
                # update other parameters
                wgt_expon = init_wgt_expon
                n_reduce_step_size = 0
                gSRS_pct_full = init_gSRS_pct
                step_size_fact = init_step_size_fact
                n_expand = 0
                n_fail = 0
                x_star_list = [None,None]
                if verbose:
                    print('Zoom-out!')
                
            elif int(np.round(n_cand*gSRS_pct)) == 0: # pure local SRS
                assert(best_Y_prev is not None)
                if best_Y_prev <= best_Y_new: # failure                
                    n_fail += 1 # count failure
                    if n_fail == max_C_fail:
                        n_fail = 0 # reset counter
                        if n_reduce_step_size < max_reduce_step_size:
                            # then increase wgt_expon and reduce step size
                            wgt_expon += wgt_expon_delta
                            step_size_fact /= 2.
                            n_reduce_step_size += 1 # update counter
                        else:
                            # check restart condition
                            fit_lb, fit_ub = zip(*fit_bd)
                            blen = np.array(fit_ub)-np.array(fit_lb) # bound length for each dimension
                            assert (np.min(blen)>0)
                            resol_cond = np.all(blen/cand_density<func_blen*resol) # resolution condition
                            if resol_cond and n_zoom >= min_zoom_in:
                                # then restart
                                if verbose:
                                    print('Restart for the next iteration!')                        
                                full_restart = True
                                # update samples
                                X_samp = np.zeros((0,dim))
                                Y_samp = np.zeros(0)
                                X_samp_out = np.zeros((0,dim))
                                Y_samp_out = np.zeros(0)                    
                                # update other parameters
                                n_zoom = 0
                                n_expand = 0
                                fit_bd = func_bd
                                fit_bd_save = {n_zoom: fit_bd}
                                wgt_expon = init_wgt_expon
                                n_reduce_step_size = 0
                                gSRS_pct_full = init_gSRS_pct
                                step_size_fact = init_step_size_fact
                                n_zoom_out = {n_zoom: 0}
                                x_star_list = [None, None]
                            else:
                                # then zoom-in
                                n_zoom += 1
                                # determine new function bounds
                                if dynam_box_fact:
                                    fit_lb = np.maximum(x_star-box_fact**n_zoom/2.0*blen,fit_lb)
                                    fit_ub = np.minimum(x_star+box_fact**n_zoom/2.0*blen,fit_ub)
                                else:
                                    fit_lb = np.maximum(x_star-box_fact/2.0*blen,fit_lb)
                                    fit_ub = np.minimum(x_star+box_fact/2.0*blen,fit_ub)
                                fit_bd = list(zip(fit_lb,fit_ub)) # the list function is used to ensure compatibility of python3
                                # update in-box and out-of-box samples
                                tot_X_samp = np.vstack((X_samp,X_samp_out))
                                tot_Y_samp = np.append(Y_samp,Y_samp_out)
                                in_box_ix, out_box_ix = get_box_samp(tot_X_samp,fit_bd)
                                X_samp, X_samp_out = tot_X_samp[in_box_ix], tot_X_samp[out_box_ix]
                                Y_samp, Y_samp_out = tot_Y_samp[in_box_ix], tot_Y_samp[out_box_ix]
                                # update other parameters
                                wgt_expon = init_wgt_expon
                                n_reduce_step_size = 0
                                gSRS_pct_full = init_gSRS_pct
                                step_size_fact = init_step_size_fact
                                fit_bd_save[n_zoom] = fit_bd
                                n_expand = 0
                                x_star_list = [None,None]
                                if n_zoom not in n_zoom_out.keys():
                                    n_zoom_out[n_zoom] = 0 # initialize counter
                                    
                else:
                    # i.e. success. Then we reset failure counter
                    n_fail = 0
                    
        else:
            # doing full restart
            n_full_restart += 1          
            if n_full_restart == init_iter:
                full_restart = False
                n_full_restart = 0
        
        # pack status variables
        status_var_list = [gSRS_pct_full,n_expand,n_zoom_out,n_zoom,fit_bd_save,
                           fit_bd,X_samp,Y_samp,X_samp_out,Y_samp_out,wgt_expon,
                           n_reduce_step_size,step_size_fact,n_fail,full_restart,
                           n_full_restart,x_star_list]
        
        return status_var_list
    
    
    def program_init():
        '''
        Program initialization
        Output:
            multi_fidelity: boolean
            status_param_list: list of parameters
        '''
        multi_fidelity = len(obj_func_list)>1 # multi fidelity indicator
    
        if multi_fidelity:
            
            # effective low fidelity efficiency
            lf_eff = np.floor(eff_list[-1])
            
            # check condition for multi fidelity
            if lf_eff >= 2:
                # then we continue with multi fidelity
                if verbose:
                    print('Multi-fidelity optimization in operation!')
                # find initial number of processors for high fidelity and low fidelity
                n_proc_hf = int(np.floor(lf_eff*n_proc/(lf_eff+1.)))
                assert(n_proc_hf >= min_n_proc_hf)
                n_proc_lf = n_proc-n_proc_hf
                assert(n_proc_lf > 0)
                # find number of workers
                n_worker_hf = n_proc_hf
                n_worker_per_proc_lf = int(lf_eff) # number of workers per processor
                n_worker_lf = n_proc_lf*n_worker_per_proc_lf
                assert(n_worker_lf >= n_worker_hf)
            else:
                # then we change to single fidelity
                if verbose:
                    print('Effective low fidelity efficiency is less than 2! Fall back to single fidelity!')
                n_worker = n_proc
                multi_fidelity = False        
        else:
            if verbose:
                print('Single-fidelity optimization in operation!')
            n_worker = n_proc
        
        if multi_fidelity:
            obj_func_lf = obj_func_list[-1]
            obj_func_hf = obj_func_list[0]
            assert(n_cand_fact*dim >= n_worker_lf), 'number of candidate points needs to be no less than number of workers'
        else:
            obj_func = obj_func_list[0]
            assert(n_cand_fact*dim >= n_worker), 'number of candidate points needs to be no less than number of workers'
        
        if multi_fidelity:
            n_worker,obj_func = [None]*2
        else:
            n_proc_hf,n_proc_lf,n_worker_hf,n_worker_per_proc_lf,n_worker_lf,obj_func_lf,obj_func_hf = [None]*7
        
        status_param_list = [n_worker,n_proc_hf,n_proc_lf,n_worker_hf,n_worker_per_proc_lf,n_worker_lf,
                             obj_func,obj_func_lf,obj_func_hf]
        
        return multi_fidelity,status_param_list
    
    
    def check_lf_acc(X_lf,Y_lf,X_hf,Y_hf,hf_prop_arr,fail_lf_acc,success_lf_acc):
        '''
        Check low fidelity model accuracy.
        Input:
            X_lf: 2d array
            Y_lf: 1d array
            X_hf: 2d array
            Y_hf: 1d array
            hf_prop_arr: 1d array
            fail_lf_acc: list of (boolean or None)
            success_lf_acc: list of (boolean or None)
        Output:
            fail_lf_acc: list of (boolean or None)
            success_lf_acc: list of (boolean or None)
            
        '''
        n_hf = len(X_hf)
        # sanity check
        assert(len(X_lf)==len(Y_lf) and n_hf==len(Y_hf))
        
        # parameter
        use_fit = True
        
        t1 = default_timer()
        
        if n_hf >= min_lf_test_samp:
            # find samples of recent iterations that have samples >= min_lf_test_samp
            cum_n_samp = np.cumsum([i for i in reversed(hf_prop_arr)])*n_proc
            ix = np.argmax(cum_n_samp>=min_lf_test_samp)
            n_samp = int(cum_n_samp[ix])
            assert(n_hf>=n_samp>=min_lf_test_samp)
            X_hf, Y_hf = X_hf[-n_samp:], Y_hf[-n_samp:]
            
            # find low fidelity Y values that correspond to X_hf
            ix_lf_at_hf = subset_arr_ix(X_lf,X_hf)
            Y_lf_at_hf = Y_lf[ix_lf_at_hf]
            
            if use_fit:
                
                rbf_lf,_,_,_ = RBF_reg(X_hf,Y_lf_at_hf,n_proc,sm_range,normalize_data=normalize_data,\
                                       kernel=rbf_kernel,wgt_expon=0,log_opt=log_opt,\
                                       n_fold=n_fold,pool=pool_rbf)
                
                rbf_hf,_,_,_ = RBF_reg(X_hf,Y_hf,n_proc,sm_range,normalize_data=normalize_data,\
                                       kernel=rbf_kernel,wgt_expon=0,log_opt=log_opt,\
                                       n_fold=n_fold,pool=pool_rbf)
                
                Y_fit_lf = rbf_lf(X_hf)
                Y_fit_hf = rbf_hf(X_hf)
                
                rho,p_val = stats.spearmanr(Y_fit_lf,Y_fit_hf)
                
            else:
                
                rho,p_val = stats.spearmanr(Y_lf_at_hf,Y_hf)
            
            if np.isnan(rho) or np.isnan(p_val):
                valid_lf = None
            else:
                if p_val >= lf_test_alpha:
                    # not statistically significant
                    valid_lf = None
                elif rho > 0:
                    valid_lf = True 
                elif rho < 0:
                    valid_lf = False
                else:
                    # rho = 0 and statistically significant
                    valid_lf = None
        else:
            valid_lf = None
                
        t2 = default_timer()
        
        if verbose:
            print('time to check accuracy of low fidelity model = %.2e sec' % (t2-t1))
            if valid_lf is not None:
                print('pass accuracy test: %d' % valid_lf)
                print('correlation coefficient: %.2e' % rho)
                print('p value: %.2e' % p_val)
                print('number of samples for test: %d' % n_samp)
            else:
                if n_hf < min_lf_test_samp:
                    print('unable conduct accuracy test (too few samples)')
                else:
                    if np.isnan(rho) or np.isnan(p_val):
                        print('unable conduct accuracy test')
                    else:
                        print('statistically uncorrelated (p_val = %.2e)' % p_val)
        
        # save test results
        fail_lf_acc[1:] = fail_lf_acc[:-1] # update previous results
        fail_lf_acc[0] = not valid_lf if valid_lf is not None else None # update current result
        success_lf_acc[1:] = success_lf_acc[:-1]
        success_lf_acc[0] = valid_lf
            
        return fail_lf_acc, success_lf_acc
    
    
    def get_alpha(x_star_list, bd):
        '''
        Get alpha value for gSRS_pct_full dynamics
        Input:
            x_star_list: list of two (1d arrays or None)
            bd: list of tuples
        Output:
            alpha: float
            x_star_err: float
        '''
        x_star_curr, x_star_last = x_star_list
        assert(x_star_curr is not None)
        if x_star_last is None:
            x_star_err = np.nan
            alpha = 0.
        else:
            bd_len = np.array([b[1]-b[0] for b in bd]) # bound length for each axis
            x_star_err = np.absolute(x_star_curr-x_star_last)*1.
            x_star_err = np.max(x_star_err/bd_len)
            assert(0<=x_star_err<=1)
            alpha = alpha_base+alpha_amp*(alpha_crit_err-x_star_err)
        
        return alpha, x_star_err
        
    ########################### Main program ############################
    
    # initialize program
    multi_fidelity,status_param_list = program_init()
    n_worker,n_proc_hf,n_proc_lf,n_worker_hf,n_worker_per_proc_lf,n_worker_lf,\
    obj_func,obj_func_lf,obj_func_hf = status_param_list
    
    intr_iter = None # intrinsic iteration, used for debugging
    if debug:
        print('Initiate debugging mode for current repeat!')
        if visual_debug and dim in [1,2]:
            if not os.path.isdir(outdir_debug):
                os.makedirs(outdir_debug)
            intr_iter = 0 # count intrinsic number of iterations
        if multi_fidelity:
            assert(os.path.isfile(result_npz_file))
            data_debug = np.load(result_npz_file)
            high_opt_debug = data_debug['high_opt_arr']
            assert(len(high_opt_debug)>=opt_iter), 'To debug multi fidelity optimization,'\
                                 +' do not set opt_iter larger than previous opt_iter.'
    
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
    n_expand_arr = np.zeros(opt_iter) # save n_expand in each iteration
    expand_status_arr = np.zeros(opt_iter) # save the status of expanding fit bound in each iteration
    expand_mag_arr = np.zeros(opt_iter) # save the magnitude of expanding fit bound in each iteration
    high_opt_arr = [None]*opt_iter # save high optimization usage status, USEFUL when try to reproduce results
    fidelity_status_arr = np.zeros(init_iter+opt_iter) # save multi-fidelity status for each iteration
    hf_prop_arr = np.ones_like(fidelity_status_arr) # save proportion of high fidelity optimization in terms of number of processors
    best_loc_it_lf = np.zeros((init_iter+opt_iter,dim)) # save best points for each iteration
    best_val_it_lf = np.zeros(init_iter+opt_iter) # save Y value for the best point for each iteration
    best_seed_it_lf = np.zeros(init_iter+opt_iter,dtype=int) # save random seed corresponding to the best point for each iteration
    best_loc_it_hf = np.zeros((init_iter+opt_iter,dim)) # save best points for each iteration
    best_val_it_hf = np.zeros(init_iter+opt_iter) # save Y value for the best point for each iteration
    best_seed_it_hf = np.zeros(init_iter+opt_iter,dtype=int) # save random seed corresponding to the best point for each iteration
    best_loc_it = np.zeros((init_iter+opt_iter,dim)) # save best points for each iteration
    best_val_it = np.zeros(init_iter+opt_iter) # save Y value for the best point for each iteration
    best_seed_it = np.zeros(init_iter+opt_iter,dtype=int) # save random seed corresponding to the best point for each iteration
    
    if resume_opt_iter is None:
        
        ########### Initial sampling (DOE) ################ 
        
        np.random.seed(int(seed))
        
        t1 = default_timer()
        t1_c = clock()
        
        if multi_fidelity:
            n_init_samp_lf = n_worker_lf*init_iter
            n_init_samp_hf = n_worker_hf*init_iter
            X_samp_lf = doe(n_init_samp_lf,func_bd) 
            X_samp_hf = sub_doe(X_samp_lf,n_init_samp_hf)
            X_samp = None
        else:
            n_init_samp = n_worker*init_iter
            X_samp = doe(n_init_samp,func_bd)
            X_samp_lf, X_samp_hf = None, None
            
        t2 = default_timer()
        t2_c = clock()
        
        t_doe = t2-t1
        t_doe_cpu = t2_c-t1_c
        
        ########### Evaluate DOE ################     
    
        
        t_eval = np.zeros(init_iter)
        t_eval_cpu = np.zeros(init_iter)
        
        seed_base = seed+1
        seed_flat_arr = np.array([])
        
        if multi_fidelity:
            
            Y_samp = None
            # generate list of objective function list
            f_list_list = [[obj_func_lf]*n_worker_per_proc_lf]*n_proc_lf+[[obj_func_hf]]*n_proc_hf
            # initialization
            Y_samp_lf = np.zeros(n_init_samp_lf)
            Y_samp_hf = np.zeros(n_init_samp_hf)
            for k in range(init_iter):
                ix1_lf,ix2_lf = k*n_worker_lf,(k+1)*n_worker_lf
                ix1_hf,ix2_hf = k*n_worker_hf,(k+1)*n_worker_hf
                X_lf = X_samp_lf[ix1_lf:ix2_lf]
                X_hf = X_samp_hf[ix1_hf:ix2_hf]
                # initialization
                x_list_list = []
                seed_list_list = []
                out_list_list = []
                for p in range(n_proc_lf):
                    ix1_plf, ix2_plf = p*n_worker_per_proc_lf,(p+1)*n_worker_per_proc_lf
                    x_list_list.append(X_lf[ix1_plf:ix2_plf])
                    seed_arr_lf = seed_base+np.arange(ix1_plf,ix2_plf)
                    seed_list_list.append(seed_arr_lf)
                    out_list = [os.path.join(outdir, RESULT_LF_SAMP_FILE_TEMP % (prob_name,k+1,n+1)) \
                                for n in range(ix1_plf,ix2_plf)]
                    out_list_list.append(out_list)
                seed_base += n_worker_lf
                for p in range(n_proc_hf):
                    x_list_list.append(X_hf[p:p+1])
                    seed_arr_hf = seed_base+np.arange(p,p+1)
                    seed_list_list.append(seed_arr_hf)
                    out_list = [os.path.join(outdir, RESULT_HF_SAMP_FILE_TEMP % (prob_name,k+1,p+1))]
                    out_list_list.append(out_list)
                seed_base += n_worker_hf
                # update seed_flat_arr for sanity check
                seed_flat = seed_list_list[0]
                for seed_arr in seed_list_list[1:]:
                    seed_flat = np.append(seed_flat,seed_arr)
                seed_flat_arr = np.append(seed_flat_arr,seed_flat)
                # get seeds of each X for low-fidelity and high-fidelity
                seed_X_lf = np.array(seed_list_list[:n_proc_lf]).reshape((1,-1)).flatten()
                seed_X_hf = np.array(seed_list_list[n_proc_lf:]).flatten()
                assert(seed_X_hf.shape == (n_worker_hf,) and seed_X_lf.shape == (n_worker_lf,))
                # evaluate function
                t1, t1_c = default_timer(), clock()
                Y_list = func_eval(f_list_list,x_list_list,seed_list_list,out_list_list,pool_eval)
                t2, t2_c = default_timer(), clock()
                t_eval[k], t_eval_cpu[k] = t2-t1, t2_c-t1_c
                # unpack results
                Y_list_lf, Y_list_hf = Y_list[:n_proc_lf], Y_list[n_proc_lf:]
                Y_lf = []
                for yl in Y_list_lf:
                    Y_lf += yl
                Y_hf = []
                for yl in Y_list_hf:
                    Y_hf += yl
                assert(len(X_lf)==len(Y_lf) and len(X_hf)==len(Y_hf))
                # get best value and best location for each iteration
                min_ix = np.argmin(Y_lf)
                best_loc_it_lf[k],best_val_it_lf[k] = X_lf[min_ix],Y_lf[min_ix]
                best_seed_it_lf[k] = seed_X_lf[min_ix]
                min_ix = np.argmin(Y_hf)
                best_loc_it_hf[k],best_val_it_hf[k] = X_hf[min_ix],Y_hf[min_ix]
                best_seed_it_hf[k] = seed_X_hf[min_ix]
                # save Y
                Y_samp_lf[ix1_lf:ix2_lf] = Y_lf
                Y_samp_hf[ix1_hf:ix2_hf] = Y_hf
                # save variables
                fidelity_status_arr[k] = multi_fidelity
                hf_prop_arr[k] = n_proc_hf/float(n_proc_hf+n_proc_lf)
                assert(hf_prop_arr[k]>=0.5),hf_prop_arr[k]
                
                if verbose:
                    print('hf_prop = %g' % hf_prop_arr[k])
                
                if debug and dim in [1,2] and visual_debug:
                    intr_iter += 1
                    nx = 100
                    plt.figure(0)
                    plt.clf()
                    if dim == 2:
                        try:
                            # Note: for some type of problem, true function (e.g., 'hf') may not exist
                            func_obj = prob.func_obj_list[0] # function object
                            # get true function
                            true_func = func_obj.hf
                            x1_arr = np.linspace(func_bd[0][0],func_bd[0][1],nx)
                            x2_arr = np.linspace(func_bd[1][0],func_bd[1][1],nx)
                            X1, X2 = np.meshgrid(x1_arr, x2_arr)
                            pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                            true_val = true_func(pts)
                            true_val = true_val.ravel()
                            # true function contour with evaluated samples
                            if len(set(true_val)) > 1:
                                plt.contour(X1,X2,true_val.reshape((nx,nx)),20)
                                plt.colorbar()
                        except:
                            pass
                        plt.plot(X_samp_lf[:ix2_lf,0],X_samp_lf[:ix2_lf,1],'o',color='gray')
                        plt.plot(X_samp_hf[:ix2_hf,0],X_samp_hf[:ix2_hf,1],'ko')
                        plt.xlim(func_bd[0])
                        plt.ylim(func_bd[1])
                        plt.xlabel(prob.x_var[0], fontsize=label_fs)
                        plt.ylabel(prob.x_var[1], fontsize=label_fs)
                            
                    elif dim == 1:
                        try:
                            # get true function
                            func_obj = prob.func_obj_list[0] # function object
                            true_func_hf = func_obj.hf
                            func_obj = prob.func_obj_list[-1] # function object
                            true_func_lf = func_obj.lf
                            pts = np.linspace(func_bd[0][0], func_bd[0][1], nx)
                            # true function with evaluated samples
                            plt.plot(pts,true_func_hf(pts).ravel(),'r-')
                            plt.plot(pts,true_func_lf(pts).ravel(),'b-')
                            plt.plot(X_samp_lf[:ix2_lf],true_func_lf(X_samp_lf[:ix2_lf]),'o',color='gray')
                            plt.plot(X_samp_hf[:ix2_hf],true_func_lf(X_samp_hf[:ix2_hf]),'ko')
                            plt.ylabel('true %s' % prob.y_var, fontsize=label_fs)
                        except:
                            # then we plot noisy evaluations with samples
                            plt.plot(X_samp_lf[:ix2_lf],Y_samp_lf[:ix2_lf],'o',color='gray')
                            plt.plot(X_samp_hf[:ix2_hf],Y_samp_hf[:ix2_hf],'ko')
                            plt.ylabel('noisy %s' % prob.y_var, fontsize=label_fs)    
                        plt.xlabel(prob.x_var[0], fontsize=label_fs)
                        plt.xlim(func_bd[0])
                            
                    ax = plt.gca()
                    plt.text(0.5, 1.03, 'Iteration %d: design of experiments' % intr_iter,
                                 horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                                 fontweight='bold')
                    plt.savefig('%s/%s_iter%d_doe.pdf' % (outdir_debug,prob_name,k+1))
                    plt.close()
    
                # save to netcdf file
                if save_samp:
                    out_file = os.path.join(outdir, RESULT_GENERAL_SAMP_FILE_TEMP % (prob_name,k+1))
                    if os.path.isfile(out_file):
                        os.remove(out_file) # remove file if exists
                    with netcdf.netcdf_file(out_file,'w') as f:
                        f.createDimension('var',dim)
                        f.createDimension('samp_lf',n_worker_lf)
                        f.createDimension('samp_hf',n_worker_hf)
                        nc_multi_fidelity = f.createVariable('multi_fidelity','d',())
                        nc_multi_fidelity.assignValue(multi_fidelity)
                        nc_samp = f.createVariable('x_hf','d',('samp_hf','var'))
                        nc_samp[:] = X_hf
                        nc_samp = f.createVariable('y_hf','d',('samp_hf',))
                        nc_samp[:] = Y_hf
                        nc_samp = f.createVariable('x_lf','d',('samp_lf','var'))
                        nc_samp[:] = X_lf
                        nc_samp = f.createVariable('y_lf','d',('samp_lf',))
                        nc_samp[:] = Y_lf
                        nc_seed = f.createVariable('seed_hf','i',('samp_hf',))
                        nc_seed[:] = seed_X_hf
                        nc_seed = f.createVariable('seed_lf','i',('samp_lf',))
                        nc_seed[:] = seed_X_lf
                        nc_wall_time = f.createVariable('tw_eval','d',())
                        nc_wall_time.assignValue(t_eval[k])
                        nc_cpu_time = f.createVariable('tc_eval','d',())
                        nc_cpu_time.assignValue(t_eval_cpu[k])
                        nc_wall_time = f.createVariable('tw_doe','d',())
                        nc_wall_time.assignValue(t_doe)
                        nc_cpu_time = f.createVariable('tc_doe','d',())
                        nc_cpu_time.assignValue(t_doe_cpu)
        else:
            
            Y_samp_lf, Y_samp_hf = None, None
            # generate list of objective function list
            f_list_list = [[obj_func]]*n_worker
            # initialization
            Y_samp = np.zeros(n_init_samp)
            for k in range(init_iter):
                ix1,ix2 = k*n_worker,(k+1)*n_worker
                X = X_samp[ix1:ix2]
                seed_arr = seed_base+np.arange(n_worker)
                seed_flat_arr = np.append(seed_flat_arr,seed_arr) # update seed_flat_arr for sanity check
                seed_base += n_worker
                x_list_list = [[xs] for xs in X]
                seed_list_list = [[sd] for sd in seed_arr]
                out_list_list = [[os.path.join(outdir, RESULT_SAMP_FILE_TEMP % (prob_name,k+1,n+1))] \
                                  for n in range(n_worker)]
                t1, t1_c = default_timer(), clock()
                # evaluate function
                Y_list = func_eval(f_list_list,x_list_list,seed_list_list,out_list_list,pool_eval)
                t2, t2_c = default_timer(), clock()
                t_eval[k], t_eval_cpu[k] = t2-t1, t2_c-t1_c
                # unpack results
                Y = []
                for yl in Y_list:
                    Y += yl
                # get best value and best location for each iteration
                min_ix = np.argmin(Y)
                best_loc_it[k],best_val_it[k] = X[min_ix],Y[min_ix]
                best_seed_it[k] = seed_arr[min_ix]
                # save Y
                Y_samp[ix1:ix2] = Y
                # save variables
                fidelity_status_arr[k] = multi_fidelity
                
                if debug and dim in [1,2] and visual_debug:
                    intr_iter += 1
                    nx = 100
                    plt.figure(0)
                    plt.clf()
                    if dim == 2:
                        try:
                            # get true function
                            func_obj = prob.func_obj_list[0] # function object
                            true_func = func_obj.hf
                            x1_arr = np.linspace(func_bd[0][0],func_bd[0][1],nx)
                            x2_arr = np.linspace(func_bd[1][0],func_bd[1][1],nx)
                            X1, X2 = np.meshgrid(x1_arr, x2_arr)
                            pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                            true_val = true_func(pts)
                            true_val = true_val.ravel()
                            # true function contour with evaluated samples
                            if len(set(true_val)) > 1:
                                plt.contour(X1,X2,true_val.reshape((nx,nx)),20) 
                                plt.colorbar()
                        except:
                            pass
                        plt.plot(X_samp[:ix2,0],X_samp[:ix2,1],'ko')
                        plt.xlabel(prob.x_var[0], fontsize=label_fs)
                        plt.ylabel(prob.x_var[1], fontsize=label_fs)
                        plt.xlim(func_bd[0])
                        plt.ylim(func_bd[1])    
                        
                    elif dim == 1:
                        try:
                            # get true function
                            func_obj = prob.func_obj_list[0] # function object
                            true_func = func_obj.hf
                            pts = np.linspace(func_bd[0][0], func_bd[0][1], nx)
                            true_val = true_func(pts)
                            true_val = true_val.ravel()
                            # true function with evaluated samples
                            plt.plot(pts,true_val,'r-')
                            plt.plot(X_samp[:ix2],true_func(X_samp[:ix2]),'ko')
                            plt.ylabel('true %s' % prob.y_var, fontsize=label_fs)
                        except:
                            # then we plot noisy evaluations with samples
                            plt.plot(X_samp[:ix2],Y_samp[:ix2],'ko')
                            plt.ylabel('noisy %s' % prob.y_var, fontsize=label_fs)   
                        plt.xlabel(prob.x_var[0], fontsize=label_fs)
                        plt.xlim(func_bd[0])
                            
                    ax = plt.gca()
                    plt.text(0.5, 1.03, 'Iteration %d: design of experiments' % intr_iter,
                                 horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                                 fontweight='bold')
                    plt.savefig('%s/%s_iter%d_doe.pdf' % (outdir_debug,prob_name,k+1))
                    plt.close()
                
                # save to netcdf file
                if save_samp:
                    out_file = os.path.join(outdir, RESULT_GENERAL_SAMP_FILE_TEMP % (prob_name,k+1))
                    if os.path.isfile(out_file):
                        os.remove(out_file) # remove file if exists
                    with netcdf.netcdf_file(out_file,'w') as f:
                        f.createDimension('var',dim)
                        f.createDimension('samp',n_worker)
                        nc_multi_fidelity = f.createVariable('multi_fidelity','d',())
                        nc_multi_fidelity.assignValue(multi_fidelity)
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
            
        # zoom level indicator
        n_zoom = 0
        # weight exponent in rbf fitting
        wgt_expon = init_wgt_expon
        # count number of consecutive failures in pure local SRS
        n_fail = 0
        # count number of consecutive fit bound expansion
        n_expand = 0
        # count number of times of reducing step size in pure local SRS
        n_reduce_step_size = 0
        # global SRS percent (full, without rounding to 0.1 resolution)
        gSRS_pct_full = init_gSRS_pct
        # running fit bound
        fit_bd = func_bd
        # fit bound for each zoom level
        fit_bd_save = {n_zoom: fit_bd}
        # count number of full restart
        n_full_restart = 0
        # full restart flag
        full_restart = False
        # step size factor
        step_size_fact = init_step_size_fact
        # count number of zoom-out per level
        n_zoom_out = {n_zoom: 0}
        # save x_star for most recent consecutive iterations
        x_star_list = [None, None]
        if multi_fidelity:
            
            X_samp_out,Y_samp_out = None,None
            # points that are out of box
            X_samp_lf_out = np.zeros((0,dim))
            Y_samp_lf_out = np.zeros(0)
            X_samp_hf_out = np.zeros((0,dim))
            Y_samp_hf_out = np.zeros(0)
            # variables for low-fidelity accuracy test
            fail_lf_acc = [None]*max_C_lf_fail
            success_lf_acc = [None]*max_C_lf_success
            
            if doe_check_lf_acc:
                # check accuracy of low fidelity model
                fail_lf_acc,success_lf_acc = check_lf_acc(X_samp_lf,Y_samp_lf,X_samp_hf,Y_samp_hf,hf_prop_arr[:init_iter],
                                                          fail_lf_acc,success_lf_acc)
                if verbose:
                    print('fail_lf_acc:')
                    print(fail_lf_acc)
                    print('success_lf_acc:')
                    print(success_lf_acc)
        else:
            
            X_samp_lf_out,Y_samp_lf_out,X_samp_hf_out,Y_samp_hf_out,fail_lf_acc,success_lf_acc = [None]*6
            # points that are out of box
            X_samp_out = np.zeros((0,dim))
            Y_samp_out = np.zeros(0)
    
    else:
        t1 = default_timer()
        
        resume_opt_iter = int(resume_opt_iter) # convert to integer type if not
        # remove lock file if exists
        result_npz_lock_file = result_npz_file+'.lock'
        if os.path.isfile(result_npz_lock_file):
            os.remove(result_npz_lock_file)
        # read experiment conditions from previous trials
        data = np.load(result_npz_file)
        assert(resume_opt_iter==data['opt_iter']),'Please resume from where it ended last time.'
        # sanity check for consistency of experiment conditions
        assert(init_iter==data['init_iter'] and n_proc==data['n_proc'] and seed==data['seed']
               and serial_mode==data['serial_mode'] and np.all(func_bd==data['func_bd']) 
               and n_cand_fact == data['n_cand_fact'] and doe_check_lf_acc==data['doe_check_lf_acc']
               and normalize_data==data['normalize_data'] and init_wgt_expon==data['init_wgt_expon']
               and wgt_expon_delta==data['wgt_expon_delta'] and max_reduce_step_size==data['max_reduce_step_size']
               and box_fact==data['box_fact'] and init_step_size_fact==data['init_step_size_fact']
               and init_gSRS_pct==data['init_gSRS_pct'] and alpha_amp==data['alpha_amp'] 
               and np.all(sm_range==data['sm_range']) and log_opt==data['log_opt']
               and n_fold==data['n_fold'] and resol==data['resol'] and min_zoom_in==data['min_zoom_in']
               and max_zoom_out==data['max_zoom_out'] and min_lf_test_samp==data['min_lf_test_samp']
               and x_star_std_factor==data['x_star_std_factor'] and max_C_expand_fit==data['max_C_expand_fit']
               and use_eff_n_samp==data['use_eff_n_samp'] and rbf_kernel==data['rbf_kernel']
               and soft_func_bd==data['soft_func_bd'] and max_C_lf_fail==data['max_C_lf_fail']
               and lf_test_alpha==data['lf_test_alpha'] and max_C_lf_success==data['max_C_lf_success']
               and max_C_fail==data['max_C_fail'] and dynam_box_fact==data['dynam_box_fact']
               and round_gSRS_pct_full==data['round_gSRS_pct_full'] and alpha_crit_err==data['alpha_crit_err']
               and alpha_base==data['alpha_base'])
        # read optimization results and status variables from previous
        intr_iter = data['intr_iter'].item(0)
        full_restart = data['full_restart'].item(0)
        multi_fidelity = data['multi_fidelity'].item(0)
        X_samp_lf = data['X_samp_lf']
        Y_samp_lf = data['Y_samp_lf']
        wgt_expon = data['wgt_expon'].item(0)
        X_samp_hf = data['X_samp_hf']
        Y_samp_hf = data['Y_samp_hf']
        X_samp = data['X_samp']
        Y_samp = data['Y_samp']
        opt_sm_arr[:resume_opt_iter] = data['opt_sm_arr']
        cv_err_arr[:resume_opt_iter] = data['cv_err_arr']
        t_build_mod[:resume_opt_iter] = data['t_build_mod']
        t_build_mod_cpu[:resume_opt_iter] = data['t_build_mod_cpu']
        x_star_list = list(data['x_star_list'])
        gSRS_pct_full = data['gSRS_pct_full'].item(0)
        fit_bd = [(bd[0],bd[1]) for bd in data['fit_bd']]
        assert(np.all(fit_bd==data['fit_bd'])) # sanity check
        n_proc_hf = data['n_proc_hf'].item(0)
        n_worker_hf = data['n_worker_hf'].item(0)
        n_proc_lf = data['n_proc_lf'].item(0)
        n_worker_per_proc_lf = data['n_worker_per_proc_lf'].item(0)
        step_size_fact = data['step_size_fact'].item(0)
        n_fail = data['n_fail'].item(0)
        n_expand = data['n_expand'].item(0)
        n_reduce_step_size = data['n_reduce_step_size'].item(0)
        n_zoom = data['n_zoom'].item(0)
        n_zoom_out = data['n_zoom_out'].item(0)
        X_samp_lf_out = data['X_samp_lf_out']
        Y_samp_lf_out = data['Y_samp_lf_out']
        expand_status_arr[:resume_opt_iter] = data['expand_status_arr']
        expand_mag_arr[:resume_opt_iter] = data['expand_mag_arr']
        n_worker_lf = data['n_worker_lf'].item(0)
        X_samp_hf_out = data['X_samp_hf_out']
        Y_samp_hf_out = data['Y_samp_hf_out']
        X_samp_out = data['X_samp_out']
        Y_samp_out = data['Y_samp_out']
        n_worker = data['n_worker'].item(0)
        t_prop_pt[:resume_opt_iter] = data['t_prop_pt']
        t_prop_pt_cpu[:resume_opt_iter] = data['t_prop_pt_cpu']
        n_full_restart = data['n_full_restart'].item(0)
        gSRS_pct_arr[:resume_opt_iter] = data['gSRS_pct_arr']
        n_zoom_arr[:resume_opt_iter] = data['n_zoom_arr']
        n_expand_arr[:resume_opt_iter] = data['n_expand_arr']
        fidelity_status_arr[:init_iter+resume_opt_iter] = data['fidelity_status_arr']
        hf_prop_arr[:init_iter+resume_opt_iter] = data['hf_prop_arr']
        seed_base = data['seed_base'].item(0)
        seed_flat_arr = data['seed_flat_arr']
        t_eval_prop[:resume_opt_iter] = data['t_eval_prop']
        t_eval_prop_cpu[:resume_opt_iter] = data['t_eval_prop_cpu']
        best_loc_it_lf[:init_iter+resume_opt_iter] = data['best_loc_it_lf']
        best_val_it_lf[:init_iter+resume_opt_iter] = data['best_val_it_lf']
        best_seed_it_lf[:init_iter+resume_opt_iter] = data['best_seed_it_lf']
        best_loc_it_hf[:init_iter+resume_opt_iter] = data['best_loc_it_hf']
        best_val_it_hf[:init_iter+resume_opt_iter] = data['best_val_it_hf']
        best_seed_it_hf[:init_iter+resume_opt_iter] = data['best_seed_it_hf']
        best_loc_it[:init_iter+resume_opt_iter] = data['best_loc_it']
        best_val_it[:init_iter+resume_opt_iter] = data['best_val_it']
        best_seed_it[:init_iter+resume_opt_iter] = data['best_seed_it']
        fail_lf_acc = None if data['fail_lf_acc'] == np.array(None) else list(data['fail_lf_acc'])
        success_lf_acc = None if data['success_lf_acc'] == np.array(None) else list(data['success_lf_acc'])
        high_opt_arr[:resume_opt_iter] = data['high_opt_arr']
        fit_bd_save = data['fit_bd_save'].item(0)
        t_opt[:resume_opt_iter] = data['t_opt']
        t_opt_cpu[:resume_opt_iter] = data['t_opt_cpu']
        alpha = data['alpha'].item(0)
        x_star_err = data['x_star_err'].item(0)
        gSRS_pct = data['gSRS_pct'].item(0)
        
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
    
    start_iter = 0 if resume_opt_iter is None else resume_opt_iter
    
    for k in range(start_iter,opt_iter):
        
        if debug and dim in [1,2] and visual_debug:
            seq = 0 # initialization for sequence counter
            intr_iter += 1 # update counter
        else:
            seq, intr_iter = None, None
        
        if verbose:
            print('\nStart optimization iteration = %d/%d' % (k+1,opt_iter))
        
        tt1 = default_timer()
        tt1_c = clock()
        
        if not full_restart:
            
            ########### Fit model ##############
            
            t1 = default_timer()
            t1_c = clock()

            if multi_fidelity:
                # low fidelity model
                rbf_mod_lf,opt_sm,cv_err,_ = RBF_reg(X_samp_lf,Y_samp_lf,n_proc,sm_range,normalize_data=normalize_data,\
                                                     kernel=rbf_kernel,wgt_expon=wgt_expon,log_opt=log_opt,\
                                                     n_fold=n_fold,pool=pool_rbf)
                # high fidelity model              
                rbf_mod_hf,_,_,_ = RBF_reg(X_samp_hf,Y_samp_hf,n_proc,sm_range,normalize_data=normalize_data,\
                                           kernel=rbf_kernel,wgt_expon=wgt_expon,log_opt=log_opt,\
                                           n_fold=n_fold,pool=pool_rbf)
            else:
                rbf_mod,opt_sm,cv_err,_ = RBF_reg(X_samp,Y_samp,n_proc,sm_range,normalize_data=normalize_data,\
                                                  kernel=rbf_kernel,wgt_expon=wgt_expon,log_opt=log_opt,\
                                                  n_fold=n_fold,pool=pool_rbf)
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
            
            # find x_star
            if multi_fidelity:
                if mf_x_star_by_fit:
                    Y_fit = rbf_mod_hf(X_samp_hf)
                    min_ix = np.argmin(Y_fit)
                else:
                    min_ix = np.argmin(Y_samp_hf)
                x_star = X_samp_hf[min_ix]
            else:
                Y_fit = rbf_mod(X_samp)
                min_ix = np.argmin(Y_fit)
                x_star = X_samp[min_ix]
            # update x_star_list
            x_star_list[1] = x_star_list[0]
            x_star_list[0] = X_samp_lf[np.argmin(rbf_mod_lf(X_samp_lf))] if multi_fidelity else x_star
            # find gSRS_pct
            if round_gSRS_pct_full:
                if gSRS_pct_full >= 0.1:
                    # compute alpha
                    alpha, x_star_err = get_alpha(x_star_list,fit_bd)
                    # compute gSRS_pct_full
                    if use_eff_n_samp:
                        eff_n = eff_n_samp(X_samp_lf,fit_bd) if multi_fidelity else eff_n_samp(X_samp,fit_bd)
                        beta_base = eff_n**(-1./dim)
                    else:
                        n_samp = len(X_samp_lf) if multi_fidelity else len(X_samp)
                        beta_base = n_samp**(-1./dim)
                    gSRS_pct_full = min(1.,gSRS_pct_full*beta_base**alpha)
                gSRS_pct = np.floor(10*gSRS_pct_full)/10.0
            else:
                # compute alpha
                alpha, x_star_err = get_alpha(x_star_list,fit_bd)
                # compute gSRS_pct_full
                if use_eff_n_samp:
                    eff_n = eff_n_samp(X_samp_lf,fit_bd) if multi_fidelity else eff_n_samp(X_samp,fit_bd)
                    beta_base = eff_n**(-1./dim)
                else:
                    n_samp = len(X_samp_lf) if multi_fidelity else len(X_samp)
                    beta_base = n_samp**(-1./dim)
                gSRS_pct_full = min(1.,gSRS_pct_full*beta_base**alpha)
                gSRS_pct = gSRS_pct_full
            assert(0<=gSRS_pct<=1)
            
            if verbose:
                if multi_fidelity:
                    assert(n_proc_hf == n_worker_hf)
                    print('n_proc_hf = %d' % n_proc_hf)
                    print('n_proc_lf = %d' % n_proc_lf)
                    print('n_worker_per_proc_lf = %d' % n_worker_per_proc_lf)
                print('step_size_fact = %g' % step_size_fact)
                print('wgt_expon = %g' % wgt_expon)
                print('opt_sm = %.1e' % opt_sm)
                print('x_star_err = %g' % x_star_err)
                print('alpha = %g' % alpha)
                print('gSRS_pct = %g' % gSRS_pct)
                print('n_fail = %d' % n_fail)
                print('n_expand = %d' % n_expand)
                print('n_reduce_step_size = %d' % n_reduce_step_size)
                print('n_zoom = %d' % n_zoom)
                print('x_star =')
                print(x_star)
                print('n_zoom_out =')
                print(list(n_zoom_out.values()))
                print('fit_bd =')
                print(fit_bd)
                
            assert(np.all(np.array(list(n_zoom_out.values()))<=max_zoom_out)) # list is needed for Python3
            assert(np.all([bd[0]<=x_star[j]<=bd[1] for j,bd in enumerate(fit_bd)]))
            
            if debug and dim in [1,2] and visual_debug:
                seq += 1
                nx = 100
                plt.figure(0)
                plt.clf()
                if dim == 2:
                    x1_arr = np.linspace(fit_bd[0][0],fit_bd[0][1],nx)
                    x2_arr = np.linspace(fit_bd[1][0],fit_bd[1][1],nx)
                    X1, X2 = np.meshgrid(x1_arr, x2_arr)
                    pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                    # fit function contour with evaluated samples
                    fit_val = rbf_mod_lf(pts) if multi_fidelity else rbf_mod(pts)
                    if len(set(fit_val)) > 1:
                        plt.contour(X1,X2,fit_val.reshape((nx,nx)),20)
                        plt.colorbar()
                    if multi_fidelity:
                        plt.plot(X_samp_lf[:,0],X_samp_lf[:,1],'o',color='gray')
                        plt.plot(X_samp_hf[:,0],X_samp_hf[:,1],'ko')
                    else:
                        plt.plot(X_samp[:,0],X_samp[:,1],'ko')
                    plt.xlabel(prob.x_var[0], fontsize=label_fs)
                    plt.ylabel(prob.x_var[1], fontsize=label_fs)
                
                elif dim == 1:
                    pts = np.linspace(fit_bd[0][0], fit_bd[0][1], nx).reshape((-1,1))
                    fit_val = rbf_mod_lf(pts) if multi_fidelity else rbf_mod(pts)
                    plt.plot(pts,fit_val,'b-')
                    if multi_fidelity:
                        plt.plot(X_samp_lf,rbf_mod_lf(X_samp_lf),'o',color='gray')
                        plt.plot(X_samp_hf,rbf_mod_lf(X_samp_hf),'ko')
                    else:    
                        plt.plot(X_samp,rbf_mod(X_samp),'ko')
                    plt.xlabel(prob.x_var[0], fontsize=label_fs)
                    plt.ylabel(prob.y_var, fontsize=label_fs)
                    
                ax = plt.gca()
                plt.text(0.5, 1.03, 'Iteration %d: fit with RBF\n' % intr_iter,
                         horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                         fontweight='bold')
                plt.title('gSRS_p = %g, reduce = %d, expand = %d, zoom = %d' \
                          % (gSRS_pct, n_reduce_step_size, n_expand, n_zoom), 
                          fontsize=title_fs, y = 1.01)
                plt.savefig('%s/%s_iter%d_seq%d_rbf_fit.pdf' % (outdir_debug,prob_name,k+1+init_iter,seq))
                plt.close()
            
            if multi_fidelity:
                
                # propose points for low fidelity
                fit_bd_old = fit_bd # save fit bound
                status_var_list = [fit_bd,X_samp_lf,Y_samp_lf,n_expand,X_samp_lf_out,Y_samp_lf_out]
                prop_pt_arr_lf,expand_status,expand_mag,status_var_list,seq = \
                        propose(gSRS_pct,x_star,rbf_mod_lf,step_size_fact,n_zoom,n_worker_lf,
                                status_var_list,seq,intr_iter,k+1+init_iter,X_samp_hf=X_samp_hf)
                        
                # update variables
                expand_status_arr[k], expand_mag_arr[k] = expand_status, expand_mag
                fit_bd,X_samp_lf,Y_samp_lf,n_expand,X_samp_lf_out,Y_samp_lf_out = status_var_list
        
                if fit_bd_old != fit_bd:
                    # update in-box and out-of-box samples for high fidelity
                    tot_X_samp = np.vstack((X_samp_hf,X_samp_hf_out))
                    tot_Y_samp = np.append(Y_samp_hf,Y_samp_hf_out)
                    in_box_ix, out_box_ix = get_box_samp(tot_X_samp,fit_bd)
                    X_samp_hf, X_samp_hf_out = tot_X_samp[in_box_ix], tot_X_samp[out_box_ix]
                    Y_samp_hf, Y_samp_hf_out = tot_Y_samp[in_box_ix], tot_Y_samp[out_box_ix]
                # propose points for high fidelity
                prop_pt_arr_hf = SRS(rbf_mod_lf,prop_pt_arr_lf,X_samp_hf,n_worker_hf,wgt_pat_bd)
                    
            else:
                
                status_var_list = [fit_bd,X_samp,Y_samp,n_expand,X_samp_out,Y_samp_out]
                prop_pt_arr,expand_status,expand_mag,status_var_list,seq = \
                        propose(gSRS_pct,x_star,rbf_mod,step_size_fact,n_zoom,n_worker,
                                status_var_list,seq,intr_iter,k+1+init_iter)
                # update variables
                expand_status_arr[k], expand_mag_arr[k] = expand_status, expand_mag
                fit_bd,X_samp,Y_samp,n_expand,X_samp_out,Y_samp_out = status_var_list
                
            t2 = default_timer()
            t2_c = clock()
            
            t_prop_pt[k] = t2-t1
            t_prop_pt_cpu[k] = t2_c-t1_c
            
            if verbose:
                print('time to propose points = %.2e sec' % t_prop_pt[k])
    
        else:
            
            # i.e. do full restart
            if n_full_restart == 0:
                if multi_fidelity:
                    n_init_samp_lf = n_worker_lf*init_iter
                    n_init_samp_hf = n_worker_hf*init_iter
                    X_samp_restart_lf = doe(n_init_samp_lf,func_bd) 
                    X_samp_restart_hf = sub_doe(X_samp_restart_lf,n_init_samp_hf)    
                else:
                    n_init_samp = n_worker*init_iter
                    X_samp_restart = doe(n_init_samp,func_bd)
                    
            if multi_fidelity:       
                ix1_lf, ix2_lf = n_full_restart*n_worker_lf, (n_full_restart+1)*n_worker_lf
                ix1_hf, ix2_hf = n_full_restart*n_worker_hf, (n_full_restart+1)*n_worker_hf
                prop_pt_arr_lf = X_samp_restart_lf[ix1_lf:ix2_lf]
                prop_pt_arr_hf = X_samp_restart_hf[ix1_hf:ix2_hf]
            else:
                ix1, ix2 = n_full_restart*n_worker, (n_full_restart+1)*n_worker
                prop_pt_arr = X_samp_restart[ix1:ix2]
            
            
            if debug and dim in [1,2] and visual_debug:
                seq += 1
                nx = 100
                plt.figure(0)
                plt.clf()
                if dim == 2:
                    try:
                        # get true function
                        func_obj = prob.func_obj_list[0] # function object
                        true_func = func_obj.hf
                        x1_arr = np.linspace(func_bd[0][0],func_bd[0][1],nx)
                        x2_arr = np.linspace(func_bd[1][0],func_bd[1][1],nx)
                        X1, X2 = np.meshgrid(x1_arr, x2_arr)
                        pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                        true_val = true_func(pts)
                        true_val = true_val.ravel()
                        # true function contour with evaluated samples
                        if len(set(true_val)) > 1:
                            plt.contour(X1,X2,true_val.reshape((nx,nx)),20)
                            plt.colorbar()
                    except:
                        pass
                    if multi_fidelity:
                        plt.plot(X_samp_restart_lf[:ix2_lf,0],X_samp_restart_lf[:ix2_lf,1],'o',color='gray')
                        plt.plot(X_samp_restart_hf[:ix2_hf,0],X_samp_restart_hf[:ix2_hf,1],'ko')
                    else:
                        plt.plot(X_samp_restart[:ix2,0],X_samp_restart[:ix2,1],'ko')
                    plt.xlabel(prob.x_var[0], fontsize=label_fs)
                    plt.ylabel(prob.x_var[1], fontsize=label_fs)
                    plt.xlim(func_bd[0])
                    plt.ylim(func_bd[1])
                    
                elif dim == 1:
                    try:
                        # get true function
                        func_obj = prob.func_obj_list[0] # function object
                        true_func_hf = func_obj.hf
                        pts = np.linspace(func_bd[0][0], func_bd[0][1], nx)
                        # true function with evaluated samples
                        plt.plot(pts,true_func_hf(pts).ravel(),'r-')
                        if multi_fidelity:
                            func_obj = prob.func_obj_list[-1] # function object
                            true_func_lf = func_obj.lf
                            plt.plot(pts,true_func_lf(pts).ravel(),'b-')
                            plt.plot(X_samp_restart_lf[:ix2_lf],true_func_lf(X_samp_restart_lf[:ix2_lf]),'o',color='gray')
                            plt.plot(X_samp_restart_hf[:ix2_hf],true_func_lf(X_samp_restart_hf[:ix2_hf]),'ko')
                        else:
                            plt.plot(X_samp_restart[:ix2],true_func_hf(X_samp_restart[:ix2]),'ko')
                        plt.xlabel(prob.x_var[0], fontsize=label_fs)
                        plt.ylabel('true %s' % prob.y_var, fontsize=label_fs)
                    except:
                        pass
                        
                ax = plt.gca()
                plt.text(0.5, 1.03, 'Iteration %d: design of experiments' % intr_iter,
                         horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                         fontweight='bold')
                plt.savefig('%s/%s_iter%d_seq%d_doe.pdf' % (outdir_debug,prob_name,k+1+init_iter,seq))
                plt.close()
        
        tm1 = default_timer()        
        # save variables
        gSRS_pct_arr[k] = gSRS_pct # save gSRS_pct
        n_zoom_arr[k] = n_zoom # save n_zoom
        n_expand_arr[k] = n_expand # save n_expand
        fidelity_status_arr[k+init_iter] = multi_fidelity # save multifidelity status
        if multi_fidelity:
            hf_prop_arr[k+init_iter] = n_proc_hf/float(n_proc_hf+n_proc_lf) # save proportion of hf
            assert(hf_prop_arr[k+init_iter]>=0.5),hf_prop_arr[k+init_iter]
            if verbose: 
                print('hf_prop = %g' % hf_prop_arr[k+init_iter])
        
        
        ############ Evaluate proposed points #############
        
        if multi_fidelity:
            
            # generate list of objective function list
            f_list_list = [[obj_func_lf]*n_worker_per_proc_lf]*n_proc_lf+[[obj_func_hf]]*n_proc_hf
            # initialization
            x_list_list = []
            seed_list_list = []
            out_list_list = []
            for p in range(n_proc_lf):
                ix1_plf, ix2_plf = p*n_worker_per_proc_lf,(p+1)*n_worker_per_proc_lf
                x_list_list.append(prop_pt_arr_lf[ix1_plf:ix2_plf])
                seed_arr_lf = seed_base+np.arange(ix1_plf,ix2_plf)
                seed_list_list.append(seed_arr_lf)
                out_list = [os.path.join(outdir, RESULT_LF_SAMP_FILE_TEMP % (prob_name,k+init_iter+1,n+1)) \
                            for n in range(ix1_plf,ix2_plf)]
                out_list_list.append(out_list)
            seed_base += n_worker_lf
            for p in range(n_proc_hf):
                x_list_list.append(prop_pt_arr_hf[p:p+1])
                seed_arr_hf = seed_base+np.arange(p,p+1)
                seed_list_list.append(seed_arr_hf)
                out_list = [os.path.join(outdir, RESULT_HF_SAMP_FILE_TEMP % (prob_name,k+init_iter+1,p+1))]
                out_list_list.append(out_list)
            seed_base += n_worker_hf
            # update seed_flat_arr for sanity check
            seed_flat = seed_list_list[0]
            for seed_arr in seed_list_list[1:]:
                seed_flat = np.append(seed_flat,seed_arr)
            seed_flat_arr = np.append(seed_flat_arr,seed_flat)
            # get seeds of each X for low-fidelity and high-fidelity
            seed_X_lf = np.array(seed_list_list[:n_proc_lf]).reshape((1,-1)).flatten()
            seed_X_hf = np.array(seed_list_list[n_proc_lf:]).flatten()
            assert(seed_X_hf.shape == (n_worker_hf,) and seed_X_lf.shape == (n_worker_lf,))
            # evaluate function
            t1, t1_c = default_timer(), clock()
            Y_list = func_eval(f_list_list,x_list_list,seed_list_list,out_list_list,pool_eval)
            t2, t2_c = default_timer(), clock()
            # unpack results
            Y_list_lf, Y_list_hf = Y_list[:n_proc_lf], Y_list[n_proc_lf:]
            Y_prop_pt_lf = []
            for yl in Y_list_lf:
                Y_prop_pt_lf += yl
            Y_prop_pt_hf = []
            for yl in Y_list_hf:
                Y_prop_pt_hf += yl
            assert(len(prop_pt_arr_lf)==len(Y_prop_pt_lf) and len(prop_pt_arr_hf)==len(Y_prop_pt_hf))
            Y_prop_pt_lf = np.array(Y_prop_pt_lf)
            Y_prop_pt_hf = np.array(Y_prop_pt_hf)
            
        else:
            
            # generate list of objective function list
            f_list_list = [[obj_func]]*n_worker
            seed_arr = seed_base+np.arange(n_worker)
            seed_flat_arr = np.append(seed_flat_arr,seed_arr) # update seed_flat_arr for sanity check
            seed_base += n_worker
            x_list_list = [[xs] for xs in prop_pt_arr]
            seed_list_list = [[sd] for sd in seed_arr]
            out_list_list = [[os.path.join(outdir, RESULT_SAMP_FILE_TEMP % (prob_name,k+init_iter+1,n+1))] \
                              for n in range(n_worker)]
            t1, t1_c = default_timer(), clock()
            Y_list = func_eval(f_list_list,x_list_list,seed_list_list,out_list_list,pool_eval)
            t2, t2_c = default_timer(), clock()
            # unpack results
            Y_prop_pt = []
            for yl in Y_list:
                Y_prop_pt += yl
            Y_prop_pt = np.array(Y_prop_pt)
        
        assert(np.all(seed_flat_arr == np.arange(np.min(seed_flat_arr),np.max(seed_flat_arr)+1)))
        
        t_eval_prop[k] = t2-t1
        t_eval_prop_cpu[k] = t2_c-t1_c
        
        if verbose:
            print('time to evaluate points = %.2e sec' % t_eval_prop[k])
        
        if debug and dim in [1,2] and visual_debug and not full_restart:
            seq += 1
            nx = 100
            plt.figure(0)
            plt.clf()
            if dim == 2:
                x1_arr = np.linspace(fit_bd[0][0],fit_bd[0][1],nx)
                x2_arr = np.linspace(fit_bd[1][0],fit_bd[1][1],nx)
                X1, X2 = np.meshgrid(x1_arr, x2_arr)
                pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                # fit function contour with evaluated samples
                fit_val = rbf_mod_lf(pts) if multi_fidelity else rbf_mod(pts)
                if len(set(fit_val)) > 1:
                    plt.contour(X1,X2,fit_val.reshape((nx,nx)),20)
                    plt.colorbar()
                if multi_fidelity:
                    plt.plot(X_samp_lf[:,0],X_samp_lf[:,1],'o',color='gray')
                    plt.plot(X_samp_hf[:,0],X_samp_hf[:,1],'ko')
                    plt.plot(prop_pt_arr_lf[:,0],prop_pt_arr_lf[:,1],'o',color='violet')
                    plt.plot(prop_pt_arr_hf[:,0],prop_pt_arr_hf[:,1],'mo')
                else:
                    plt.plot(X_samp[:,0],X_samp[:,1],'ko')
                    plt.plot(prop_pt_arr[:,0],prop_pt_arr[:,1],'mo')
                plt.xlabel(prob.x_var[0], fontsize=label_fs)
                plt.ylabel(prob.x_var[1], fontsize=label_fs)
            
            elif dim == 1:
                pts = np.linspace(fit_bd[0][0], fit_bd[0][1], nx).reshape((-1,1))
                fit_val = rbf_mod_lf(pts) if multi_fidelity else rbf_mod(pts)
                plt.plot(pts,fit_val,'b-')
                if multi_fidelity:
                    plt.plot(X_samp_lf,rbf_mod_lf(X_samp_lf),'o',color='gray')
                    plt.plot(X_samp_hf,rbf_mod_lf(X_samp_hf),'ko')
                    plt.plot(prop_pt_arr_lf,rbf_mod_lf(prop_pt_arr_lf),'o',color='violet')
                    plt.plot(prop_pt_arr_hf,rbf_mod_lf(prop_pt_arr_hf),'mo')
                else:    
                    plt.plot(X_samp,rbf_mod(X_samp),'ko')
                    plt.plot(prop_pt_arr,rbf_mod(prop_pt_arr),'mo')
                plt.xlabel(prob.x_var[0], fontsize=label_fs)
                plt.ylabel(prob.y_var, fontsize=label_fs)
            
            if multi_fidelity:
                plt.title('gSRS_p = %g, reduce = %d, expand = %d, zoom = %d, n_lf = %d, n_hf = %d' \
                          % (gSRS_pct, n_reduce_step_size, n_expand, n_zoom, len(prop_pt_arr_lf), len(prop_pt_arr_hf)), 
                          fontsize=title_fs, y = 1.01)
            else:
                plt.title('gSRS_p = %g, reduce = %d, expand = %d, zoom = %d' \
                          % (gSRS_pct, n_reduce_step_size, n_expand, n_zoom), 
                          fontsize=title_fs, y = 1.01)
            
            ax = plt.gca()
            plt.text(0.5, 1.03, 'Iteration %d: propose points\n' % intr_iter,
                     horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                     fontweight='bold')
            plt.savefig('%s/%s_iter%d_seq%d_prop_pts.pdf' % (outdir_debug,prob_name,k+1+init_iter,seq))
            plt.close()
                
        # add evaluated points to existing data
        if multi_fidelity:
            best_Y_prev_hf = np.min(Y_samp_hf) if len(Y_samp_hf) > 0 else None
            X_samp_lf = np.vstack((X_samp_lf,prop_pt_arr_lf))
            Y_samp_lf = np.append(Y_samp_lf,Y_prop_pt_lf)
            X_samp_hf = np.vstack((X_samp_hf,prop_pt_arr_hf))
            Y_samp_hf = np.append(Y_samp_hf,Y_prop_pt_hf)    
        else:
            best_Y_prev = np.min(Y_samp) if len(Y_samp) > 0 else None
            X_samp = np.vstack((X_samp,prop_pt_arr))
            Y_samp = np.append(Y_samp,Y_prop_pt)
        
        # save to netcdf file
        t1, t1_c = default_timer(), clock()
        if save_samp:
            out_file = os.path.join(outdir, RESULT_GENERAL_SAMP_FILE_TEMP % (prob_name,k+init_iter+1))
            if os.path.isfile(out_file):
                os.remove(out_file) # remove file if exists 
            if multi_fidelity:
                with netcdf.netcdf_file(out_file,'w') as f:
                    f.createDimension('var',dim)
                    f.createDimension('samp_lf',n_worker_lf)
                    f.createDimension('samp_hf',n_worker_hf)
                    nc_multi_fidelity = f.createVariable('multi_fidelity','d',())
                    nc_multi_fidelity.assignValue(multi_fidelity)
                    nc_samp = f.createVariable('x_lf','d',('samp_lf','var'))
                    nc_samp[:] = prop_pt_arr_lf
                    nc_samp = f.createVariable('y_lf','d',('samp_lf',))
                    nc_samp[:] = Y_prop_pt_lf
                    nc_samp = f.createVariable('x_hf','d',('samp_hf','var'))
                    nc_samp[:] = prop_pt_arr_hf
                    nc_samp = f.createVariable('y_hf','d',('samp_hf',))
                    nc_samp[:] = Y_prop_pt_hf
                    nc_seed = f.createVariable('seed_hf','i',('samp_hf',))
                    nc_seed[:] = seed_X_hf
                    nc_seed = f.createVariable('seed_lf','i',('samp_lf',))
                    nc_seed[:] = seed_X_lf
                    nc_wall_time = f.createVariable('tw_eval','d',())
                    nc_wall_time.assignValue(t_eval_prop[k])
                    nc_cpu_time = f.createVariable('tc_eval','d',())
                    nc_cpu_time.assignValue(t_eval_prop_cpu[k])
            else:
                with netcdf.netcdf_file(out_file,'w') as f:
                    f.createDimension('var',dim)
                    f.createDimension('samp',n_worker)
                    nc_multi_fidelity = f.createVariable('multi_fidelity','d',())
                    nc_multi_fidelity.assignValue(multi_fidelity)
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
        if multi_fidelity:
            min_ix = np.argmin(Y_prop_pt_lf)
            best_loc_it_lf[k+init_iter],best_val_it_lf[k+init_iter] = prop_pt_arr_lf[min_ix],Y_prop_pt_lf[min_ix]
            best_seed_it_lf[k+init_iter] = seed_X_lf[min_ix]
            min_ix = np.argmin(Y_prop_pt_hf)
            best_loc_it_hf[k+init_iter],best_val_it_hf[k+init_iter] = prop_pt_arr_hf[min_ix],Y_prop_pt_hf[min_ix]
            best_seed_it_hf[k+init_iter] = seed_X_hf[min_ix]
        else:
            min_ix = np.argmin(Y_prop_pt)
            best_loc_it[k+init_iter],best_val_it[k+init_iter] = prop_pt_arr[min_ix],Y_prop_pt[min_ix]
            best_seed_it[k+init_iter] = seed_arr[min_ix]
        
        ############# Prepare for next iteration #############
        
        tp1, tp1_c = default_timer(), clock()
        
        if multi_fidelity:
            
            # check validity of multi fidelity optimization
            if not full_restart:
                
                # check accuracy of low fidelity model
                tot_X_lf = np.vstack((X_samp_lf,X_samp_lf_out))
                tot_Y_lf = np.append(Y_samp_lf,Y_samp_lf_out)
                tot_X_hf = np.vstack((X_samp_hf,X_samp_hf_out))
                tot_Y_hf = np.append(Y_samp_hf,Y_samp_hf_out)
                fail_lf_acc,success_lf_acc = check_lf_acc(tot_X_lf,tot_Y_lf,tot_X_hf,tot_Y_hf,hf_prop_arr[:k+init_iter+1],
                                                          fail_lf_acc,success_lf_acc)
            
            elif n_full_restart == init_iter-1 and doe_check_lf_acc:
                
                # check accuracy of low fidelity model (this is just like what we do after DOE)
                fail_lf_acc,success_lf_acc = check_lf_acc(X_samp_lf,Y_samp_lf,X_samp_hf,Y_samp_hf,hf_prop_arr[:k+init_iter+1],
                                                          fail_lf_acc,success_lf_acc)
                    
            if verbose:
                print('fail_lf_acc:')
                print(fail_lf_acc)
                print('success_lf_acc:')
                print(success_lf_acc)
            
            if debug:
                assert(high_opt_debug[k] in [True,False]),'please ensure experiment condition is the same as previous one'
                high_opt_arr[k] = high_opt_debug[k]
            else:
                # find evaluation time and optimization time
                eval_t = t_eval_prop[k] if t_eval_func is None else t_eval_func
                opt_t = default_timer()-tt1-t_eval_prop[k]
                high_opt_arr[k] = opt_t > eval_t*max_t_opt_ratio
            
            if verbose:
                print('opt_t > %g*eval_t? %d' % (max_t_opt_ratio, high_opt_arr[k]))
            
            # determine whether we continue with multifidelity optimization
            valid_multi_fidelity = False if all(fail_lf_acc) or high_opt_arr[k] else True
            
            # adjust mf utilization
            if valid_multi_fidelity:
                if all(success_lf_acc) and not high_opt_arr[k]:
                    # then double utilization of low fidelity
                    if 2*n_proc_lf <= n_proc-min_n_proc_hf:
                        n_proc_lf = 2*n_proc_lf
                        if verbose:
                            print('Double utilization of low fidelity!')
                    else:
                        n_proc_lf = n_proc-min_n_proc_hf
                    n_proc_hf = n_proc-n_proc_lf
                    n_worker_hf = n_proc_hf
                    n_worker_lf = n_proc_lf*n_worker_per_proc_lf
            
                assert(n_proc_hf >= min_n_proc_hf and n_proc_lf > 0)
                assert(n_worker_lf >= n_worker_hf) # minimum requirement for multifidelity
                    
            # prepare low-fidelity optimization
            fit_bd_old = fit_bd # save fit bound
            n_zoom_old = n_zoom # save n_zoom
            best_Y_new_hf = best_val_it_hf[k+init_iter] # minimum of Y values of newly proposed points
            # find x_star
            if not full_restart:
                if mf_x_star_by_fit:
                    Y_fit = rbf_mod_hf(X_samp_hf)
                    min_ix = np.argmin(Y_fit)
                else:
                    min_ix = np.argmin(Y_samp_hf)
                x_star = X_samp_hf[min_ix]
            else:
                x_star = None
            status_var_list = [gSRS_pct_full,n_expand,n_zoom_out,n_zoom,fit_bd_save,
                               fit_bd,X_samp_lf,Y_samp_lf,X_samp_lf_out,Y_samp_lf_out,wgt_expon,
                               n_reduce_step_size,step_size_fact,n_fail,full_restart,n_full_restart,
                               x_star_list]
            status_var_list = prepare(gSRS_pct,best_Y_new_hf,best_Y_prev_hf,x_star,status_var_list)
            # update variables
            gSRS_pct_full,n_expand,n_zoom_out,n_zoom,fit_bd_save,\
            fit_bd,X_samp_lf,Y_samp_lf,X_samp_lf_out,Y_samp_lf_out,wgt_expon,\
            n_reduce_step_size,step_size_fact,n_fail,full_restart,n_full_restart,\
            x_star_list = status_var_list
            
            if not full_restart:
                
                if fit_bd_old != fit_bd:
                    
                    # update in-box and out-of-box samples for high fidelity
                    tot_X_samp = np.vstack((X_samp_hf,X_samp_hf_out))
                    tot_Y_samp = np.append(Y_samp_hf,Y_samp_hf_out)
                    in_box_ix, out_box_ix = get_box_samp(tot_X_samp,fit_bd)
                    X_samp_hf, X_samp_hf_out = tot_X_samp[in_box_ix], tot_X_samp[out_box_ix]
                    Y_samp_hf, Y_samp_hf_out = tot_Y_samp[in_box_ix], tot_Y_samp[out_box_ix]
                
                    if debug and dim in [1,2] and visual_debug and n_zoom_old != n_zoom:
                        assert(abs(n_zoom_old-n_zoom) == 1)
                        zoom_in = True if n_zoom_old < n_zoom else False
                        zoom_str = 'in' if zoom_in else 'out'
                        seq += 1
                        nx = 100
                        plt.figure(0)
                        plt.clf()
                        
                        if dim == 2:
                            if zoom_in:
                                # new fit bound
                                bd1, bd2 = zip(*fit_bd)
                                plt.plot([bd1[0],bd1[0],bd2[0],bd2[0],bd1[0]],[bd1[1],bd2[1],bd2[1],bd1[1],bd1[1]],'k--')
                                # old fit bound
                                x1_arr = np.linspace(fit_bd_old[0][0],fit_bd_old[0][1],nx)
                                x2_arr = np.linspace(fit_bd_old[1][0],fit_bd_old[1][1],nx)
                                X1, X2 = np.meshgrid(x1_arr, x2_arr)
                                pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                                # fit function contour
                                fit_val = rbf_mod_lf(pts)
                                if len(set(fit_val)) > 1:
                                    plt.contour(X1,X2,fit_val.reshape((nx,nx)),20)
                            else:
                                # old fit bound
                                bd1, bd2 = zip(*fit_bd_old)
                                plt.plot([bd1[0],bd1[0],bd2[0],bd2[0],bd1[0]],[bd1[1],bd2[1],bd2[1],bd1[1],bd1[1]],'k--')
                                # new fit bound
                                x1_arr = np.linspace(fit_bd[0][0],fit_bd[0][1],nx)
                                x2_arr = np.linspace(fit_bd[1][0],fit_bd[1][1],nx)
                                X1, X2 = np.meshgrid(x1_arr, x2_arr)
                                pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                                # fit function contour
                                fit_val = rbf_mod_lf(pts)
                                if len(set(fit_val)) > 1:
                                    plt.contour(X1,X2,fit_val.reshape((nx,nx)),20)
                            
                            # evaluated samples
                            plt.plot(X_samp_lf[:,0],X_samp_lf[:,1],'o',color='gray')
                            plt.plot(X_samp_hf[:,0],X_samp_hf[:,1],'ko')
                            if len(X_samp_lf_out) > 0:
                                plt.plot(X_samp_lf_out[:,0],X_samp_lf_out[:,1],color='white',marker='o',ls='')
                            if len(X_samp_hf_out) > 0:
                                plt.plot(X_samp_hf_out[:,0],X_samp_hf_out[:,1],color='white',marker='o',ls='')
                                
                            # x_star
                            if gSRS_pct < 1 and zoom_in:
                                plt.plot(x_star[0],x_star[1],'rx',mew=1) # mew: marker edge width
                            
                            plt.xlabel(prob.x_var[0], fontsize=label_fs)
                            plt.ylabel(prob.x_var[1], fontsize=label_fs)
                            
                            if zoom_in:
                                plt.xlim(fit_bd_old[0])
                                plt.ylim(fit_bd_old[1])
                            else:
                                plt.xlim(fit_bd[0])
                                plt.ylim(fit_bd[1])
                            
                            if len(set(fit_val)) > 1:
                                plt.colorbar()
                        
                        elif dim == 1:
                            
                            if zoom_in:
                                # new fit bound
                                plt.axvline(x=fit_bd[0][0],color='k',ls='--')
                                plt.axvline(x=fit_bd[0][1],color='k',ls='--')
                                # old fit bound
                                pts = np.linspace(fit_bd_old[0][0], fit_bd_old[0][1], nx).reshape((-1,1))
                                fit_val = rbf_mod_lf(pts)
                                plt.plot(pts,fit_val,'b-')
                                
                            else:
                                # old fit bound
                                plt.axvline(x=fit_bd_old[0][0],color='k',ls='--')
                                plt.axvline(x=fit_bd_old[0][1],color='k',ls='--')
                                # new fit bound
                                pts = np.linspace(fit_bd[0][0], fit_bd[0][1], nx).reshape((-1,1))
                                fit_val = rbf_mod_lf(pts)
                                plt.plot(pts,fit_val,'b-')
                            
                            # evaluated samples
                            plt.plot(X_samp_lf,rbf_mod_lf(X_samp_lf),'o',color='gray')
                            plt.plot(X_samp_hf,rbf_mod_lf(X_samp_hf),'ko')

                            if len(X_samp_lf_out) > 0:
                                plt.plot(X_samp_lf_out,rbf_mod_lf(X_samp_lf_out),color='white',marker='o',ls='')
                            if len(X_samp_hf_out) > 0:
                                plt.plot(X_samp_hf_out,rbf_mod_lf(X_samp_hf_out),color='white',marker='o',ls='')
                                
                            # x_star
                            if gSRS_pct < 1 and zoom_in:
                                plt.plot(x_star,rbf_mod_lf(x_star.reshape((-1,1))),'rx',mew=1) # mew: marker edge width
                            
                            plt.xlabel(prob.x_var[0], fontsize=label_fs)
                            plt.ylabel(prob.y_var, fontsize=label_fs)
                            
                            if zoom_in:
                                plt.xlim(fit_bd_old[0])
                            else:
                                plt.xlim(fit_bd[0])
                                
                        ax = plt.gca()
                        plt.text(0.5, 1.03, 'Iteration %d: zoom %s\n' % (intr_iter,zoom_str),
                                 horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                                 fontweight='bold')
                        plt.title('gSRS_p = %g, reduce = %d, expand = %d, zoom = %d' \
                                  % (gSRS_pct, n_reduce_step_size, n_expand, n_zoom), 
                                  fontsize=title_fs, y = 1.01)
                        plt.savefig('%s/%s_iter%d_seq%d_zoom_%s.pdf' % (outdir_debug,prob_name,k+1+init_iter,seq,zoom_str))
                        plt.close()
                    
                if not valid_multi_fidelity:
                    
                     # then switch to single-fidelity optimization
                    if verbose:
                        print('Switch to single-fidelity optimization!')
                    
                    # update in-box and out-of-box samples
                    X_samp = X_samp_hf.copy()
                    Y_samp = Y_samp_hf.copy()
                    X_samp_out = X_samp_hf_out.copy()
                    Y_samp_out = Y_samp_hf_out.copy()
                    # update variables
                    multi_fidelity = False
                    # set status parameters
                    n_worker = n_proc
                    assert(n_cand_fact*dim >= n_worker), 'number of candidate points needs to be no less than number of workers'
                    obj_func = obj_func_list[0]
                    
                    if debug and dim in [1,2] and visual_debug:
                        seq += 1
                        nx = 100
                        plt.figure(0)
                        plt.clf()
                        
                        if dim == 2:    
                            x1_arr = np.linspace(fit_bd[0][0],fit_bd[0][1],nx)
                            x2_arr = np.linspace(fit_bd[1][0],fit_bd[1][1],nx)
                            X1, X2 = np.meshgrid(x1_arr, x2_arr)
                            pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                            # fit function contour with evaluated samples
                            fit_val = rbf_mod_lf(pts)
                            if len(set(fit_val)) > 1:
                                plt.contour(X1,X2,fit_val.reshape((nx,nx)),20)
                                plt.colorbar()
                            plt.plot(X_samp[:,0],X_samp[:,1],'ko')
                            plt.xlabel(prob.x_var[0], fontsize=label_fs)
                            plt.ylabel(prob.x_var[1], fontsize=label_fs)
                        
                        elif dim == 1:
                            pts = np.linspace(fit_bd[0][0], fit_bd[0][1], nx).reshape((-1,1))
                            fit_val = rbf_mod_lf(pts)
                            plt.plot(pts,fit_val,'b-')
                            plt.plot(X_samp,rbf_mod_lf(X_samp),'ko')
                            plt.xlabel(prob.x_var[0], fontsize=label_fs)
                            plt.ylabel(prob.y_var, fontsize=label_fs)
                                
                        ax = plt.gca()
                        plt.text(0.5, 1.03, 'Iteration %d: switch to single fidelity\n' % intr_iter,
                                 horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                                 fontweight='bold')
                        plt.title('gSRS_p = %g, reduce = %d, expand = %d, zoom = %d' \
                                  % (gSRS_pct, n_reduce_step_size, n_expand, n_zoom), 
                                  fontsize=title_fs, y = 1.01)
                        plt.savefig('%s/%s_iter%d_seq%d_switch_to_sf.pdf' % (outdir_debug,prob_name,k+1+init_iter,seq))
                        plt.close()
                
            elif n_full_restart == 0:
                
                multi_fidelity,status_param_list = program_init()
                assert(multi_fidelity) # sanity check
                # update status parameters
                n_proc_hf,n_proc_lf,n_worker_hf,n_worker_per_proc_lf,n_worker_lf,\
                obj_func_lf,obj_func_hf = status_param_list
                # update in-box and out-of-box samples for high fidelity
                X_samp_hf = np.zeros((0,dim))
                Y_samp_hf = np.zeros(0)
                X_samp_hf_out = np.zeros((0,dim))
                Y_samp_hf_out = np.zeros(0)
                # reset variables
                fail_lf_acc = [None]*max_C_lf_fail
                success_lf_acc = [None]*max_C_lf_success
                if debug and dim in [1,2] and visual_debug:
                    intr_iter = 0
            
        else: # i.e., single-fidelity optimization 
            
            fit_bd_old = fit_bd # save fit bound
            n_zoom_old = n_zoom # save n_zoom
            best_Y_new = best_val_it[k+init_iter] # minimum of Y values of newly proposed points
            # find x_star
            if not full_restart:
                Y_fit = rbf_mod(X_samp)
                min_ix = np.argmin(Y_fit)
                x_star = X_samp[min_ix]
            else:
                x_star = None
            status_var_list = [gSRS_pct_full,n_expand,n_zoom_out,n_zoom,fit_bd_save,
                               fit_bd,X_samp,Y_samp,X_samp_out,Y_samp_out,wgt_expon,
                               n_reduce_step_size,step_size_fact,n_fail,full_restart,
                               n_full_restart,x_star_list]
            status_var_list = prepare(gSRS_pct,best_Y_new,best_Y_prev,x_star,status_var_list)            
            # update variables
            gSRS_pct_full,n_expand,n_zoom_out,n_zoom,fit_bd_save,\
            fit_bd,X_samp,Y_samp,X_samp_out,Y_samp_out,wgt_expon,\
            n_reduce_step_size,step_size_fact,n_fail,full_restart,n_full_restart,\
            x_star_list = status_var_list
            
            if debug and dim in [1,2] and visual_debug and n_zoom_old != n_zoom and not full_restart:
                abs(n_zoom_old-n_zoom) == 1
                zoom_in = True if n_zoom_old < n_zoom else False
                zoom_str = 'in' if zoom_in else 'out'
                seq += 1
                nx = 100
                plt.figure(0)
                plt.clf()
                
                if dim == 2:
                    if zoom_in:
                        # new fit bound
                        bd1, bd2 = zip(*fit_bd)
                        plt.plot([bd1[0],bd1[0],bd2[0],bd2[0],bd1[0]],[bd1[1],bd2[1],bd2[1],bd1[1],bd1[1]],'k--')
                        # old fit bound
                        x1_arr = np.linspace(fit_bd_old[0][0],fit_bd_old[0][1],nx)
                        x2_arr = np.linspace(fit_bd_old[1][0],fit_bd_old[1][1],nx)
                        X1, X2 = np.meshgrid(x1_arr, x2_arr)
                        pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                        # fit function contour
                        fit_val = rbf_mod(pts)
                        if len(set(fit_val)) > 1:
                            plt.contour(X1,X2,fit_val.reshape((nx,nx)),20)
                    else:
                        # old fit bound
                        bd1, bd2 = zip(*fit_bd_old)
                        plt.plot([bd1[0],bd1[0],bd2[0],bd2[0],bd1[0]],[bd1[1],bd2[1],bd2[1],bd1[1],bd1[1]],'k--')
                        # new fit bound
                        x1_arr = np.linspace(fit_bd[0][0],fit_bd[0][1],nx)
                        x2_arr = np.linspace(fit_bd[1][0],fit_bd[1][1],nx)
                        X1, X2 = np.meshgrid(x1_arr, x2_arr)
                        pts = np.hstack((X1.reshape(nx*nx,1),X2.reshape(nx*nx,1)))
                        # fit function contour
                        fit_val = rbf_mod(pts)
                        if len(set(fit_val)) > 1:
                            plt.contour(X1,X2,fit_val.reshape((nx,nx)),20)
                    
                    # evaluated samples
                    plt.plot(X_samp[:,0],X_samp[:,1],'ko')
                    if len(X_samp_out) > 0:
                        plt.plot(X_samp_out[:,0],X_samp_out[:,1],color='white',marker='o',ls='')
                        
                    # x_star
                    if gSRS_pct < 1 and zoom_in:
                        plt.plot(x_star[0],x_star[1],'rx',mew=1) # mew: marker edge width
                    
                    plt.xlabel(prob.x_var[0], fontsize=label_fs)
                    plt.ylabel(prob.x_var[1], fontsize=label_fs)
                    
                    if zoom_in:
                        plt.xlim(fit_bd_old[0])
                        plt.ylim(fit_bd_old[1])
                    else:
                        plt.xlim(fit_bd[0])
                        plt.ylim(fit_bd[1])
                        
                    if len(set(fit_val)) > 1:
                        plt.colorbar()
                
                elif dim == 1:
                            
                    if zoom_in:
                        # new fit bound
                        plt.axvline(x=fit_bd[0][0],color='k',ls='--')
                        plt.axvline(x=fit_bd[0][1],color='k',ls='--')
                        # old fit bound
                        pts = np.linspace(fit_bd_old[0][0], fit_bd_old[0][1], nx).reshape((-1,1))
                        fit_val = rbf_mod(pts)
                        plt.plot(pts,fit_val,'b-')
                        
                    else:
                        # old fit bound
                        plt.axvline(x=fit_bd_old[0][0],color='k',ls='--')
                        plt.axvline(x=fit_bd_old[0][1],color='k',ls='--')
                        # new fit bound
                        pts = np.linspace(fit_bd[0][0], fit_bd[0][1], nx).reshape((-1,1))
                        fit_val = rbf_mod(pts)
                        plt.plot(pts,fit_val,'b-')
                    
                    # evaluated samples
                    plt.plot(X_samp,rbf_mod(X_samp),'ko')
                    if len(X_samp_out) > 0:
                        plt.plot(X_samp_out,rbf_mod(X_samp_out),color='white',marker='o',ls='')
                        
                    # x_star
                    if gSRS_pct < 1 and zoom_in:
                        plt.plot(x_star,rbf_mod(x_star.reshape((-1,1))),'rx',mew=1) # mew: marker edge width
                    
                    plt.xlabel(prob.x_var[0], fontsize=label_fs)
                    plt.ylabel(prob.y_var, fontsize=label_fs)
                    
                    if zoom_in:
                        plt.xlim(fit_bd_old[0])
                    else:
                        plt.xlim(fit_bd[0])
                
                ax = plt.gca()
                plt.text(0.5, 1.03, 'Iteration %d: zoom %s\n' % (intr_iter,zoom_str),
                         horizontalalignment='center', fontsize=main_title_fs, transform=ax.transAxes,
                         fontweight='bold')
                plt.title('gSRS_p = %g, reduce = %d, expand = %d, zoom = %d' \
                          % (gSRS_pct, n_reduce_step_size, n_expand, n_zoom), 
                          fontsize=title_fs, y = 1.01)
                plt.savefig('%s/%s_iter%d_seq%d_zoom_%s.pdf' % (outdir_debug,prob_name,k+1+init_iter,seq,zoom_str))
                plt.close()
            
            if full_restart and n_full_restart == 0:
                # reset variables
                if debug and dim in [1,2] and visual_debug:
                    intr_iter = 0
                # re-initialize program to see if we need to go back to multi-fidelity optimization
                multi_fidelity,status_param_list = program_init()
                if multi_fidelity:
                    # update status parameters
                    n_proc_hf,n_proc_lf,n_worker_hf,n_worker_per_proc_lf,n_worker_lf,\
                    obj_func_lf,obj_func_hf = status_param_list
                    # reset in-box and out-of-box samples
                    X_samp_lf = np.zeros((0,dim))
                    Y_samp_lf = np.zeros(0)
                    X_samp_lf_out = np.zeros((0,dim))
                    Y_samp_lf_out = np.zeros(0) 
                    X_samp_hf = np.zeros((0,dim))
                    Y_samp_hf = np.zeros(0)
                    X_samp_hf_out = np.zeros((0,dim))
                    Y_samp_hf_out = np.zeros(0)
                    # reset variables
                    fail_lf_acc = [None]*max_C_lf_fail
                    success_lf_acc = [None]*max_C_lf_success
        
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
                 init_iter=init_iter,opt_iter=k+1,n_proc=n_proc,seed=seed,outdir=outdir,
                 save_samp=save_samp,verbose=verbose,n_cand_fact=n_cand_fact,doe_check_lf_acc=doe_check_lf_acc,
                 normalize_data=normalize_data,init_wgt_expon=init_wgt_expon,wgt_expon_delta=wgt_expon_delta,
                 max_reduce_step_size=max_reduce_step_size,box_fact=box_fact,init_step_size_fact=init_step_size_fact,
                 init_gSRS_pct=init_gSRS_pct,alpha_amp=alpha_amp,serial_mode=serial_mode,
                 debug=debug,log_std_out=log_std_out,log_std_err=log_std_err,sm_range=sm_range,log_opt=log_opt,
                 n_fold=n_fold,resol=resol,min_zoom_in=min_zoom_in,max_zoom_out=max_zoom_out,min_lf_test_samp=min_lf_test_samp,
                 x_star_std_factor=x_star_std_factor,max_C_expand_fit=max_C_expand_fit,use_eff_n_samp=use_eff_n_samp,
                 rbf_kernel=rbf_kernel,soft_func_bd=soft_func_bd,max_C_lf_fail=max_C_lf_fail,lf_test_alpha=lf_test_alpha,
                 func_bd=func_bd,max_C_lf_success=max_C_lf_success,max_C_fail=max_C_fail,visual_debug=visual_debug,
                 dynam_box_fact=dynam_box_fact,round_gSRS_pct_full=round_gSRS_pct_full,alpha_crit_err=alpha_crit_err,
                 alpha_base=alpha_base,resume_opt_iter=resume_opt_iter,
                 # optimization results
                 t_build_mod=t_build_mod[:k+1],t_build_mod_cpu=t_build_mod_cpu[:k+1],t_prop_pt=t_prop_pt[:k+1],
                 t_prop_pt_cpu=t_prop_pt_cpu[:k+1],t_eval_prop=t_eval_prop[:k+1],t_eval_prop_cpu=t_eval_prop_cpu[:k+1],
                 t_opt=t_opt[:k+1],t_opt_cpu=t_opt_cpu[:k+1],t_prep=t_prep[:k+1],t_prep_cpu=t_prep_cpu[:k+1],
                 opt_sm_arr=opt_sm_arr[:k+1],cv_err_arr=cv_err_arr[:k+1],
                 gSRS_pct_arr=gSRS_pct_arr[:k+1],n_zoom_arr=n_zoom_arr[:k+1],n_expand_arr=n_expand_arr[:k+1],
                 expand_status_arr=expand_status_arr[:k+1],expand_mag_arr=expand_mag_arr[:k+1],
                 hf_prop_arr=hf_prop_arr[:init_iter+k+1],fidelity_status_arr=fidelity_status_arr[:init_iter+k+1],
                 best_loc_it_lf=best_loc_it_lf[:init_iter+k+1],high_opt_arr=high_opt_arr[:k+1],
                 best_val_it_lf=best_val_it_lf[:init_iter+k+1],best_loc_it_hf=best_loc_it_hf[:init_iter+k+1],
                 best_val_it_hf=best_val_it_hf[:init_iter+k+1],best_loc_it=best_loc_it[:init_iter+k+1],
                 best_val_it=best_val_it[:init_iter+k+1], best_seed_it=best_seed_it[:init_iter+k+1],
                 best_seed_it_lf=best_seed_it_lf[:init_iter+k+1],best_seed_it_hf=best_seed_it_hf[:init_iter+k+1],
                 # state variables
                 intr_iter=intr_iter,full_restart=full_restart,multi_fidelity=multi_fidelity,
                 X_samp_lf=X_samp_lf,Y_samp_lf=Y_samp_lf,wgt_expon=wgt_expon,X_samp_hf=X_samp_hf,
                 Y_samp_hf=Y_samp_hf,X_samp=X_samp,Y_samp=Y_samp,x_star_list=x_star_list,
                 gSRS_pct_full=gSRS_pct_full,fit_bd=fit_bd,n_proc_hf=n_proc_hf,n_worker_hf=n_worker_hf,
                 n_proc_lf=n_proc_lf,n_worker_per_proc_lf=n_worker_per_proc_lf,step_size_fact=step_size_fact,
                 n_fail=n_fail,n_expand=n_expand,n_reduce_step_size=n_reduce_step_size,n_zoom=n_zoom,
                 n_zoom_out=n_zoom_out,X_samp_lf_out=X_samp_lf_out,Y_samp_lf_out=Y_samp_lf_out,
                 n_worker_lf=n_worker_lf,X_samp_hf_out=X_samp_hf_out,Y_samp_hf_out=Y_samp_hf_out,
                 X_samp_out=X_samp_out,Y_samp_out=Y_samp_out,n_worker=n_worker,n_full_restart=n_full_restart,
                 seed_base=seed_base,seed_flat_arr=seed_flat_arr,fail_lf_acc=fail_lf_acc,
                 success_lf_acc=success_lf_acc,fit_bd_save=fit_bd_save,alpha=alpha,x_star_err=x_star_err,
                 gSRS_pct=gSRS_pct)
        
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
        
    # find best point and its value
    if not multi_fidelity:
        min_ix = np.argmin(best_val_it)
        best_loc = best_loc_it[min_ix]
        best_val = best_val_it[min_ix]
    else:
        raise ValueError('For multi-fidelity optimization, finding the best point is not implemented at the moment.')
        
    return best_loc, best_val