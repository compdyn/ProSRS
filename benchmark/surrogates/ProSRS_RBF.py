"""
Copyright (C) 2018 Chenchao Shou
Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

This code implements RBF regression used in ProSRS algorithm. The functions here are essentially
copies from 'src/ProSRS.py'.
"""
from __future__ import division, absolute_import

import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from functools import partial
from scipy import linalg
from scipy.special import xlogy
import warnings

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
def RBF_reg(X,Y,n_proc=1,sm_range=[1e-7,1e4],normalize_data=True,wgt_expon=0.0,use_scipy_rbf=False,
            log_opt=True,n_fold=5,n_min_sm=10,kernel='multiquadric',pool=None,poly_deg=0):
    '''
    Input:
    X: x values of data points (2d array, different rows are different points)
    Y: y values of data points (1d array)
    n_proc: number of processors (cores) for parallel optimization.
    sm_range: range of smooth parameter ([low,upper]) or a single smooth parameter in a list
    normalize_data: boolean. Whether to normalize data before training
    wgt_expon: scalar, weight exponent used in building weighted rbf, useful ONLY when use_scipy_rbf = False
    log_opt: whether to optimize the smooth parameter on a log scale (boolean)
    n_fold: number of folds for cross validation
    n_min_sm: minimum number of points for selecting the optimal smoothing parameter (Must be > 0)
    kernel: RBF kernel (see scipy.interpolate.Rbf for details)
    pool: pool of workers (used for parallel computing), an instance of pathos.multiprocessing.ProcessingPool object
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

