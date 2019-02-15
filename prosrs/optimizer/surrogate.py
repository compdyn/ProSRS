"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Contain functions and class that deal with surrogates.
"""
import numpy as np
from scipy.special import xlogy
from scipy import linalg
import warnings
from sklearn import preprocessing
from sklearn.model_selection import KFold
from functools import partial
from ..utility.function import unique_row


class Rbf(object):
    """
    A class dealing with L2-regularized RBF regression.
    
    The code is inherited from `scipy.interpolate.Rbf <https://github.com/scipy/scipy/blob/v0.19.0/scipy/interpolate/rbf.py>`,
    and is further developed by Chenchao Shou.
    
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
        Whether to use scipy rbf. Default: False.
    suppress_warning: boolean, optional
        Whether to suppress warnings when doing linalg.solve. Default: True.
    
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
        self.use_scipy_rbf = kwargs.pop('use_scipy_rbf', False)
        self.suppress_warning = kwargs.pop('suppress_warning', True)
        self.wgt = kwargs.pop('wgt',np.ones(self.N)) # weight vector        
        assert ((len(self.wgt) == self.N) and np.min(self.wgt)>=0), 'invalid weight vector'
        W = np.diag(self.wgt) # construct weight matrix
        self.deg = kwargs.pop('deg',0) # degree of polynomials (if = 0, then no polynomial tail)
        assert(self.deg in [0,1]), 'currently we only support degree up to 1'
        
        with warnings.catch_warnings():
            if self.suppress_warning:
                warnings.simplefilter('ignore')
            else:
                warnings.simplefilter('default')
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
    
    
def RBF_reg(X, Y, sm_range, normalize_data=True, wgt_expon=0., use_scipy_rbf=False,
            log_opt=True, n_fold=5, n_min_sm=10, kernel='multiquadric', pool=None, poly_deg=0):
    """
    Construct a RBF regression model using L2 regularization.
    
    Args:
        
        X (2d array): Data points (each row is one point).
        
        Y (1d array): (Noisy) function evaluation of each point in `X`.
        
        sm_range (list or tuple or float): Range of smoothing parameter.
            If ``list`` or ``tuple``, then `sm_range` = [lower bound, upper bound].
            The value of the smoothing parameter will be determined via a cross validation procedure
            on the domain `sm_range`. If ``float``, then the smoothing parameter is equal to
            `sm_range` (no cross validation in this case).
        
        normalize_data (bool, optional): Whether to normalize data before training a RBF model.
        
        wgt_expon (float, optional): Weight exponent in RBF regression.
            Useful only when `use_scipy_rbf` = False.
        
        use_scipy_rbf (bool, optional): Whether to use Scipy implementation for the RBF model.
        
        log_opt (bool, optional): Whether to optimize the smooth parameter on a log scale 
            during cross validation.
        
        
        n_fold (int, optional): Number of folds for the cross validation.
        
        n_min_sm (int, optional): Minimum number of samples on the domain `sm_range`.
            The value of the smoothing parameter will be selected to be the one, among
            these samples, of which the cross-validated error is the lowest.
        
        kernel (str, optional): RBF kernel. For the possible values and their meanings, see
            `scipy.interpolate.Rbf <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.interpolate.Rbf.html>`.
        
        pool (`pathos.multiprocessing.ProcessingPool` type or None, optional): Pool object for parallel computing. 
            Here we use `pathos package <https://pypi.org/project/pathos/0.2.0>` for parallelism.
            If None, then the cross validation will be conducted in serial.
            
        poly_deg (int, optional): Degree of RBF polynomial tail (either 0 or 1). 
            If zero, then no polynomial tail. Useful only when `use_scipy_rbf` = False.
    
    Returns:
        
        rbf_mod (callable): RBF regression model.
        
        opt_sm (float): (Optimal) smooth parameter found through cross validation.
        
        cv_err (list or None): cross validated error for each smooth parameter sample.
            If None, then there is no cross validation.
        
        rbf_obj (`Rbf` type or None): Original RBF instance before normalization. 
            If None, then the model is degenerated to a constant function.
    """
    # sanity check
    assert(len(X) == len(Y) > 0)
    assert(len(sm_range) in [1, 2])
    assert(n_fold > 1)
    
    n_uniq = len(unique_row(X)) # find number of unique X
    if n_uniq > 1:
        # then we build RBF surrogate
        if normalize_data:
            # normalize X and Y to [0,1]
            X_scaler = preprocessing.MinMaxScaler()
            Y_scaler = preprocessing.MinMaxScaler()
            X = X_scaler.fit_transform(X)
            Y = Y_scaler.fit_transform(Y.reshape((-1, 1))).flatten()
        else:
            X_scaler = Y_scaler = None 
            
        if len(sm_range) == 2:
            sm_lw, sm_up = sm_range
            assert(sm_lw < sm_up), 'invalid sm_range'
            
            n_proc = pool.nodes
            n_iter = int(np.ceil(n_min_sm/float(n_proc))) # number of iterations
            n_req = n_proc*n_iter # total number of requested points
            assert(n_req >= n_min_sm > 0) # sanity check
            # find candidate smooth parameters
            if log_opt:
                smooth = np.logspace(np.log10(sm_lw), np.log10(sm_up), n_req)
            else:
                smooth = np.logspace(sm_lw,sm_up,n_req)
            
            if pool is None:
                # then we compute in serial 
                cv_err = [CV_smooth(s, X, Y, n_fold, kernel, use_scipy_rbf, wgt_expon, poly_deg) for s in smooth]
            else:
                # then we compute in parallel
                CV_smooth_partial = partial(CV_smooth, X=X, Y=Y, n_fold=n_fold, kernel=kernel,
                                            use_scipy_rbf=use_scipy_rbf, wgt_expon=wgt_expon, poly_deg=poly_deg)        
                cv_err = pool.map(CV_smooth_partial, smooth, chunksize=n_iter)
            # find smooth parameter that has the lowest cross-validated error
            opt_ix = np.argmin(cv_err)
            opt_sm = smooth[opt_ix]
            
        else: # i.e., we specify the opt_sm
            opt_sm = sm_range[0]
            cv_err = None
            
        # then we build RBF surrogate with the optimal smooth parameter using all the data
        XY_data = np.vstack((X.T, Y.reshape((1, -1))))
        if not use_scipy_rbf:
            wgt = rbf_wgt(Y, wgt_expon)
            rbf_obj = Rbf(*XY_data, wgt=wgt, function=kernel, smooth=opt_sm, deg=poly_deg)        
        else:
            rbf_obj = Rbf(*XY_data, use_scipy_rbf=use_scipy_rbf, function=kernel, smooth=opt_sm)
        rbf_mod = normalize_rbf(rbf_obj, X_scaler, Y_scaler)
        
    else:
        # pathological case: X overlaps. Then the model is simply the mean of y values.
        opt_sm, cv_err, rbf_obj = np.nan, None, None
        def rbf_mod(x):
            """
            Degenerated RBF model.
            
            Args:
                
                x (2d array): Points to be evaluated. Each row is one point.
            
            Returns:
                
                y (1d array): Function evaluations at points in `x`.
            """
            assert(x.ndim == 2)
            nx = len(x)
            y = np.mean(Y)*np.ones(nx)
            
            return y
        
    return rbf_mod, opt_sm, cv_err, rbf_obj


def normalize_rbf(rbf_obj, X_scaler, Y_scaler):
    """
    Generate a normalized RBF model.
    
    Args:
        
        rbf_obj (`Rbf` type): Original RBF model.
        
        X_scaler (`sklearn.preprocessing` scaler type or None): Scaler for input X. 
            If None, then no normalization will be performed on the model input.
                    
        Y_scaler (`sklearn.preprocessing` scaler type or None): Scaler for output Y. 
            If None, then no normalization will be performed on the model output.
            
    Returns:
        
        norm_rbf_mod (callable): Normalized Rbf model.
    """
    def norm_rbf_mod(X):
        """
        Normalized RBF model.
        
        Args:
            
            X (2d array): Points to be evaluated. Each row is one point.
        
        Returns:
            
            Y (1d array): Function evaluations at points in `X`.
        """
        assert(X.ndim == 2)
        
        if X_scaler is not None:
            X = X_scaler.transform(X)
        Y = rbf_obj(*X.T)
        if Y_scaler is not None:
            Y = Y_scaler.inverse_transform(Y.reshape((-1, 1))).flatten()
            
        return Y
    
    return norm_rbf_mod


def rbf_wgt(Y, wgt_expon):
    """
    Generate weights of training data for RBF regression.
    
    Args:
        
        Y (1d array): Y values of training data.
        
        wgt_expon (float): Weight exponent.
    
    Returns:
        
        wgt (1d array): Weights of training data.
    """
    assert(Y.ndim == 1 and wgt_expon >= 0)
    
    min_Y, max_Y = np.min(Y), np.max(Y)
    if min_Y == max_Y:
        Y = np.zeros_like(Y)
    else:
        Y = (Y-min_Y)/(max_Y-min_Y) # normalization
    assert (np.min(Y) >= 0 and np.max(Y) <= 1) # sanity check
    wgt = np.exp(-wgt_expon*Y)
    
    return wgt
 
       
def CV_smooth(smooth, X, Y, n_fold, kernel, use_scipy_rbf, wgt_expon, poly_deg):
     """
     Get cross validated error for a smoothing parameter.
     
     Args:
         
         smooth (float): smoothing parameter.
         
         X (2d array): X values of training data. Each row is one point.
         
         Y (1d array): Y values of training data (1d array).
         
         n_fold (int): Number of folds for cross validation.
         
         kernel (str): RBF kernel. For the possible values and their meanings, see
            `scipy.interpolate.Rbf <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.interpolate.Rbf.html>`.
         
         use_scipy_rbf (bool): Whether to use Scipy implementation for the RBF model.
         
         wgt_expon (float): Weight exponent in RBF regression.
            Useful only when `use_scipy_rbf` = False.
     
         poly_deg (int): Degree of RBF polynomial tail (either 0 or 1). 
            If zero, then no polynomial tail. Useful only when `use_scipy_rbf` = False.
     
     Returns:
         
         cv_err (float): Cross validated error.
     """
     nX = len(X)
     n_fold = min(nX, n_fold)
     assert(n_fold > 1) # sanity check
     
     kf = KFold(n_splits=n_fold)
     test_err = np.zeros(n_fold)
     for f, index in enumerate([j for j in kf.split(np.ones(nX))]):
        train_ix, test_ix = index
        X_train, X_test = X[train_ix], X[test_ix]
        Y_train, Y_test = Y[train_ix], Y[test_ix]
        # get number of unique X_train
        n_uniq = len(unique_row(X_train))
        if n_uniq > 1:
            # then we build RBF model
            XY_data = np.vstack((X_train.T, Y_train.reshape((1, -1))))
            if not use_scipy_rbf:
                wgt = rbf_wgt(Y_train, wgt_expon)
                rbf_obj = Rbf(*XY_data, wgt=wgt, function=kernel, smooth=smooth, deg=poly_deg)
            else:
                rbf_obj = Rbf(*XY_data, use_scipy_rbf=use_scipy_rbf, function=kernel, smooth=smooth)
            test_err[f], _ = eval_rbf_mod(X_test, Y_test, rbf_obj)
        else:
            # pathological case: X_train overlaps. Then the model is simply the mean of y values.
            Y_pred = np.mean(Y_train)
            error = Y_pred - Y_test
            test_err[f] = np.sqrt(np.mean(error**2)) # RMSE
     cv_err = np.mean(test_err)
     
     return cv_err


def eval_rbf_mod(X, Y, rbf_obj):
    """
    Evaluate performance of a RBF model on a test dataset.
    
    Args:
        
        X (2d array): X values of test data. Each row is one point.
         
        Y (1d array): Y values of test data (1d array).
        
        rbf_obj (`Rbf` type): RBF model to be evaluated.
    
    Returns:
        
        RMSE (float): Root mean square error on the test dataset.
        
        error (1d array): Prediction errors on the test dataset.
    """
    assert(X.ndim == 2 and Y.ndim == 1)
    
    error = rbf_obj(*X.T)-Y
    RMSE = np.sqrt(np.mean(error**2))
    
    return RMSE, error