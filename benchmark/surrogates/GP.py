"""
Copyright (C) 2018 Chenchao Shou
Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

This code implements Gaussian process regression using Python scikit-learn package
(http://scikit-learn.org/stable/modules/gaussian_process.html).
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel

def GP_reg(X,Y,kernel='Matern',kernel_param={'nu': 2.5},n_restarts_optimizer=0):
    '''
    Input:
        X: x values of data points (2d array, different rows are different points)
        Y: y values of data points (1d array)
        kernel: string, GP kernel: ['RBF', 'Matern']
        kernel_param: dictionary, kernel parameters
        n_restarts_optimizer: int, number of restarts for the optimizer (see scikit-learn documentation for more)
    Output:
        gp_mod: function that takes 2d array and output 1d array, GP model (see scikit-learn documentation for more)
        gp_obj: GP object (see scikit-learn documentation for more)
    '''
    assert(kernel in ['RBF', 'Matern'] and len(X) == len(Y) > 0 and n_restarts_optimizer>=0)
    assert(X.ndim == 2 and Y.ndim == 1)
    
    dimX = X.shape[1] # dimension of X
    # get GP kernel (ConstantKernel is for mean level estimation and WhiteKernel is for noise level estimation)
    if kernel == 'RBF':
        knl = ConstantKernel()+1.*RBF(length_scale=np.ones(dimX))+WhiteKernel() # ARD (i.e., a general anisotropic kernel)
    elif kernel == 'Matern':
        assert(kernel_param['nu'] > 0)
        knl = ConstantKernel()+1.* Matern(length_scale=np.ones(dimX), nu=kernel_param['nu'])+WhiteKernel() # ARD (i.e., a general anisotropic kernel)
    # train GP model
    gp_obj = GaussianProcessRegressor(kernel=knl,alpha=0.0,n_restarts_optimizer=n_restarts_optimizer).fit(X, Y)
    gp_mod = gp_obj.predict
    
    return gp_mod, gp_obj

