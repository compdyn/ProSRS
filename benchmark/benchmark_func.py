"""
Copyright (c) 2016, the GPyOpt Authors (https://github.com/SheffieldML/GPyOpt/tree/master/GPyOpt/objective_examples)
Licensed under the BSD 3-clause license:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Further developed by Chenchao Shou
Copyright (C) 2016-2018 Chenchao Shou

This code contains optimization benchmark problems.
"""
import numpy as np
    
########### Benchmark Function Classes ##############

class schaffer2:
    '''
    Schaffer N2 function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=0.):
        self.input_dim = 2
        self.bounds = [(-100,100),(-100,100)] if bounds is None else bounds
        self.min = [(0.,0.)]
        self.fmin = 0.
        self.sd = sd
        self.name = 'Schaffer2'
        
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        x1 = X[:,0]
        x2 = X[:,1]
        fval = 0.5+(np.sin(x1**2-x2**2)**2-0.5)/(1+(1e-3)*(x1**2+x2**2))**2
        
        return fval
    
class goldstein:
    '''
    Goldstein function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=0.):
        self.input_dim = 2
        self.bounds = [(-2.,2.),(-2.,2.)] if bounds is None else bounds
        self.min = [(0.,-1.)]
        self.fmin = 3.
        self.sd = sd
        self.name = 'Goldstein'
    
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function    
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        x1 = X[:,0]
        x2 = X[:,1]
        fact1a = (x1 + x2 + 1)**2
        fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
        fact1 = 1 + fact1a*fact1b
        fact2a = (2*x1 - 3*x2)**2
        fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
        fact2 = 30 + fact2a*fact2b
        fval = fact1*fact2
        
        return fval

class sixhumpcamel:
    '''
    Six-hump camel function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=0.):
        self.input_dim = 2
        self.bounds = [(-3,3),(-2,2)] if bounds is None else bounds
        self.min = [(0.0898,-0.7126),(-0.0898,0.7126)]
        self.fmin = -1.0316
        self.sd = sd
        self.name = 'Six_hump_camel'

    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        x1 = X[:,0]
        x2 = X[:,1]
        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
        term2 = x1*x2
        term3 = (-4+4*x2**2) * x2**2
        fval = term1 + term2 + term3
        
        return fval

class dropwave:
    '''
    Dropwave function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=0.):
        self.input_dim = 2
        self.bounds = [(-5.12,5.12),(-5.12,5.12)] if bounds is None else bounds
        self.min = [(0., 0.)]
        self.fmin = -1.
        self.sd = sd
        self.name = 'dropwave'

    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        fval = -(1+np.cos(12*np.sqrt(X[:,0]**2+X[:,1]**2))) / (0.5*(X[:,0]**2+X[:,1]**2)+2)
        
        return fval
        
class alpine1:
    '''
    Alpine1 function
    
    :param input_dim: the dimension of the function.
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self,input_dim,bounds=None,sd=0.):
        
        self.input_dim = input_dim
        self.bounds = [(-10,10)]*input_dim if bounds is None else bounds
        self.min = [tuple([0.]*input_dim)]
        self.fmin = 0.
        self.sd = sd
        self.name = 'alpine1_%d' % self.input_dim
    
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        fval = np.absolute(X*np.sin(X) + 0.1*X).sum(axis=1)
        
        return fval

class ackley:
    '''
    Ackley function 

    :param input_dim: the dimension of the function.
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=0.):
        self.input_dim = input_dim
        self.bounds =[(-32.768,32.768)]*self.input_dim if bounds is None else bounds
        self.min = [tuple([0.]*self.input_dim)]
        self.fmin = 0.
        self.sd = sd
        self.name = 'Ackley%d' % self.input_dim

    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(1)/self.input_dim))\
            -np.exp(np.cos(2*np.pi*X).sum(1)/self.input_dim))
        
        return fval
        
class hartmann6:
    '''
    Hartmann6 function 

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, bounds=None,sd=0.):
        self.input_dim = 6
        self.bounds =[(0,1)]*self.input_dim if bounds is None else bounds
        self.min = [(0.20169,0.150011,0.476874,0.275332,0.311652,0.6573)]
        self.fmin = -3.32237
        self.sd = sd
        self.name = 'Hartmann6'
        
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        n = X.shape[0]
        a = np.array([1.0,1.2,3.0,3.2])
        A = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])     
        P = 10**(-4)*np.array([[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],\
                [2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]])
        fval = np.zeros((n,1))
        for i in range(4):
            ai,Ai,Pi = a[i],A[i],P[i]
            fval += -ai*np.exp(-np.dot((X-np.outer(np.ones(n),Pi))**2,Ai.reshape((-1,1))))
        fval = fval.flatten()
        
        return fval
        
class rastrigin:
    '''
    Rastrigin function 

    :param input_dim: the dimension of the function.
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=0.):
        self.input_dim = input_dim
        self.bounds =[(-5.12,5.12)]*self.input_dim if bounds is None else bounds
        self.min = [tuple([0.]*self.input_dim)]
        self.fmin = 0.
        self.sd = sd
        self.name = 'Rastrigin%d' % self.input_dim
        
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        fval = 10*self.input_dim+np.sum(X**2-10*np.cos(2*np.pi*X),axis=1)
        
        return fval

class griewank:
    '''
    Griewank function 

    :param input_dim: the dimension of the function.
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=0.):
        self.input_dim = input_dim
        self.bounds =[(-600.0,600.0)]*self.input_dim if bounds is None else bounds
        self.min = [tuple([0.]*self.input_dim)]
        self.fmin = 0.
        self.sd = sd
        self.name = 'Griewank%d' % self.input_dim
        
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        n = X.shape[0]
        fval = np.sum(X**2,axis=1)/4000.0-np.prod(np.cos(X/np.sqrt(np.outer(np.ones(n),range(1,self.input_dim+1)))),axis=1)+1
        fval = fval.flatten()
        
        return fval

class levy:
    '''
    Levy function 

    :param input_dim: the dimension of the function.
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=0.):
        self.input_dim = input_dim
        self.bounds =[(-10.0,10.0)]*self.input_dim if bounds is None else bounds
        self.min = [tuple([1.]*self.input_dim)]
        self.fmin = 0.
        self.sd = sd
        self.name = 'Levy%d' % self.input_dim
        
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        W = 1+(X-1)/4.0
        W0 = W[:,0]
        W1 = W[:,:-1]
        W2 = W[:,-1]
        term1 = np.sin(np.pi*W0)**2
        term2 = np.sum((W1-1)**2*(1+10*np.sin(np.pi*W1+1)**2),axis=1)
        term3 = (W2-1)**2*(1+np.sin(2*np.pi*W2)**2)
        fval = term1+term2+term3
        
        return fval

class sum_power:
    '''
    Sum of Powers function 

    :param input_dim: the dimension of the function.
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=0.):
        self.input_dim = input_dim
        self.bounds =[(-1.0,1.0)]*self.input_dim if bounds is None else bounds
        self.min = [tuple([0.]*self.input_dim)]
        self.fmin = 0.
        self.sd = sd
        self.name = 'Sum_Power%d' % self.input_dim
        
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        n = X.shape[0]
        fval = np.sum(np.absolute(X)**np.outer(np.ones(n),range(2,self.input_dim+2)),axis=1)
        fval = fval.flatten()
        
        return fval

class power_sum:
    '''
    Power Sum function 

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, bounds=None,sd=0.):
        self.input_dim = 4
        self.bounds =[(0.0,4.0)]*self.input_dim if bounds is None else bounds
        self.min = [(1.,2.,2.,3.)]
        self.fmin = 0.
        self.sd = sd
        self.name = 'Power_Sum'
        
    def f(self,X,seed=None):
        """
        Objective function (corrupted with random noise)
        Input:
            X: list or 1d array or 2d array, input value(s) to the function. If 2d array, then each row is a point.
            seed: None or int, random seed. If None, then we do not set random seed
        Output:
            y: 1d array, output value of the function 
        """
        if seed is not None:
            seed = int(seed)
            assert(seed >= 0)
            np.random.seed(seed) # set seed for reproducibility
        X = np.array(X).reshape((-1,self.input_dim))
        n = X.shape[0]
        # generate random noise
        noise = np.zeros(n) if self.sd == 0 else np.random.normal(0,self.sd,n)
        # add noise to the true function
        y = self.tf(X)+noise
        assert(y.ndim == 1 and len(y) == n)
        
        return y
    
    def tf(self,X):
        """
        True objective function (expected value)
        Input:
            X: 2d array, input value(s) to the function (each row is a point).
        Output:
            fval: 1d array, output value of the function 
        """
        assert(X.ndim == 2 and X.shape[1]==self.input_dim), 'Wrong input dimension to tf'
        n = X.shape[0]
        fval = np.zeros(n)
        bval = [8.0,18.0,44.0,114.0]
        for i,b in enumerate(bval):
            fval += (np.sum(X**(i+1),axis=1)-b)**2
        
        return fval

####################### Function #################################
        
def get_benchmark_func(func_name):
    """
    A helper that returns an optimization benchmark object based on the function name.
    
    Input:
        func_name: string, function name
    Output:
        func: function object
    """
    if func_name == 'schaffer2':
        sd = 0.02
        dim = 2
        func = schaffer2(sd=sd)
    elif func_name == 'goldstein':
        sd = 2.
        dim = 2
        func = goldstein(sd=sd)
    elif func_name == 'dropwave':
        sd = 0.02
        dim = 2
        func = dropwave(sd=sd)  
    elif func_name == 'sixhumpcamel':
        sd = 0.1
        dim = 2
        func = sixhumpcamel(sd=sd)
    elif func_name == 'ackley10':
        sd = 1.
        dim = 10
        func = ackley(dim,sd=sd)
    elif func_name == 'alpine1_10':
        sd = 1.
        dim = 10
        func = alpine1(dim,sd=sd)
    elif func_name == 'hartmann6':
        sd = 0.05
        dim = 6
        func = hartmann6(sd=sd)
    elif func_name == 'rastrigin2':
        sd = 0.5
        dim = 2
        func = rastrigin(dim,sd=sd)
    elif func_name == 'griewank10':
        sd = 2.
        dim = 10
        func = griewank(dim,sd=sd)
    elif func_name == 'levy10':
        sd = 1.
        dim = 10
        func = levy(dim,sd=sd)
    elif func_name == 'sum_power10':
        sd = 0.05
        dim = 10
        func = sum_power(dim,sd=sd)
    elif func_name == 'power_sum':
        sd = 1.
        dim = 4
        func = power_sum(sd=sd)
    else:
        raise ValueError('invalid func_name to get_benchmark_func()')
    
    func_bd = func.bounds
    # sanity check
    assert(dim==len(func_bd)),'incorrect dim for optimization function %s!' % func_name    
    
    return func