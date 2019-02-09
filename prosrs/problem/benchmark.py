"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Contains optimization benchmark problems.
"""
import numpy as np
from .problem import Problem
    
def benchmark(func_name, domain=None, sd=None):
    """
    Generate an optimization benchmark problem.
    
    Args:
        
        func_name (str): Benchmark function name.
        
        domain (list of tuples or None, optional): Optimization domain.
                For example, `domain` = [(0, 1), (0, 2)] means optimizing a
                2D function of argument ``x`` on the domain: 
                ``0 <= x[0] <= 1`` and ``0 <= x[1] <= 2``. If None, we use default
                value associated with `func_name`.
                
        sd (float, optional): Standard deviation of Gaussian noise added to the benchmark function.
            If None, then we use the default value associated with `func_name`.
    
    Returns:
        
        prob (`Problem` type): Optimization problem instance for benchmark `func_name`.
        
    Raises:
        
        ValueError: If `func_name` is not implemented.
    """
    if func_name == 'Schaffer':
        sd = 0.02 if sd is None else sd
        prob = benchmark_to_problem(Schaffer(domain), sd)
        
    elif func_name == 'Goldstein':
        sd = 2. if sd is None else sd
        prob = benchmark_to_problem(Goldstein(domain), sd)
        
    elif func_name == 'DropWave':
        sd = 0.02 if sd is None else sd
        prob = benchmark_to_problem(DropWave(domain), sd)
        
    elif func_name == 'SixHumpCamel':
        sd = 0.1 if sd is None else sd
        prob = benchmark_to_problem(SixHumpCamel(domain), sd)
        
    elif func_name == 'Ackley10':
        sd = 1. if sd is None else sd
        prob = benchmark_to_problem(Ackley(dim=10, domain), sd)
        
    elif func_name == 'Alpine10':
        sd = 1. if sd is None else sd
        prob = benchmark_to_problem(Alpine(dim=10, domain), sd)
        
    elif func_name == 'Hartmann6':
        sd = 0.05 if sd is None else sd
        prob = benchmark_to_problem(Hartmann6(domain), sd)
        
    elif func_name == 'Rastrigin2':
        sd = 0.5 if sd is None else sd
        prob = benchmark_to_problem(Rastrigin(dim=2, domain), sd)
        
    elif func_name == 'Griewank10':
        sd = 2. if sd is None else sd
        prob = benchmark_to_problem(Griewank(dim=10, domain), sd)
        
    elif func_name == 'Levy10':
        sd = 1. if sd is None else sd
        prob = benchmark_to_problem(Levy(dim=10, domain), sd)
        
    elif func_name == 'SumPower10':
        sd = 0.05 if sd is None else sd
        prob = benchmark_to_problem(SumPower(dim=10, domain), sd)
        
    elif func_name == 'PowerSum':
        sd = 1. if sd is None else sd
        prob = benchmark_to_problem(PowerSum(domain), sd)
        
    else:
        raise ValueError('Unknown func_name.') 
    
    return prob


def benchmark_to_problem(benchmark, sd):
    """
    Convert a benchmark instance to its corresponding optimization problem instance.
    
    Args:
        
        benchmark (Benchmark function type (e.g. `Schaffer`)): Benchmark instance.
        
        sd (float): Standard deviation of Gaussian noise added to the benchmark function.
        
    Returns:
        
        prob (`Problem` type): Optimization problem instance corresponding to `benchmark`.
    
    Raises:
        
        ValueError: If `sd` < 0.
    
    """
    if sd < 0:
        raise ValueError('sd must be non-negative.')
        
    func = lambda x: benchmark.f(x)+np.random.normal(0, sd) # add Gaussian noise
    prob = Problem(func, domain=benchmark.domain, name=benchmark.name, 
                   true_func=benchmark.f, min_loc=benchmark.min, 
                   min_true_func=benchmark.fmin)
    
    return prob


class Schaffer:
    """
    `Schaffer N.2 function <https://www.sfu.ca/~ssurjano/schaffer2.html>`.
    """
    def __init__(self, domain=None):
        """
        Constructor.
        
        Args:
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.domain = [(-100., 100.), (-100., 100.)] if domain is None else domain
        self.dim = len(self.domain)
        self.min = [(0., 0.)] # global minimum locations
        self.fmin = 0. # global minimum
        self.name = 'Schaffer'
        
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        x1 = x[0]
        x2 = x[1]
        y = 0.5+(np.sin(x1**2-x2**2)**2-0.5)/(1+(1e-3)*(x1**2+x2**2))**2
        
        return y
        
    
class Goldstein:
    """
    `Goldstein-Price function <https://www.sfu.ca/~ssurjano/goldpr.html>`.
    """
    def __init__(self, domain=None):
        """
        Constructor.
        
        Args:
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.domain = [(-2., 2.),(-2., 2.)] if domain is None else domain
        self.dim = len(self.domain)
        self.min = [(0., -1.)] # global minimum locations
        self.fmin = 3. # global minimum
        self.name = 'Goldstein'
    
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        x1 = x[0]
        x2 = x[1]
        fact1a = (x1 + x2 + 1)**2
        fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
        fact1 = 1 + fact1a*fact1b
        fact2a = (2*x1 - 3*x2)**2
        fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
        fact2 = 30 + fact2a*fact2b
        y = fact1*fact2
        
        return y


class SixHumpCamel:
    """
    `Six-hump camel function <https://www.sfu.ca/~ssurjano/camel6.html>`.
    """
    def __init__(self, domain=None):
        """
        Constructor.
        
        Args:
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.domain = [(-3., 3.), (-2., 2.)] if domain is None else domain
        self.dim = len(self.domain)
        self.min = [(0.0898, -0.7126), (-0.0898, 0.7126)] # global minimum locations
        self.fmin = -1.0316 # global minimum
        self.name = 'SixHumpCamel'

    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        x1 = x[0]
        x2 = x[1]
        term1 = (4-2.1*x1**2 + (x1**4)/3) * x1**2
        term2 = x1*x2
        term3 = (-4+4*x2**2) * x2**2
        y = term1 + term2 + term3
        
        return y


class DropWave:
    """
    `Drop-wave function <https://www.sfu.ca/~ssurjano/drop.html>`.
    """
    def __init__(self, domain=None):
        """
        Constructor.
        
        Args:
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.domain = [(-5.12, 5.12), (-5.12, 5.12)] if domain is None else domain
        self.dim = len(self.domain)
        self.min = [(0., 0.)] # global minimum locations
        self.fmin = -1. # global minimum
        self.name = 'DropWave'

    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        y = -(1+np.cos(12*np.sqrt(x[0]**2+x[1]**2))) / (0.5*(x[0]**2+x[1]**2)+2)
        
        return y

        
class Alpine:
    """
    `Alpine function <http://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Alpine01>`.
    """
    def __init__(self, dim, domain=None):
        """
        Constructor.
        
        Args:
            
            dim (int): Dimension of function.
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.dim = dim
        self.domain = [(-10., 10.)]*dim if domain is None else domain
        self.min = [tuple([0.]*dim)] # global minimum locations
        self.fmin = 0. # global minimum
        self.name = 'Alpine%d' % dim
    
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        y = np.absolute(x*np.sin(x) + 0.1*x).sum()
        
        return y


class Ackley:
    """
    `Ackley function <https://www.sfu.ca/~ssurjano/ackley.html>`.
    """
    def __init__(self, dim, domain=None):
        """
        Constructor.
        
        Args:
            
            dim (int): Dimension of function.
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.dim = dim
        self.domain =[(-32.768, 32.768)]*dim if domain is None else domain
        self.min = [tuple([0.]*dim)] # global minimum locations
        self.fmin = 0. # global minimum
        self.name = 'Ackley%d' % dim

    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        y = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((x**2).sum()/self.dim))\
            -np.exp(np.cos(2*np.pi*x).sum()/self.dim))
        
        return y

        
class Hartmann6:
    """
    `Hartmann6 function <https://www.sfu.ca/~ssurjano/hart6.html>`.
    """
    def __init__(self, domain=None):
        """
        Constructor.
        
        Args:
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.dim = 6
        self.domain =[(0., 1.)]*self.dim if domain is None else domain
        self.min = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)] # global minimum locations
        self.fmin = -3.32237 # global minimum
        self.name = 'Hartmann6'
        
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        a = np.array([1, 1.2, 3, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], 
                      [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])     
        P = 10**(-4)*np.array([[1312, 1696, 5569, 124, 8283, 5886], 
                [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381]])
        x = x.reshape((1, -1))
        y = np.zeros((1, 1))
        for i in range(4):
            ai, Ai, Pi = a[i], A[i], P[i]
            y += -ai*np.exp(-np.dot((x-np.outer(np.ones(1), Pi))**2, Ai.reshape((-1,1))))
        y = y.item(0)
        
        return y

        
class Rastrigin:
    """
    `Rastrigin function <https://www.sfu.ca/~ssurjano/rastr.html>`.
    """
    def __init__(self, dim, domain=None):
        """
        Constructor.
        
        Args:
            
            dim (int): Dimension of function.
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.dim = dim
        self.domain =[(-5.12, 5.12)]*dim if domain is None else domain
        self.min = [tuple([0.]*dim)] # global minimum locations
        self.fmin = 0. # global minimum
        self.name = 'Rastrigin%d' % dim
        
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        fval = 10*self.dim+np.sum(x**2-10*np.cos(2*np.pi*x))
        
        return y


class Griewank:
    """
    `Griewank function <https://www.sfu.ca/~ssurjano/griewank.html>`.
    """
    def __init__(self, dim, domain=None):
        """
        Constructor.
        
        Args:
            
            dim (int): Dimension of function.
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.dim = dim
        self.domain =[(-600., 600.)]*dim if domain is None else domain
        self.min = [tuple([0.]*dim)] # global minimum locations
        self.fmin = 0. # global minimum
        self.name = 'Griewank%d' % dim
        
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        x = x.reshape((1, -1))
        y = np.sum(x**2, axis=1)/4000.-np.prod(np.cos(x/np.sqrt(np.outer(np.ones(1), range(1, self.dim+1)))), axis=1)+1
        y = y.item(0)
        
        return y


class Levy:
    """
    `Levy function <https://www.sfu.ca/~ssurjano/levy.html>`.
    """
    def __init__(self, dim, domain=None):
        """
        Constructor.
        
        Args:
            
            dim (int): Dimension of function.
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.dim = dim
        self.domain =[(-10., 10.)]*dim if domain is None else domain
        self.min = [tuple([1.]*dim)] # global minimum locations
        self.fmin = 0. # global minimum
        self.name = 'Levy%d' % dim
        
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        W = 1+(x-1)/4.
        W0 = W[0]
        W1 = W[:-1]
        W2 = W[-1]
        term1 = np.sin(np.pi*W0)**2
        term2 = np.sum((W1-1)**2*(1+10*np.sin(np.pi*W1+1)**2))
        term3 = (W2-1)**2*(1+np.sin(2*np.pi*W2)**2)
        y = term1+term2+term3
        
        return y


class SumPower:
    """
    `Sum of powers function <https://www.sfu.ca/~ssurjano/sumpow.html>`.
    """
    def __init__(self, dim, domain=None):
        """
        Constructor.
        
        Args:
            
            dim (int): Dimension of function.
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.dim = dim
        self.domain =[(-1., 1.)]*dim if domain is None else domain
        self.min = [tuple([0.]*dim)] # global minimum locations
        self.fmin = 0. # global minimum
        self.name = 'SumPower%d' % dim
        
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        y = np.sum(np.absolute(x)**np.arange(2, self.dim+2))
        
        return y


class PowerSum:
    """
    `Power sum function <https://www.sfu.ca/~ssurjano/powersum.html>`.
    """
    def __init__(self, domain=None):
        """
        Constructor.
        
        Args:
            
            domain (list of tuples or None, optional): Optimization domain.
                If None, we use the default value.
                
        """
        self.dim = 4
        self.domain =[(0., 4.)]*self.dim if domain is None else domain
        self.min = [(1., 2., 2., 3.)] # global minimum locations
        self.fmin = 0. # global minimum
        self.name = 'PowerSum'
        
    def f(self, x):
        """
        Expression of function.
        
        Args:
            
            x (1d array): Function input.
                
        Returns:
            
            y (float): Function evaluation at `x`.
            
        Raises:
            
            ValueError: If dimension of `x` is wrong.
            
        """
        # sanity check
        if x.shape != (self.dim, ):
            raise ValueError('Wrong dimension for x.')
        # evaluate the function
        y = 0.
        bval = [8., 18., 44., 114.]
        for i,b in enumerate(bval):
            y += (np.sum(x**(i+1))-b)**2
        
        return y
