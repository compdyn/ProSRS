"""
Copyright (C) 2018 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Define optimization problems.
"""

import numpy as np

####################### Functions #########################

def gen_problem(prob_def):
    """
    Generate an optimization problem from its definition.
    
    Input:
        prob_def: an object that defines an optimization problem.
    Output:
        prob: an optim_prob class object that is ready to be fed into ProSRS algorithm.
    """
    # Vectorize output of objective function. This ensures that the output
    # has type of numpy array for compatibility with ProSRS algorithm.
    def object_func(x):
        y = prob_def.f(x)
        y = np.array(y)
        return y
          
    class optim_prob:
        
        def __init__(self):
            self.prob_name = prob_def.name
            self.domain = prob_def.bounds
            self.x_var = prob_def.x_var
            self.y_var = prob_def.y_var
            self.dim = len(self.domain) # dimension of optimization problem
            assert(self.dim == len(self.x_var)), 'inconsistent dimension for x_var and domain'
            self.object_func = object_func # objective function (corrupted with noise)
        
    prob = optim_prob()
    
    return prob
        

################### Problem Definitions ##################


class test_func:
    """
    A simple test function corrupted by random noise.
    """
   
    def __init__(self):
        
        self.name = 'sphere' # problem name
        self.x_var = ['x1','x2'] # x variable names
        self.y_var = 'y' # y variable name
        self.bounds = [(-3.,3.),(-3.,3.)] # problem domain (bound for each variable in self.var)
    
    def f(self,x):
        """
        Optimization objective function.
        
        Input:
            x: 1d array, input to objective function.
        Output:
            y: float, objective value at x.
        """
        x1, x2 = x
        noise = np.random.normal(scale=0.02) # random noise
        y = x1**2+x2**2 # true function
        y += noise
        
        return y

## TODO: define more problems below ...
# 
#class PROBLEM:
#    """
#    Another problem ...
#    """
#   
#    def __init__(self):
#        
#        self.name =  # problem name
#        self.x_var = # x variable names
#        self.y_var = # y variable name
#        self.bounds =  # problem domain (bound for each variable in self.var)
#    
#    def f(self,x):
#        """
#        Optimization objective function.
#        
#        Input:
#            x: 1d array, input to objective function.
#        Output:
#            y: float, objective value at x.
#        """
#        
#        # BLA BLA ...
#        
#        return y


################### End of Problem Definitions ##################
