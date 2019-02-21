"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Define an optimization problem.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
from ..utility.functions import eval_func
from mpl_toolkits.mplot3d import Axes3D # needed for ``fig.add_subplot(111, projection='3d')``   


class Problem:
    """
    An optimization problem class that works with ProSRS algorithm.
    """
    def __init__(self, func, domain, x_var=None, y_var='y', name='demo', true_func=None, 
                 min_loc=None, min_true_func=None):
        """
        Constructor. 
        
        Args:
            
            func (callable): Function to be optimized over, for which ithe nput is 1d array,
                the output is a scalar. For example, ``func = lambda x: x[0]**2+x[1]**2``.
            
            domain (list of tuples): Optimization domain.
                For example, `domain` = [(0, 1), (0, 2)] means optimizing the
                2D function `func` of argument ``x`` on the domain: 
                ``0 <= x[0] <= 1`` and ``0 <= x[1] <= 2``.
                
            x_var (list of str or None, optional): Argument names of function `func`.
                If None, then `x_var` defaults to ['x1', 'x2', ...]
                such that length of `x_var` = length of `domain`.
            
            y_var (str, optional): Response variable name of function `func`.
            
            name (str, optional): Name of this optimization problem.
            
            true_func (callable or None, optional): Underlying true function of
                `func` (i.e., expectation of function `func`). If None,
                then the underlying true function is unknown.
            
            min_loc (2d array or None, optional): Global minimum locations of function 
                `true_func` on the `domain`. Each row defines one minimum point. 
                If None, then the minimum locations are unknown.
            
            min_true_func (float or None, optional): Global minimum of function 
                `true_func` on the `domain`. If None, then global minimum is unknown.
        
        Raises:
            
            ValueError: If dimension of `domain` and dimension of `x_var` do not match.
        
        """
        self.f = func
        self.domain = domain
        self.dim = len(domain) # dimension of optimization problem
        self.x_var = ['x%d' % i for i in range(1, self.dim+1)] if x_var is None else x_var
        self.y_var = y_var
        self.name = name
        self.F = true_func
        self.min_loc = min_loc
        self.min_true_func = min_true_func
        self.domain_lb, self.domain_ub = zip(*self.domain) # domain lower bounds and domain upper bounds
        self.domain_lb, self.domain_ub = np.array(self.domain_lb), np.array(self.domain_ub) # convert to 1d array
        # sanity check
        if self.dim != len(self.x_var):
            raise ValueError('Inconsistent dimension for x_var and domain.')
        if self.min_loc is not None:
            assert(self.min_loc.shape[1] == self.dim), 'Wrong shape for min_loc.'
        assert(np.all(self.domain_lb < self.domain_ub))
    
    def __str__(self):
        """
        Print out optimization problem info.
        
        Use ``print()`` method to call this function.
        """
        line = 'Optimization problem (dim = %d): %s\n' % (self.dim, self.name)
        var_domain = {v:d for v,d in zip(self.x_var, self.domain)}
        line += '- Domain: %s\n' % str(var_domain)
        line += '- Y variable: %s\n' % str(self.y_var)
        if self.min_true_func is not None:
            line += '- Global minimum: %g\n' % self.min_true_func
        if self.min_loc is not None:
            line += '- Global minimum locations:\n %s\n' % str(self.min_loc)
        return line
            
    def visualize(self, true_func=False, n_samples=None, plot_2d='contour', 
                  contour_levels=100, min_marker_size=10, n_proc=1, fig_path=None):
        """
        Visualize optimization function (only if dimension of function is <= 2).
        
        Args:
            
            true_func (bool, optional): Whether we plot the underlying expectation 
                function `true_func`. If False, then plot (noisy) function `func`. 
            
            n_samples (int or None, optional): Number of samples for generating
                plots. If None, `n_samples` assumes default values: 20 for 1D 
                problems; 100 for 2D problems.
            
            plot_2d (str, optional): Plot style for 2D problems. Must be one 
                of ['contour', 'surface'].
                
            contour_levels (int, optional): Number of contour lines in the contour plot.
                Useful only when `plot_2d` = 'contour'. For more explanation, see
                `Matplotlib Document <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html>`.
            
            min_marker_size (int, optional): Marker size for global minimum locations.
            
            n_proc (int, optional): Number of parallel processes for parallel function 
                evaluations. If = 1, then optimization function is evaluated in serial.
            
            fig_path (str, optional): Path of figure (i.e., where to save the plot). 
                If None, then we do not save visualization.
        
        Raises:
            
            TypeError: If function to be plotted is not callable.
            ValueError: If dimension is 2 and `plot_2d` is not one of ['contour', 'surface'].
        
        Warnings:
            
            UserWarning: If dimension of function is greater than 2.
            
        """
        if self.dim > 2:
            warnings.warn('Visualization for problems of dimension greater than 2 is not implemented.')
        else:
            plot_f = self.F if true_func else self.f # function to be plotted
            # sanity check
            if not callable(plot_f):
                raise TypeError('Optimization function to be plotted is not callable.')
            
            fig = plt.figure()
            
            if self.dim == 1:
                
                n_samples = 20 if n_samples is None else n_samples
                X = np.linspace(self.domain[0][0], self.domain[0][1], n_samples)
                Y = eval_func(plot_f, X.reshape((-1, 1)), n_proc=n_proc)
                plt.plot(X, Y, 'b-')
                if self.min_loc is not None:
                    Ymin = [plot_f(x) for x in self.min_loc]
                    plt.plot(self.min_loc, Ymin, 'rx', markersize=min_marker_size)
                
            else: # i.e., self.dim == 2
                
                n_samp_per_dim = 10 if n_samples is None else int(n_samples**0.5) # number of samples per dimension
                n_samples = n_samp_per_dim**2
                # generate and evaluate samples
                x1 = np.linspace(self.domain[0][0], self.domain[0][1], n_samp_per_dim)
                x2 = np.linspace(self.domain[1][0], self.domain[1][1], n_samp_per_dim)
                x1, x2 = np.meshgrid(x1, x2)
                X = np.hstack((x1.reshape((n_samples, 1)), x2.reshape((n_samples, 1))))
                Y = eval_func(plot_f, X, n_proc=n_proc).reshape((n_samp_per_dim, n_samp_per_dim))
                
                if plot_2d == 'contour':
                        
                    plt.contourf(x1, x2, Y, contour_levels)
                    if self.min_loc is not None:    
                        plt.plot(self.min_loc[:,0], self.min_loc[:,1], 'w.', markersize=min_marker_size)
                    plt.colorbar()
                    plt.xlabel(self.x_var[0])
                    plt.ylabel(self.x_var[1])
                    plt.title('Contour plot of optimization function (problem: %s)' % self.name)
                
                elif plot_2d == 'surface':

                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(x1, x2, Y, rstride=1, cstride=1, cmap=cm.rainbow,
                                    linewidth=0, antialiased=False)
                    ax.axis('off')
                    ax.set_xlabel(self.x_var[0])
                    ax.set_ylabel(self.x_var[1])
                    ax.set_title('Surface plot of optimization function (problem: %s)' % self.name)
                    
                else:
                    raise ValueError("Invalid plot_2d value. Must be one of ['contour', 'surface']")
                
            plt.show()
            if fig_path is not None:
                fig.savefig(fig_path)

