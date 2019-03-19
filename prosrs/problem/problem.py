"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Define an optimization problem.
"""
import sys
import numpy as np
from ..utility.functions import eval_func
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D # needed for ``fig.add_subplot(111, projection='3d')``
except:
    pass


class Problem:
    """
    An optimization problem class that works with ProSRS algorithm.
    """
    def __init__(self, domain, func=None, x_var=None, y_var='y', name='demo', true_func=None, 
                 min_loc=None, min_val=None, noise_type=None, sd=None):
        """
        Constructor. 
        
        Args:
            
            domain (list of tuples): Optimization domain.
                For example, `domain` = [(0, 1), (0, 2)] means optimizing a
                2D function ``f(x)`` on the domain: 
                ``0 <= x[0] <= 1`` and ``0 <= x[1] <= 2``.
            
            func (callable or None, optional): (Noisy) function to be optimized over, for which the 
                input is 1d array, the output is a scalar. 
                For example, ``func = lambda x: x[0]**2+x[1]**2+np.random.normal()``. 
                If None, then the optimization function is undefined.
                
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
            
            min_val (float or None, optional): Global minimum of function 
                `true_func` on the `domain`. If None, then global minimum is unknown.
            
            noise_type (str or None, optional): Type of random noises (e.g., 'Gaussian').
                If None, then the noise type is unknown.
            
            sd (float or None, optional): Standard deviation of random noise in `func`.
                If None, then the standard deviation is unknown.
        
        Raises:
            
            ValueError: If dimension of `domain` and dimension of `x_var` do not match.
        
        """
        self.domain = domain
        self.f = func
        self.dim = len(domain) # dimension of optimization problem
        self.x_var = ['x%d' % i for i in range(1, self.dim+1)] if x_var is None else x_var
        self.y_var = y_var
        self.name = name
        self.F = true_func
        self.min_loc = min_loc
        self.min_val = min_val
        self.noise_type = noise_type
        self.sd = sd
        self.domain_lb, self.domain_ub = zip(*self.domain) # domain lower bounds and domain upper bounds
        self.domain_lb, self.domain_ub = np.array(self.domain_lb), np.array(self.domain_ub) # convert to 1d array
        # sanity check
        assert(callable(self.f) or self.f is None)
        if self.dim != len(self.x_var):
            raise ValueError('Inconsistent dimension for x_var and domain.')
        if self.min_loc is not None:
            assert(self.min_loc.shape[1] == self.dim), 'Wrong shape for min_loc.'
        if self.sd is not None:
            assert(self.sd >= 0)
        assert(np.all(self.domain_lb < self.domain_ub))
    
    
    def __str__(self):
        """
        Display optimization problem info.
        
        Use ``print()`` method to call this function.
        """
        line = 'Optimization problem:\n'
        line += '- Name: %s\n' % self.name
        line += '- Dimension: %d\n' % self.dim
        line += '- Domain: %s\n' % ('{'+', '.join(["'%s': %s" % (x, str(tuple(v))) for x, v in zip(self.x_var, self.domain)])+'}')
        if self.f is None:
            line += "- Optimization function: not provided\n"
        line += "- Response variable: '%s'\n" % str(self.y_var)
        if self.sd is not None and self.noise_type is not None:
            line += '- Random noise: %s with standard deviation of %g\n' % (self.noise_type, self.sd)
        if self.min_val is not None:
            line += '- Global minimum: %g\n' % self.min_val
        else:
            line += '- Global minimum: unknown\n'
        if self.min_loc is not None:
            line += '- Global minimum locations: %s\n' % ', '.join(["%s = %s" % (str(tuple(self.x_var)), str(tuple(v)))\
                                                                   for v in self.min_loc])
        else:
            line += '- Global minimum locations: unknown\n'
        line = line.rstrip() # remove \n at the end
        return line
    
        
    def visualize(self, true_func=False, n_samples=None, plot='contour', 
                  contour_levels=100, min_marker_size=10, n_proc=1, fig_path=None):
        """
        Visualize optimization function (only if dimension of function is <= 2).
        
        Args:
            
            true_func (bool, optional): Whether we plot the underlying expectation 
                function `true_func`. If False, then plot (noisy) function `func`. 
            
            n_samples (int or None, optional): Number of samples for generating
                plots. If None, `n_samples` assumes default values: 20 for 1D 
                problems; 100 for 2D problems.
            
            plot (str, optional): Plot style for 2D problems. Must be one 
                of ['contour', 'surface'].
                
            contour_levels (int, optional): Number of contour lines in the contour plot.
                Useful only when `plot` = 'contour'. For more explanation, see
                `Matplotlib Document <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html>`.
            
            min_marker_size (int, optional): Marker size for global minimum locations.
            
            n_proc (int, optional): Number of parallel processes for parallel function 
                evaluations. If = 1, then optimization function is evaluated in serial.
            
            fig_path (str, optional): Path of figure (i.e., where to save the plot). 
                If None, then we do not save visualization.
        
        Raises:
            
            ValueError: If dimension is 2 and `plot` is not one of ['contour', 'surface'] or 
                        if dimension of function is greater than 2.           
        """
        if self.dim > 2:
            raise ValueError('Visualization for problems of dimension greater than 2 is not implemented.')
        else:
            plot_f = self.F if true_func else self.f # function to be plotted
            # sanity check
            assert(callable(plot_f)), "The function to be plotted is not callable. Please check 'Problem' defininition."
            try:    
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
                    
                    if plot == 'contour':
                            
                        plt.contourf(x1, x2, Y, contour_levels)
                        if self.min_loc is not None:    
                            plt.plot(self.min_loc[:,0], self.min_loc[:,1], 'w.', markersize=min_marker_size)
                        plt.colorbar()
                        plt.xlabel(self.x_var[0])
                        plt.ylabel(self.x_var[1])
                        plt.title('Contour plot of optimization function (problem: %s)' % self.name)
                    
                    elif plot == 'surface':
    
                        ax = fig.add_subplot(111, projection='3d')
                        ax.plot_surface(x1, x2, Y, rstride=1, cstride=1, cmap=cm.rainbow,
                                        linewidth=0, antialiased=False)
                        ax.axis('off')
                        ax.set_xlabel(self.x_var[0])
                        ax.set_ylabel(self.x_var[1])
                        ax.set_title('Surface plot of optimization function (problem: %s)' % self.name)
                        
                    else:
                        raise ValueError("Invalid plot value. Must be one of ['contour', 'surface']")
                    
                plt.show()
                if fig_path is not None:
                    fig.savefig(fig_path)
            except Exception as e:
                sys.exit('Error! Unable to generate plots for visualization: %s. This may be due to unsuccessful installation of matplotlib package. For more, please see the installation note in the README file at `https://github.com/compdyn/ProSRS`.' % str(e))

