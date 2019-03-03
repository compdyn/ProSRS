"""
Copyright (C) 2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

"""
from .problem.problem import Problem
from .problem.benchmark import benchmark
from .optimizer.optimizer import Optimizer
from .version import __version__
# Try to import matplotlib packages. Issue a warning if any error occurs.
import warnings
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
except Exception as e:
    warnings.warn('Unable to import modules from matplotlib: %s\nAs a result, the visualization capabilities in the prosrs package may be impaired. For the workarounds and the implications of this, please see the installation note in the README file at `https://github.com/compdyn/ProSRS`.' % str(e))