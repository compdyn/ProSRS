# Files in this directory

  - [`benchmark`](benchmark): benchmarks ProSRS against Bayesian optimization algorithms.
   
  - [`src/problem.py`](src/problem.py): defines optimization problems.

  - [`src/ProSRS.py`](src/ProSRS.py): implements ProSRS algorithm. 

  - [`src/run.py`](src/run.py): runs ProSRS algorithm to solve optimization problems.
  
# Installation
  
  Python packages:
  
  - `sklearn`: http://scikit-learn.org/stable/install.html
  - `pyDOE`: https://pythonhosted.org/pyDOE/
  - `pathos`: https://pypi.python.org/pypi/pathos/0.2.0
  - `mpi4py`: http://mpi4py.scipy.org (optional)

  For a quick sanity check of your installation, cd into `src` directory and type the following on the command line: 
  
  > python run.py
  
  and you should get

  > best point found:
  
  > x1 = -0.0358601, x2 = -0.0132618
  
  > (noisy) function value corresponding to the best point:
  
  > y = -0.0424788

  **Note:** ProSRS algorithm can be run with either Python2 or Python3 so you may install packages for either Python version.
  
# Use ProSRS to solve your own optimization problem

  1. Write your optimization problem in `src/problem.py`.

  1. In `src/run.py`,

      1. Import your problem at the top.
      
      1. Set parameters at the beginning of the `main()` function.
      
      1. Edit the line `prob = gen_problem()` to load your problem.

# Guidelines of running ProSRS

Below we give some general guidelines for setting parameters in `src/run.py`:

  - Generally speaking, set large `n_iter` for difficult problems.
  
  - Set `parallel_node` based on how you would like to evaluate your expensive tasks in parallel. ProSRS supports two parallel computing modes: parallelism across multiple nodes in a cluster environment (`parallel_node = True`) and parallelism of multiple cores within a node (`parallel_node = False`). You need to install `mpi4py` package if you turn on `parallel_node`.

  - Setting `resume_iter` to an integer allows you to resume from last run. In other words, you do not have to complete the optimization in one go:
     
     For example, to complete a run of `n_iter = 20`. You could first run the code with `n_iter = 10` and `resume_iter = None`. After this, you can set `n_iter = 20` and `resume_iter = 10` to finish off the remaining 10 iterations.

# Interpretation of results after running ProSRS

  - All the results are saved to the output directory (`outdir` in `src/run.py`).

  - Explanation of output files:

    - `optim_result_{PROBLEM_NAME}.npz`: save all the states and the results of ProSRS algorithm. This file is generated/updated in each iteration and is required for resuming.

    - `optim_result_{PROBLEM_NAME}.pkl`: save the random state of ProSRS algorithm. This file is generated/updated in each iteration and is required for resuming.

    - `samp_{PROBLEM_NAME}_t{ITER_INDEX}.nc`: save all the function evaluations of a specific iteration. Here `{ITER_INDEX}` is the iteration index starting from 1. This file is generated once all the evaluations of that iteration have been completed.

    - `samp_{PROBLEM_NAME}_t{ITER_INDEX}_p{PROC}.nc`: save the function evaluation of a specific iteration and a specific process. Here `{PROC}` is the process index starting from 1. The value of `{PROC}` should be one of {1,2,...,N}, where N is `n_proc` in `src/run.py`. This file is generated as soon as the function evaluation is completed. 

    - `std_output_log_{PROBLEM_NAME}.txt`: log the standard output. This file is generated when `verbose = True` in `src/run.py`.

  - ProSRS returns the point with the lowest noisy function value among all the evaluations. In general, this point may not be the actual best point (i.e., the point with the lowest *expected* function value). Post Monte Carlo sampling can be used for selecting the actual best one among the evaluations.


