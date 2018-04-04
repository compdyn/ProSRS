# Files in this directory

  - `src/problem.py`: Definitions of optimization problems.

  - `src/ProSRS.py`: A parallel optimization algorithm for expensive noisy black-box functions. 

  - `src/run.py`: A main code that runs ProSRS algorithm to solve optimization problems.
  
  - `job.pbs`: A sample job script.
   
# Installation
  
  Required Python packages:
  
  - `sklearn`: http://scikit-learn.org/stable/install.html
  - `pyDOE`: https://pythonhosted.org/pyDOE/
  - `pathos`: https://pypi.python.org/pypi/pathos/0.2.0

  For a quick sanity check of your installation, please cd into `src` directory and type the following on the command line: 
  
  > python run.py
  
  and you should get the following result:

  > best point found:
  
  > x1 = -0.00783916, x2 = 0.0620596
  
  > (noisy) function value corresponding to the best point:
  
  > y = -0.269442

  **Note:** ProSRS algorithm can be run with either Python2 or Python3 so you may install packages for either Python version.
  
# Use ProSRS to solve your own optimization problem

  1. Write your optimization problem in `src/problem.py`.

  1. In `src/run.py`,

      1. Import your problem at the top.
      
      1. Set parameters at the beginning of the `main()` function.
      
      1. Edit the line `prob = gen_problem()` to load your problem.
 
  1. If run locally, then simply cd into `src` directory and type `python run.py` on the command line.

  1. If run on a cluster, write a job script that runs `src/run.py`.


# Guidelines of running ProSRS

Below we give some general guidelines for setting parameters in `src/run.py`:

  - Generally speaking, set larger `opt_iter` for more difficult problems (e.g., problems with higher dimension). Default value of `opt_iter` is 30, which should be adequate for most problems.

  - When running on a cluster (i.e., `run_mode = 'cluster'`), `n_proc` should be set equal to the maximum number of cores available on a compute node. In this way, you enjoy the full benefits of ProSRS algorithm.

  - Setting `resume_opt_iter` to an integer allows you to resume from last run. In other words, you do not have to complete the optimization in one go. 
     
     For example, to complete a run of `opt_iter = 20`. You could first run the code with `opt_iter = 10` and `resume_opt_iter = None`. After this, you can set `opt_iter = 20` and `resume_opt_iter = 10` to finish off the remaining 10 iterations.

     Sometimes the wall time of a cluster job may not be set properly so that the job is terminated before it's finished. In this case, you can continue optimization by first reading the value of `opt_iter` variable in the file `{outdir}/optim_result_{PROBLEM_NAME}.npz` and then setting `resume_opt_iter` equal to this value. Here `{PROBLEM_NAME}` is your optimization problem name (i.e., the value of `self.name` in your problem class).

  - Because the function is expensive to evaluate, most of the time you should run optimization on a cluster. Local runs (i.e., `run_mode = 'local'`) will evaluate expensive functions in serial. As a result, they are mainly intended for short non-intensive runs (e.g., runs with small `opt_iter` values). You can use local runs as preliminaries for the production runs on the cluster.

# Interpretation of results after running ProSRS

  - All the results will be saved to the output directory (i.e., `outdir` in `src/run.py`).

  - Explanation of different types of output files:

    - `optim_result_{PROBLEM_NAME}.npz`: save all the states and the results of ProSRS algorithm. This file will be generated/updated for each iteration and is needed for resuming.

    - `optim_result_{PROBLEM_NAME}.pkl`: save the random state of ProSRS algorithm. This file will be generated/updated for each iteration and is needed for resuming.

    - `samp_{PROBLEM_NAME}_t{ITER_INDEX}.nc`: save all the function evaluations of a specific iteration. Here `{ITER_INDEX}` is the iteration index starting from 1. This file will be generated once all the evaluations in that iteration have been completed.

    - `samp_{PROBLEM_NAME}_t{ITER_INDEX}_p{PROC}.nc`: save the function evaluation for a specific iteration and a specific process. Here `{PROC}` is the process index starting from 1. The value of `{PROC}` should be one of {1,2,...,N}, where N is `n_proc` in `src/run.py`. This file will be generated as soon as the function evaluation for that iteration and that process is completed. 

    - `std_output_log_{PROBLEM_NAME}.txt`: log the standard output of ProSRS. This file is essentially a copy of what is displayed in the terminal and will be generated only when `verbose = True` in `src/run.py`.

  - The result shown in the terminal after running `src/run.py` is the point with the smallest noisy function value among all the evaluated points. However, this point in general may not be the actual best point (i.e., the point with the smallest *expected* function value). Post Monte Carlo sampling can be used to select the actual best one. A simple post-sampling procedure can be as follows:
     
    1. Obtain all the points and their noisy evaluations by reading `x` and `y` variables from the netcdf files `samp_{PROBLEM_NAME}_t1.nc`, `samp_{PROBLEM_NAME}_t2.nc` ...

    1. Choose, say the top 10 points with the lowest noisy function values as the candidates for the best point.
  
    1. Run Monte Carlo on the candidate points and choose the best point to be the one with the smallest mean value.


