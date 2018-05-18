# Files in this directory

  - `src/problem.py`: Define optimization problems.

  - `src/ProSRS.py`: Implement ProSRS algorithm. 

  - `src/run.py`: Run ProSRS algorithm to solve optimization problems.
   
# Installation
  
  Required Python packages:
  
  - `sklearn`: http://scikit-learn.org/stable/install.html
  - `pyDOE`: https://pythonhosted.org/pyDOE/
  - `pathos`: https://pypi.python.org/pypi/pathos/0.2.0

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

  - To take full advantage of ProSRS algorithm, generally set `n_proc` to be the number of cores of a machine.

  - Setting `resume_iter` to an integer allows you to resume from last run. In other words, you do not have to complete the optimization in one go:
     
     For example, to complete a run of `n_iter = 20`. You could first run the code with `n_iter = 10` and `resume_iter = None`. After this, you can set `n_iter = 20` and `resume_iter = 10` to finish off the remaining 10 iterations.

  - Set `serial_mode = False` to enable paralle evaluations.

# Interpretation of results after running ProSRS

  - All the results are saved to the output directory (`outdir` in `src/run.py`).

  - Explanation of output files:

    - `optim_result_{PROBLEM_NAME}.npz`: save all the states and the results of ProSRS algorithm. This file is generated/updated in each iteration and is required for resuming.

    - `optim_result_{PROBLEM_NAME}.pkl`: save the random state of ProSRS algorithm. This file is generated/updated in each iteration and is required for resuming.

    - `samp_{PROBLEM_NAME}_t{ITER_INDEX}.nc`: save all the function evaluations of a specific iteration. Here `{ITER_INDEX}` is the iteration index starting from 1. This file is generated once all the evaluations of that iteration have been completed.

    - `samp_{PROBLEM_NAME}_t{ITER_INDEX}_p{PROC}.nc`: save the function evaluation of a specific iteration and a specific process. Here `{PROC}` is the process index starting from 1. The value of `{PROC}` should be one of {1,2,...,N}, where N is `n_proc` in `src/run.py`. This file is generated as soon as the function evaluation is completed. 

    - `std_output_log_{PROBLEM_NAME}.txt`: log the standard output. This file is generated when `verbose = True` in `src/run.py`.

  - ProSRS returns the point with the lowest noisy function value among all the evaluations. In general, this point may not be the actual best point (i.e., the point with the lowest *expected* function value). Post Monte Carlo sampling can be used for selecting the actual best one among the evaluations.


