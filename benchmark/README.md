# Benchmark ProSRS performance
In this section, we show the performance of our ProSRS algorithm on 12 standard optimization benchmark problems (see [``benchmark_func.py``](benchmark_func.py) for function definitions). We compare ProSRS to three state-of-the-art parallel Bayesian optimization algorithms:

- GP-EI-MCMC (spearmint): https://github.com/JasperSnoek/spearmint
- GP-LP-LCB and GP-LP-EI: https://github.com/SheffieldML/GPyOpt with acquisition functions LCB and EI, respectively.

Here each benchmark function is contaminated by Gaussian noise. The goal is to minimize the underlying (unknown) objective function using as few iterations as possible. For a fair comparison, all the algorithms were started with the same initial design of experiments. The number of parallel cores was 12 (i.e., 12 function evaluations were performed in parallel per iteration).

## Optimization performance

[benchmark_opt_curve]

The figure above shows the optimization progress versus the number of iterations. The objective function on the y axis is the evaluation of the underlying true expectation function (not the noisy function) at the proposed point in each iteration. The error bar is the standard deviation of 20 independent runs. The last numeric figure in the function name indicates the dimension of the problem (e.g., PowerSum4 is scalar function on R^4).

[simple regret table]

The table above summarizes the optimization performances in terms of the simple regrets. Here the simple regret is defined as the difference between the minimum true sample evaluation among all the iterations and the global minimum (i.e., the distance between the lowest point of an optimization curve and the global minimum).

As we can see from the optimization curves and the summary table above, our ProSRS algorithm performs the best on almost all of the problems. In particular, ProSRS is significantly better on high-dimensional functions such as Ackley and Levy, as well as highly-complex functions such as Dropwave and Schaffer.

## Computational time of algorithms

[benchmark_wall_time]

The figure above shows the cost of different optimization algorithms. The "wall time" here refers to the actual time that was consumed by an algorithm in each iteration, and does not include the time of parallel function evaluations. Since the first iteration was design of experiments, in which no optimization was run, the time measurement effectively started from the second iteration (this is why the x label in this figure is "Optimization iteration" instead of "Iteration" in the last figure, and it goes one less in total). The time was benchmarked on [Blue Waters](https://bluewaters.ncsa.illinois.edu) XE compute nodes. The error bar shows the standard deviation of 20 independent runs.

We can see that our ProSRS method is about 1 to 4 orders of magnitude cheaper than the other algorithms. The fact that ProSRS is significantly cheaper means that it is suitable for a wider range of optimization problems, not just very expensive ones. 

# Analyzing ProSRS performance

In this section, we give some insight into why our ProSRS algorithm performs better than the Bayesian optimization algorithms compared above. We start the analysis with a numerical experiment that studies the modeling capability of RBF (as used in ProSRS) and GP models (as used in the Bayesian optimization methods), and conclude with a discussion.

## Experimental method

- We investigated RBF and GP regression on 12 optimization benchmark functions (the same functions as those in the optimization curves above), varying the number ``n`` of training data points from 10 to 100.
- For each test function and every ``n``, we first randomly sampled ``n`` points (X<sub>1</sub>, X<sub>2</sub>, ..., X<sub>n</sub>) over the function domain using Latin hypercube sampling, and then evaluated these ``n`` samples to get noisy responses (Y<sub>1</sub>, Y<sub>2</sub>, ..., Y<sub>n</sub>).
- Given the data (X<sub>1</sub>, Y<sub>1</sub>), (X<sub>2</sub>, Y<sub>2</sub>), ... , (X<sub>n</sub>, Y<sub>n</sub>), we trained 4 models: a RBF model using the cross validation procedure developed in the ProSRS algorithm, and 3 GP models with commonly used GP kernels: Matern1.5, Matern2.5 and RBF. To ensure a fair comparison, all the models were trained with the same training data.
- We used the Python ``scikit-learn`` package for the implementations of GP regression (http://scikit-learn.org/stable/modules/gaussian_process.html). The maximum likelihood problem in GP regression may have multiple local optima. As a result, it is advised to [repeat the optimizer in the GP regression several times](http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html). In this study, we set the number of restarts for the optimizer to be 10.
- We evaluated each regression model by measuring the relative error in terms of the L<sub>2</sub> norm of the difference between a model ``g`` and the underlying true function ``E[f]`` over the function domain.
- We repeated the forementioned procedure for 10 times, and reported the mean and the standard deviation of the measured relative errors.
- For more details of this numerical study, please refer to the source code file [``surrogate_compare.py``](surrogate_compare.py) in this directory.

## Results

[surrogate_performance]

We can see from the figure above that cross-validated RBF regression (as used in our ProSRS method) generally produces a better model than those from GP regression (as used in the Bayesian optimization methods). Specifically, the RBF model from ProSRS is significantly better for the test functions Griewank, Levy, Goldstein and PowerSum, and is on par with GP models for Schaffer, Dropwave and Hartmann.

## Discussion

There are two conclusions that we can draw from the experiment above:

1. The ProSRS RBF models seem to be able to better capture the objective functions than GP regression models. One possible explanation for this is that the ProSRS RBF regression uses a cross validation procedure so that the best model is selected directly according to the data, whereas GP regression builds models relying on the prior distributional assumptions about the data (i.e., Gaussian process with some kernel). Therefore, in a way the ProSRS regression procedure makes fewer assumptions about the data and is more "data-driven" than GP. Since the quality of a surrogate has a direct impact on how well the proposed points exploit the objective function, we believe that the superiority of the RBF models plays an important part in the success of our ProSRS algorithm over those Bayesian optimization algorithms.

2. For those test functions where ProSRS RBF and GP have similar modeling performances (i.e., Schaffer, Dropwave and Hartmann), the optimization performance of ProSRS (using RBF) is nonetheless generally better than Bayesian optimization (using the GP models), as we see from the optimization curves in the first figure in this document. This suggests that with surrogate modeling performance being equal, the ProSRS sample selection strategy (i.e., SRS and zoom strategy) may still have an edge over the probablity-based selection criterion (e.g., EI-MCMC) of Bayesian optimization. Quantitatively, for the three test functions in consideration here, on average the sampling selection method of ProSRS results in 71.4%, 71.7% and 69.4% improvement compared to GP-LP-LCB, GP-LP-EI and GP-EI-MCMC, respectively, in terms of the simple regret.
