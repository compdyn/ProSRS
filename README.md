# ProSRS algorithm
Progressive Stochastic Response Surface (ProSRS) is a parallel surrogate-based optimization algorithm for optimizing noisy expensive functions. This algorithm utilizes a radial basis function (RBF) as the surrogate, and adopts stochastic response surface (SRS) framework to balance exploitation and exploration. Compared to the [original parallel SRS work](https://pubsonline.informs.org/doi/10.1287/ijoc.1090.0325), the novelties of this algorithm include
- Introducing a new tree-based technique, known as the "zoom strategy", for efficiency improvement.
- Extending the original work to the noisy setting (i.e., an objective function corrupted with random noise) through the development of a radial basis regression procedure. 
- Introducing weighting to the regression for exploitation enhancement.
- Implementing a new SRS that combines the two types of candidate points that were originally proposed in the SRS work.

ProSRS algorithm is configured in a master-worker structure, where in each optimization iteration, the algorithm (master) constructs a RBF surrogate using the available evaluations, then proposes new points based on the constructed RBF, and finally distributes the tasks of evaluating these points to parallel processes (workers).

Compared to the popular Bayesian optimization algorithms, ProSRS is able to achieve faster convergence on some difficult benchmark problems, and is orders of magnitude cheaper to run. Moreover, ProSRS enjoys asymptotic convergence gaurantees. The common applications of this algorithm include efficient hyperparamter tuning of machine learning models and characterizing expensive simulation models.

# Installation

Python packages:
  
  - `numpy`: http://www.numpy.org
  - `scipy`: https://www.scipy.org
  - `matplotlib`: https://matplotlib.org
  - `sklearn`: https://scikit-learn.org/stable/
  - `pyDOE`: https://pythonhosted.org/pyDOE/
  - `pathos`: https://pypi.org/project/pathos/

**Note:** ProSRS has been tested against both Python2 and Python3. Users are welcome to install packages with either Python version.

# Getting started

After having installed the required Python packages, users are ready to use ProSRS algorithm for solving optimization problems. The easiest way of getting started is to read the tutorials in the [`examples`](examples) directory, where different usages, from the basic level to the advanced level, are demonstrated through code examples. Of course, users are also encouraged to check out the source codes of the algorithm in the [`prosrs`](prosrs) directory.
