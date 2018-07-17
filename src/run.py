"""
Copyright (C) 2018 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Run ProSRS algorithm to solve optimization problems.
"""
import os
from ProSRS import run
from problem import gen_problem
from problem import test_func # TODO: import the problem you want to run (provide the name of the problem class)

def main():
    
    ##################### Parameters #########################
    
    # set parameters
    n_iter = 30 # total number of iterations
    parallel_node = False # whether we do parallel optimization across multiple nodes. If False, then we use multiple cores of one node
    n_proc = 12 # number of evaluations per iteration, typically = number of cores of a node (parallel_node = False) or number of parallel nodes (parallel_node = True)
    n_core_node = 12 # number of cores in one node 
    resume_iter = None # iteration to resume from (= 'n_iter' in last run). If None, then we run from scratch.
    seed = 1 # random seed for reproducibility
    save_samp = True # whether to save evaluations in each iteration
    verbose = False # If True, progress of every iteration is displayed at terminal as well as saved to file "{outdir}/std_out_log_{prob_name}.txt"
    outdir = '../result' # output directory where results will be saved
    
    ##################### End of Parameters #########################
    
    # sanity check
    if resume_iter is not None:
        assert(0<=resume_iter<=n_iter)
    # set MPI communicator
    if parallel_node:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD  
    else:
        comm = None
    # make output directory if it does not exist    
    if comm is None or comm.rank == 0:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
    # TODO: generate an optimization problem that you would like to run (change the following line as needed)
    prob = gen_problem(test_func()) # replace 'test_func' with the name of the imported problem class
    
    # run optimization algorithm
    if comm is None or comm.rank == 0:
        best_loc, best_val = run(prob, n_iter, n_proc, n_core_node, comm, outdir, seed=seed, save_samp=save_samp,
                                 verbose=verbose, resume_iter=resume_iter)
        
        print('\nbest point found:')
        print(', '.join(['%s = %g' % (x,v) for x,v in zip(prob.x_var, best_loc)]))
        print('(noisy) function value corresponding to the best point:')
        print('%s = %g' % (prob.y_var, best_val))
        
    else:
        # i.e., comm.rank > 0
        run(prob, n_iter, n_proc, n_core_node, comm, outdir, seed=seed, save_samp=save_samp,
            verbose=verbose, resume_iter=resume_iter)

if __name__ == '__main__':
    
    main()