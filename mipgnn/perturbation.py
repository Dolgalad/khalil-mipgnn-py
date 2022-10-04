import numpy as np
from cplex import Cplex
from progress.bar import Bar

from mipgnn.solver import MIPSolver
import matplotlib.pyplot as plt

if __name__=="__main__":
    timelimit = 30

    mip_path = "/home/aschulz/.cache/mipgnn/datasets/set_cover_n1000_nv1000_nc1000_d0.1/train/1.mps"

    stds = [0.01, 0.1, 1, 10, 100]

    final_objective_values = {std:[] for std in stds}
    times = {std:[] for std in stds}
    gaps = {std:[] for std in stds}

    for std in stds:
        # solve perturbated problems
        bar = Bar(f"Progress {std}: ", max=100)
        for i in range(100):
            cpx = Cplex()
            cpx.parameters.timelimit.set(timelimit)
            cpx.set_log_stream(None)
            cpx.set_warning_stream(None)
            cpx.set_results_stream(None)


            cpx.read(mip_path)

            c = np.array(cpx.objective.get_linear())
            #print(f"c.shape: {c.shape}")
            # perturbation
            nc = c + np.random.normal(size=c.shape) * std
            #print(f"nc.shape: {nc.shape}")

            cpx.objective.set_linear([(i,nv) for i,nv in zip(range(c.size),nc)])
            # solve
            t0 = cpx.get_time()
            cpx.solve()
            t0 = cpx.get_time() - t0
            # best objective value
            final_objective_values[std].append(cpx.solution.get_objective_value())
            gaps[std].append(cpx.solution.MIP.get_mip_relative_gap())
            times[std].append(t0)
            bar.next()
        bar.finish()

    plt.figure()
    plt.subplot(3,1,1)
    plt.title("Final objective values")
    plt.boxplot(list(final_objective_values.values()), labels=stds)
    plt.subplot(3,1,2)
    plt.title("Relative gap")
    plt.boxplot(list(gaps.values()), labels=stds)
    plt.subplot(3,1,3)
    plt.title("Time")
    plt.boxplot(list(times.values()), labels=stds)


    plt.show()

