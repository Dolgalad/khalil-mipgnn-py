"""Base solver implemented using CPLEX
"""
import tempfile
import numpy as np

from cplex import Cplex
from cplex.callbacks import MIPInfoCallback

from mipgnn.instance import MIPInstance

history_fields = ["time", "best_objective_value", "cutoff", "incumbent_objective_value", "incumbent_linear_slacks",
        "incumbent_values", "mip_relative_gap", "num_iterations", "num_nodes", "num_remaining_nodes", "has_incumbent"]

class SolverInfoCallback(MIPInfoCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {f:[] for f in history_fields}
    def __call__(self, *args, **kwargs):
        self.get_data()
    def get_data(self):
        self.history["time"].append(self.get_time())
        self.history["best_objective_value"].append(self.get_best_objective_value())
        self.history["cutoff"].append(self.get_cutoff())
        self.history["incumbent_objective_value"].append(self.get_incumbent_objective_value())
        self.history["incumbent_linear_slacks"].append(self.get_incumbent_linear_slacks())
        self.history["incumbent_values"].append(self.get_incumbent_values())
        self.history["mip_relative_gap"].append(self.get_MIP_relative_gap())
        self.history["num_iterations"].append(self.get_num_iterations())
        self.history["num_nodes"].append(self.get_num_nodes())
        self.history["num_remaining_nodes"].append(self.get_num_remaining_nodes())
        self.history["has_incumbent"].append(self.has_incumbent())


class MIPSolver(Cplex):
    def __init__(self, **kwargs):
        """Initialize the solver
        """
        super().__init__()
        self.set_cplex_parameters(**kwargs)
        self.info_callback = self.register_callback(SolverInfoCallback)
        self.solver_parameters = kwargs
        self.history = {}

    def set_cplex_parameters(self, **params):
        self.parameters.timelimit.set(params.get("timelimit",60 ))
        self.parameters.emphasis.mip.set(params.get("cpx_emphasis", 1))
        self.parameters.mip.display.set(3)
        self.parameters.threads.set(params.get("cpx_threads", 1))
        self.parameters.workmem.set(params.get("memlimit", 1024))
        self.parameters.mip.limits.treememory.set(20000)
        self.parameters.mip.strategy.file.set(2)
        self.parameters.workdir.set(params.get("cpx_tmp", "./cpx_tmp"))
        self.solver_parameters = params

    def quiet(self):
        self.set_log_stream(None)
        self.set_warning_stream(None)
        self.set_results_stream(None)


    def optimize(self, instance: MIPInstance = None, filename: str = None):
        """Solve a MIP problem
        """
        if filename:
            self.read(filename)
            t0 = self.get_time()
            self.solve()
            t0 = self.get_time() - t0
            self.history["t_solve"] = t0
        if instance:
            # dump the instance to a temporary file
            with tempfile.TemporaryDirectory() as tmpdir:
                ipath = os.path.join(tmpdir, "instance.mps")
                MIPInstance.dump(instance, tmpf)
                return self.optimize(filename = tmpf.name)
        # check that the info callback best objective value is equal to the solutions objective value
        if len(self.info_callback.history["best_objective_value"]):
            best_obv = self.info_callback.history["best_objective_value"][-1]
            #print("best obj", best_obv)
            ##print("incumbent objv : ", self.info_callback.get_incumbent_objective_value())
            #print("blabal ", self.info_callback.history["incumbent_objective_value"][-1])
            #print("sol obj", np.dot(self.solution.get_values(), self.objective.get_linear()))
            #print("sollo ", self.solution.get_objective_value())
            #print(self.objective.sense[self.objective.get_sense()])

        self.history["solution.is_primal_feasible"] = self.solution.is_primal_feasible()
        return self.solution.get_values(), {**self.history, **self.info_callback.history}


if __name__=="__main__":
    # initialize the solver
    solver = MIPSolver()
    #solver.quiet()
    #info_callback = solver.register_callback(SolverInfoCallback)


    # path to MIP instance
    mip_path = "/home/aschulz/.cache/mipgnn/datasets/set_cover_n100_nv100_nc100_d0.1/1.mps"

    # load a problem
    mip = MIPInstance.load("/home/aschulz/.cache/mipgnn/datasets/set_cover_n100_nv100_nc100_d0.1/1.mps")

    # solve the problem
    r = solver.optimize(filename=mip_path)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(solver.info_callback.history["time"], solver.info_callback.history["best_objective_value"] )
    plt.plot(solver.info_callback.history["time"], solver.info_callback.history["incumbent_objective_value"] )
    plt.subplot(3,1,2)
    plt.plot(solver.info_callback.history["time"], solver.info_callback.history["mip_relative_gap"])
    plt.subplot(3,1,3)
    plt.plot(solver.info_callback.history["time"], solver.info_callback.history["num_nodes"])

    plt.show()
