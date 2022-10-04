import cplex


def cpx_disable_output(cpx):
    cpx.set_log_stream(None)
    # instance_cpx.set_error_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_results_stream(None)

def get_silent_cpx(instance_path=None):
    cpx = cplex.Cplex(instance_path)
    cpx_disable_output(cpx)
    return cpx


