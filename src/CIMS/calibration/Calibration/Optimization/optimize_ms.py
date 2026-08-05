
import Calibration.Optimization._objectiveFunctions as ObjectiveFunctions
import Calibration.Optimization._optimize_years_sequential as OptimizeYearsSequential

def optimize_ms_via_fics(model, 
                         nodeName, 
                         init_x = "zero",
                         logFile = "log_optimize_ms_via_fics.log"):

    opt_results = OptimizeYearsSequential.optimize_years_sequential(
            ObjectiveFunctions.make_objective_localNode,
            model,
            nodeName,
            init_x = init_x,
            logFile = logFile
    )
    return opt_results
