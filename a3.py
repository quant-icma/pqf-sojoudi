import numpy as np


def mc_path_dependent(option_state, process, yield_curve, n_paths):
    """Computes the fair value of a path-dependent option using Monte-Carlo simulations.



    Parameters
    ----------
    option_state : OptionState
        The option payoff state variable recorder
    process : object with method update()
        The underlying's stochastic process evolver
    yield_curve : object with method discount()
        The yield curve for discounting future cash flows
    n_paths : int
        The number of paths to simulate
    """

    # Helper function to process one path
    def do_one_path(times):
        # move over each time node
        for t in times:
            # generate one standard normal variable
            std_norm = np.random.normal()
            # simulate the value of the underlying at the next time
            St = process.update(t, std_norm)
            # pass the value of the underlying at time t to the option_state
            # so it can use it when computing the payoff
            option_state.update(t, St)

        return option_state.calculate_payoff()

    # query for the times needed in the simulation
    times = option_state.times
    # initialize the running sum to zero
    running_sum = 0
    for i in range(n_paths):
        # reset process to start a new simulation
        process.reset()
        # reset option state to a start a new simulation
        option_state.reset()
        # simulate one path
        path_value = do_one_path(times)
        # update the running sum with the payoff at the end of the path
        running_sum += path_value

    return running_sum / n_paths * yield_curve.discount(option_state.expiry)
