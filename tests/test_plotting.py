import matplotlib
import matplotlib.pyplot as plt

import acteval as ae

matplotlib.use("Agg")


def test_mvp_plots_render() -> None:
    observed = list(range(1, 21))
    predicted = [value * 0.9 + 0.5 for value in observed]
    functions = [
        lambda: ae.plot_calibration(observed, predicted, n_bins=5),
        lambda: ae.plot_lift(observed, predicted, n_bins=5),
        lambda: ae.plot_residuals(observed, predicted),
        lambda: ae.plot_tail_diagnostics(observed, predicted, quantile=0.9),
    ]
    for make_plot in functions:
        axis = make_plot()
        assert axis.get_title()
        assert axis.figure.canvas is not None
        plt.close(axis.figure)
