import numpy as np

import parameters as params
from forward_model.forward_solver import ForwardSolver


EXAMPLE_SOURCES = np.array(
    [
        [0.1, 0.7, 3.0, 1.5],
        [0.5, 0.4, 3.0, 1.2],
        [0.8, 0.4, 3.0, 2.3],
    ]
)


def calculate_example_field():
    """Run the forward solver with three separated example sources."""
    return ForwardSolver(params).forward(EXAMPLE_SOURCES)


def test_forward_solver_returns_valid_field():
    field = calculate_example_field()

    assert field.shape == (params.N, params.N)
    assert np.all(np.isfinite(field))
    assert np.all((field >= 0.0) & (field < 1.0))
    assert field[2, 2] > field[0, 0]


def plot_example_field():
    import matplotlib.pyplot as plt

    field = calculate_example_field()
    sources = EXAMPLE_SOURCES.reshape(params.N_SOURCE, params.SOURCE_DIM)

    figure, axis = plt.subplots()
    image = axis.imshow(
        field,
        origin="lower",
        extent=(0, params.N - 1, 0, params.N - 1),
        cmap="inferno",
    )
    axis.scatter(
        sources[:, 0]*(params.N - 1),
        sources[:, 1]*(params.N - 1),
        marker="x",
        color="cyan",
        label="Source locations",
    )
    axis.set(
        title="Forward solver output",
        xlabel="x",
        ylabel="y",
        xlim=(0, params.N - 1),
        ylim=(0, params.N - 1),
    )
    axis.legend()
    figure.colorbar(image, ax=axis, label="Observed field")
    plt.show()


if __name__ == "__main__":
    plot_example_field()
