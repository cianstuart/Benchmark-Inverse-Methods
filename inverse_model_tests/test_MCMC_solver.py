from unittest import result
from itertools import permutations

import numpy as np
import pytest
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import parameters as params
from forward_model.forward_solver import ForwardSolver
from inverse_model.MCMC_solver import MCMC_Solver
from unittest.mock import patch

def test_MCMC():
    rng = np.random.default_rng(42)
    sources = rng.uniform(
        low=[0.0, 0.0, 0.05, 0.0],
        high=[1.0, 1.0, 1.0, 1.0],
        size=(params.N_SOURCE, params.SOURCE_DIM),
    )
    observed_field = ForwardSolver(params).forward(sources.reshape(-1)) #flatten the 2d array as I have a resize in the forward solver
    # test the solver with and without observation noise
    for observation_noise_std in [0.0, 0.01]:
        observed_field_noisy = observed_field + rng.normal(
            loc=0.0,
            scale=observation_noise_std,
            size=observed_field.shape,
        )

        solver = MCMC_Solver(params, noise_std=0.01, seed=456)
        with patch.object(solver.model, 'forward', wraps=solver.model.forward) as forward_calls:
            result = solver.sample(observed_field_noisy, n_iter=10000, burn_in=5000, thin=10, swap_freq=10)

        print("Forward solver calls during MCMC sampling:", forward_calls.call_count)

        print("Noise std:", observation_noise_std)
        print("Acceptance rates:", result.acceptance_rates)
        print("Swap rate:", result.swap_rate)

        idx_best = np.argmax(result.log_likelihoods)
        best_sample = result.samples[idx_best]
        best_field = solver.model.forward(best_sample)

        SSIM = ssim(observed_field, best_field, data_range=observed_field.max() - observed_field.min())
        print("SSIM between observed field and best sample:", SSIM)

        # plot the observed field and the best sample field and the true and sampled source locations
        best_sources = best_sample.reshape(params.N_SOURCE, params.SOURCE_DIM)
        true_sources = sources.reshape(params.N_SOURCE, params.SOURCE_DIM)

        # sources may be in different order
        matched = min((best_sources[list(p)] for p in permutations(range(len(best_sources)))),
              key=lambda e: np.sum((e - true_sources)**2))

        print("True sources:\n", true_sources)
        print("Matched sources:\n", matched)

        RMSE_SOURCES = np.sqrt(np.mean((matched - true_sources)**2))
        print("RMSE between true sources and best sample sources:", RMSE_SOURCES)
        print("Max log-likelihood:", result.log_likelihoods[idx_best])

        vmin = min(observed_field.min(), best_field.min())
        vmax = max(observed_field.max(), best_field.max())

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        plots = [
            (axes[0], observed_field, true_sources, "Observed field with noise_std: " + str(observation_noise_std), "True source locations"),
            (axes[1], best_field, best_sources, "Best-sample field", "Best-sample source locations"),
        ]

        for axis, field, source_locations, title, source_label in plots:
            image = axis.imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="inferno",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )

            axis.scatter(
                source_locations[:, 0],
                source_locations[:, 1],
                marker="x",
                color="cyan",
                label=source_label,
            )

            axis.set(
                title=title,
                xlabel="x",
                ylabel="y",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            axis.legend()

    plt.show()

