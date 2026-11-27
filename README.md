# Benchmark-Inverse-Methods
This repository implements optimization and machine learning methods to solve a synthetic inverse problem.

## Forward solver 

Run the solver test with:

```bash
pytest -q forward_model/test_forward_solver.py
```

To run the example and display its output as a plot:

```bash
python -m forward_model.test_forward_solver
```

## MCMC inverse solver

Run the MCMC inverse solver test with:

```bash
python -m pytest -s -q inverse_model_tests/test_MCMC_solver.py
```