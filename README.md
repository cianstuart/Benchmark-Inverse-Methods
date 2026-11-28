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
<img width="1907" height="965" alt="image" src="https://github.com/user-attachments/assets/b0e18cf1-5188-440d-8928-a34e347351d1" /> <img width="1907" height="965" alt="image" src="https://github.com/user-attachments/assets/f6fccc04-eebd-4fef-8185-ab49749f5218" /> <img width="1907" height="965" alt="image" src="https://github.com/user-attachments/assets/0d29c98c-b7f7-4947-a702-614926f7b6f9" />

## MCMC inverse solver

Run the MCMC inverse solver test with:

```bash
python -m pytest -s -q inverse_model_tests/test_MCMC_solver.py
```
<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/d52aee86-6e72-43d5-9d74-75a6ecfe4f71" />
