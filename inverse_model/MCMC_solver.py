import numpy as np
from dataclasses import dataclass
from forward_model.forward_solver import ForwardSolver

@dataclass
class MCMC_Results:
    samples: np.ndarray
    log_likelihoods: np.ndarray
    acceptance_rates: np.ndarray
    swap_rate: float

class MCMC_Solver:
    '''
    MCMC (Metropolis-Hastings) solver for the inverse problem: 
    given an observed field T_obs, find the posterior distribution of the source parameters.
    Prior distribution is uniform over the parameter space
    An iid Gaussian log-likelihood describes the difference between the observed field and the predicted field.
    Parallel tempering is used to improve convergence and avoid local minima (like the paper).
    12 chains are used, with beta values ranging from 0.0001 to 1, geometrically spaced.
    beta = 1 represents the true posterior distribution, 
    while beta < 1 represents tempered distributions.
    '''
    def __init__(self, params, noise_std=0.01, beta_min=0.0001, seed=42):
        self.model = ForwardSolver(params)
        self.N_SOURCE = params.N_SOURCE
        self.SOURCE_DIM = params.SOURCE_DIM
        self.N_CHAINS = self.N_SOURCE * self.SOURCE_DIM  # Number of chains for parallel tempering = number of parameters
        self.NOISE_STD = noise_std
        self.BETA_MIN = beta_min
        self.betas = np.geomspace(1, self.BETA_MIN, self.N_CHAINS)  # Geometrically spaced beta values for parallel tempering (1 --> beta_min)
        self.rng = np.random.default_rng(seed)                      # Random number generator

        self.lower = np.array([0, 0, 0.05, 0] * self.N_SOURCE)  # Lower bounds 
        self.upper = np.array([1, 1, 1, 1] * self.N_SOURCE)  # Upper bounds 

    def log_likelihood(self, theta, T_obs):
        error = T_obs - self.model.forward(theta)
        return -0.5 * np.sum((error / self.NOISE_STD) ** 2)

    def sample(self, T_obs, n_iter=5000, burn_in=2500, thin=10, swap_freq=10):
        '''
        Run the MCMC sampler to sample from the posterior distribution of the source parameters given the observed field T_obs.

        Args:
            T_obs: np array of shape (N, N) representing the observed field
            n_iter: total number of MCMC iterations
            burn_in: number of initial burn-in iterations to discard
            thin: thinning factor to save memory (only keep every 'thin'th sample)
            swap_freq: frequency of parallel tempering swaps
        '''
        # Initialize chains and log-likelihoods
        theta = self.rng.uniform(self.lower, self.upper, size=(self.N_CHAINS, self.N_SOURCE * self.SOURCE_DIM))
        log_likelihoods = np.array([self.log_likelihood(theta[i], T_obs) for i in range(self.N_CHAINS)])

        # define different step sizes for each chain
        step_sizes = 0.001 / np.sqrt(self.betas)    # Smaller step sizes for higher beta chains

        # keep track of acceptance rate, swap rate (after burn-in), samples and log-likelihoods
        n_acc = np.zeros(self.N_CHAINS)
        n_swap = 0
        n_swap_attempts = 0
        samples = []
        samples_log_likelihoods = []

        # Start sampling
        for iter in range(n_iter):
            # Step 1: Metropolis-Hastings updates for each chain, one source at a time
            for k in range(self.N_CHAINS):
                j = self.rng.integers(self.N_SOURCE)  # Randomly select a source to update
                idx = slice(j * self.SOURCE_DIM, (j + 1) * self.SOURCE_DIM)  # Indices for the selected source
                theta_proposal = theta[k].copy()
                theta_proposal[idx] += step_sizes[k] * self.rng.normal(size=self.SOURCE_DIM)  # Propose new parameters for the selected source using a Gaussian proposal distribution

                # Check if the proposal is within bounds and accept or reject based on the Metropolis-Hastings criterion
                if np.all(theta_proposal >= self.lower) and np.all(theta_proposal <= self.upper):
                    log_likelihood_proposal = self.log_likelihood(theta_proposal, T_obs)
                    log_accept_ratio = (log_likelihood_proposal - log_likelihoods[k]) * self.betas[k]
                    if np.log(self.rng.random()) < log_accept_ratio:
                        theta[k] = theta_proposal
                        log_likelihoods[k] = log_likelihood_proposal
                        if iter >= burn_in:  # Only count acceptance after burn-in
                            n_acc[k] += 1

            # Step 2: Parallel tempering swaps between adjacent chains (steps of 2 to avoid double swapping, alternate starts at either 0 or 1)
            if iter % swap_freq == 0:
                for k in range((iter//swap_freq) % 2, self.N_CHAINS - 1, 2): #ensures alternate starts (remainder is 0 or 1)
                    log_accept_ratio = (log_likelihoods[k + 1] - log_likelihoods[k]) * (self.betas[k] - self.betas[k + 1])

                    if iter >= burn_in:  # Only count attempted swaps after burn-in
                        n_swap_attempts += 1

                    if np.log(self.rng.random()) < log_accept_ratio:
                        theta[[k, k + 1]] = theta[[k + 1, k]]
                        log_likelihoods[[k, k + 1]] = log_likelihoods[[k + 1, k]]
                        if iter >= burn_in:  # Only count swaps after burn-in
                            n_swap += 1

            # Step 3: Store samples and log-likelihoods after burn-in and thinning (only store beta = 1 chain)
            if iter >= burn_in and iter % thin == 0:
                samples.append(theta[0].copy())  
                samples_log_likelihoods.append(log_likelihoods[0])

        acceptance_rates = n_acc / (n_iter - burn_in)
        swap_rate = (n_swap / n_swap_attempts) if n_swap_attempts > 0 else 0.0


        return MCMC_Results(
            samples=np.array(samples),
            log_likelihoods=np.array(samples_log_likelihoods),
            acceptance_rates=acceptance_rates,
            swap_rate=swap_rate
        )