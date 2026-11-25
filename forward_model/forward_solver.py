import numpy as np
import parameters as params
from dataclasses import dataclass

class ForwardSolver:
    '''
    Solver for the forward model: spatial grid with N_SOURCE sources and nonlinearity factor ALPHA.
    Sources are parametrized as a  Gaussian distribution G_i(x,y) with 
    1. Mean at the source location (x,y).
    2. Standard deviation sigma.
    3. Amplitude A.

    The observed field is T(x,y) = tanh(S(x,y) + ALPHA * S(x,y)^2) 
    where S(x,y) is the sum of all sources at (x,y)
    '''
    def __init__(self, params):
        self.N = params.N
        self.ALPHA = params.ALPHA
        self.N_SOURCE = params.N_SOURCE
        self.SOURCE_DIM = params.SOURCE_DIM

    def reshape_sources(self, sources):
        '''
        Sources are provided as a flat array of shape (N_SOURCE * SOURCE_DIM,).
        where sources = [x1, y1, sigma1, A1, x2, y2, sigma2, A2, ..., xN_SOURCE, yN_SOURCE, sigmaN_SOURCE, AN_SOURCE]
        Reshape them into a 2D array of shape (N_SOURCE, SOURCE_DIM).
        '''
        assert sources.size == self.N_SOURCE * self.SOURCE_DIM, f"Sources should have size {self.N_SOURCE * self.SOURCE_DIM}"
        return sources.reshape((self.N_SOURCE, self.SOURCE_DIM))
    
    def forward(self, sources):
        '''
        Compute the forward model given the sources.

        Args:
            sources: np array of shape (N_SOURCE, SOURCE_DIM) where each row is (x, y, sigma, A)

        x,y are continuous coordinates of the source in range [0, N-1], 
        sigma is the standard deviation of the Gaussian, 
        and A is the amplitude.

        Returns:
            T: A numpy array of shape (N, N) representing the observed field.
        '''
        # Check that sources have the correct shape
        sources_r = self.reshape_sources(sources)
        assert sources_r.shape == (self.N_SOURCE, self.SOURCE_DIM), f"Sources should have shape ({self.N_SOURCE}, {self.SOURCE_DIM})"
        
        # Create a grid of coordinates
        x = np.linspace(0, self.N - 1, self.N)
        y = np.linspace(0, self.N - 1, self.N)
        X, Y = np.meshgrid(x, y)

        # Initialize the source field S(x,y)
        S = np.zeros((self.N, self.N))

        # Sum contributions from each source
        for i in range(self.N_SOURCE):
            x_i, y_i, sigma_i, A_i = sources_r[i]
            G_i = A_i * np.exp(-((X - x_i)**2 + (Y - y_i)**2) / (2 * sigma_i**2))
            S += G_i

        # Compute the observed field T(x,y)
        T = np.tanh(S + self.ALPHA * S**2)

        return T

