import numpy as np
import parameters as params

class ForwardSolver:
    '''
    Solver for the forward model: spatial grid with N_SOURCE sources and nonlinearity factor ALPHA.
    Sources are parametrized as a  Gaussian distribution G_i(x,y) with 
    1. Mean at the source location (x,y).
    2. Standard deviation sigma.
    3. Amplitude A.

    The observed field is T(x,y) = tanh(S(x,y)) 
    where S(x,y) is the sum of all sources at (x,y) after diffusion iterations.
    '''
    def __init__(self, params):
        self.N = params.N
        self.DIF_COEF = params.DIF_COEF
        self.DIF_ITER = params.DIF_ITER
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

    def calculate_laplacian(self, field):
        '''
        Estimate the Laplacian of the field using finite differences. Grid has equal spacings of 1 in both x and y directions. 
        Thus Laplacian(i,j) = field(i+1,j) + field(i-1,j) + field(i,j+1) + field(i,j-1) - 4*field(i,j).
        Edge points are handled by assuming gradient is zero outside the grid (Neumann boundary conditions) thus pad with edge values.
        '''
        padded_field = np.pad(field, pad_width=1, mode='edge')
        
        laplacian = (
            padded_field[2:, 1:-1] +                # one row below
            padded_field[:-2, 1:-1] +               # one row above
            padded_field[1:-1, 2:] +                # one column to the right
            padded_field[1:-1, :-2] -               # one column to the left
            4 * padded_field[1:-1, 1:-1]            # center point
        )   
        return laplacian

    def forward(self, sources):
        '''
        Compute the forward model given the sources.

        Args:
            sources: np array of shape (N_SOURCE, SOURCE_DIM) where each row is (x, y, sigma, A)

        x,y are continuous coordinates of the source in range [0, 1], scaled to the grid size N.
        sigma is the standard deviation of the Gaussian, 
        and A is the amplitude.

        Forward model implements a diffusion-like process where each source gets spread out over the grid using the Laplacian.
        To make it nonlinear, we apply a tanh()-function after the diffusion step.
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
            x_i, y_i = x_i * (self.N - 1), y_i * (self.N - 1)  # Scale to grid size
            G_i = A_i * np.exp(-((X - x_i)**2 + (Y - y_i)**2) / (2 * sigma_i**2))
            S += G_i

        # Diffusion iterations
        for _ in range(self.DIF_ITER):  
            S += self.DIF_COEF*self.calculate_laplacian(S)

        # Compute the observed field T(x,y)
        T = np.tanh(S)

        return T

