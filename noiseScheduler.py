import torch

class LinearNoiseScheduler:
    # Initialize the scheduler with a number of timesteps and beta range
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Create a tensor of betas linearly spaced between start and end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Calculate alphas as the complement of betas
        self.alphas = 1 - self.betas

        # Compute the cumulative product of alphas for noise scheduling
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

        # Precompute square roots for efficiency during noise addition
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    # Add noise to the original image based on the timestep
    def add_noise(self, original, noise, t):
        batch_size = original.shape[0]

        # Expand the precomputed square roots to match the batch size
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].unsqueeze(0).expand(batch_size, -1)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].unsqueeze(0).expand(batch_size, -1)

        # Add additional dimensions to match the original image shape
        for _ in range(len(original.shape) - 2): 
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Compute the noisy image as a weighted sum of the original and noise
        noisy_image = sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
        return noisy_image
    
    def sample_prev_timestep(self, xt, noise_pred, t):
            # Estimate the original image x0 from the noisy image xt at timestep t
            x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
            x0 = torch.clamp(x0, -1, 1)  # Clamp the values to be within [-1, 1]

            # Compute the mean of the distribution for the previous timestep
            mean = xt - ((self.betas[t] * noise_pred)) / self.sqrt_one_minus_alpha_cum_prod[t]
            mean = mean / torch.sqrt(self.alphas[t])

            if t == 0:
                return mean, x0
            else:
                # Compute the variance and standard deviation for the noise
                variance = (1 - self.alpha_cum_prod[t-1]) / (1 - self.alpha_cum_prod[t]) * self.betas[t]
                sigma = torch.sqrt(variance)
                z = torch.randn(xt.shape).to(xt.device)
                return mean + sigma * z, x0

# Unit test function for sample_prev_timestep
def test_sample_prev_timestep():
    scheduler = LinearNoiseScheduler(10, 0.0001, 0.02)
    x = torch.randn((3, 3, 2, 2))
    noise_pred = torch.randn_like(x)
    t = 5

    mean, x0 = scheduler.sample_prev_timestep(x, noise_pred, t)
    assert mean.shape == x.shape, f"Expected shape {x.shape}, but got {mean.shape}"
    assert x0.shape == x.shape, f"Expected shape {x.shape}, but got {x0.shape}"
    print("test_sample_prev_timestep passed.")

# Run unit tests
if __name__ == "__main__":
    test_sample_prev_timestep()