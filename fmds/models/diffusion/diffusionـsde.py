import torch
import torch.nn as nn
import torch.optim as optim
import gc
from sklearn.preprocessing import StandardScaler

from fmds.models.diffusion.stein_score_network import SteinScoreNetwork

def get_vp_noise_schedule(num_steps: int, beta_min: float = 0.1, beta_max: float = 20.0) -> tuple:
    """
    Generate a VP (Variance Preserving) noise schedule.

    Args:
        num_steps: Number of discrete time steps.
        beta_min: Minimum beta (controls early noise).
        beta_max: Maximum beta (controls final noise).

    Returns:
        tuple: (sigma_t, beta_t) - standard deviations and beta values for each time step.
    """
    t = torch.linspace(0., 1., steps=num_steps)
    beta_t = beta_min + t * (beta_max - beta_min)
    log_alpha = -0.5 * torch.cumsum(beta_t * (1.0 / num_steps), dim=0)
    alpha = torch.exp(log_alpha)
    sigma = torch.sqrt(1 - alpha ** 2)
    return sigma, beta_t


class DiffusionSDE:
    def __init__(self,
                 data: torch.Tensor,
                 input_dimension: int = 2,
                 hidden_dimension: int = 64,
                 number_hidden_layers: int = 3,
                 output_dimension: int = 2,
                 epochs: int = 2500,
                 lr: float = 1e-3,
                 num_noise_levels: int = 1000,
                 dropout_prob: float = 0.25,
                 beta_min: float = 0.1,
                 beta_max: float = 20.0):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dimension = input_dimension
        self.epochs = epochs
        self.lr = lr
        self.num_noise_levels = num_noise_levels
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Normalize data
        self.scaler = StandardScaler()
        data_np = data.cpu().numpy()
        normalized_data_np = self.scaler.fit_transform(data_np)
        self.data = torch.tensor(normalized_data_np, dtype=torch.float32).to(self.device)

        # Get noise schedule
        self.noise_schedule, self.beta_schedule = get_vp_noise_schedule(
            num_noise_levels, beta_min=beta_min, beta_max=beta_max
        )
        self.noise_schedule = self.noise_schedule.to(self.device)
        self.beta_schedule = self.beta_schedule.to(self.device)

        # Initialize score network
        self.score_net = SteinScoreNetwork(
            input_dimension=input_dimension + 1,  # data + sigma
            hidden_dimension=hidden_dimension,
            number_hidden_layers=number_hidden_layers,
            output_dimension=output_dimension,
            dropout_prob=dropout_prob
        ).to(self.device)

        self.optimizer = optim.Adam(self.score_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def fit(self):
        """Train the score network using denoising score matching."""
        self.score_net.train()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # Sample random noise levels
            sigma_indices = torch.randint(0, self.num_noise_levels, (self.data.size(0),), device=self.device)
            sigmas = self.noise_schedule[sigma_indices].view(-1, 1)

            # Add noise to data
            noise = torch.randn_like(self.data)
            noisy_data = self.data + sigmas * noise

            # Prepare input and compute score
            net_input = torch.cat([noisy_data, sigmas], dim=1)
            score_pred = self.score_net(net_input)

            # Score matching objective: predict -noise/sigma
            score_target = -noise / (sigmas + 1e-8)
            loss = self.criterion(score_pred, score_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=10.0)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    def corrector_step(self, samples: torch.Tensor, sigma: torch.Tensor,
                       n_steps: int = 1, snr: float = 0.16) -> torch.Tensor:
        """
        Perform Langevin MCMC corrector steps.

        Args:
            samples: Current samples
            sigma: Current noise level
            n_steps: Number of corrector steps
            snr: Signal-to-noise ratio for step size
        """
        step_size = 2 * (snr * sigma) ** 2

        for _ in range(n_steps):
            noise = torch.randn_like(samples)

            # Ensure sigma has correct shape
            sigma_expanded = sigma.view(-1, 1).expand(samples.shape[0], 1)
            net_input = torch.cat([samples, sigma_expanded], dim=1)

            with torch.no_grad():
                score = self.score_net(net_input)

            samples = samples + step_size * score + torch.sqrt(2 * step_size) * noise

        return samples

    def get_beta_t(self, t: float) -> float:
        """Get beta value at time t (0 to 1)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def generate(self,
                 n_samples: int = 1000,
                 steps: int = 1000,
                 num_corrector_steps: int = 1,
                 batch_size: int = 100,
                 denormalize: bool = True,
                 snr: float = 0.16) -> torch.Tensor:
        """
        Generate samples using PC sampler (Predictor-Corrector).

        Args:
            n_samples: Number of samples to generate
            steps: Number of discretization steps
            num_corrector_steps: Number of Langevin MCMC steps per iteration
            batch_size: Batch size for generation
            denormalize: Whether to denormalize the generated samples
            snr: Signal-to-noise ratio for corrector steps
        """
        self.score_net.eval()
        all_samples = []

        # Time steps from T to 0
        time_steps = torch.linspace(1., 0., steps + 1).to(self.device)
        dt = 1.0 / steps

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            current_batch_size = batch_end - batch_start

            # Initialize with pure noise
            samples = torch.randn(current_batch_size, self.input_dimension).to(self.device)
            samples = samples * self.noise_schedule[-1].item()  # Scale by maximum sigma

            with torch.no_grad():
                for i in range(steps):
                    t_current = time_steps[i]
                    t_next = time_steps[i + 1]

                    # Get noise levels
                    idx_current = int(t_current * (self.num_noise_levels - 1))
                    idx_next = int(t_next * (self.num_noise_levels - 1))

                    sigma_current = self.noise_schedule[idx_current]
                    sigma_next = self.noise_schedule[idx_next]

                    # Get beta value
                    beta_t = self.get_beta_t(t_current)

                    # Prepare input for score network
                    sigma_expanded = sigma_current.view(1, 1).expand(current_batch_size, 1)
                    net_input = torch.cat([samples, sigma_expanded], dim=1)
                    score = self.score_net(net_input)

                    # Predictor step (reverse SDE)
                    drift = -0.5 * beta_t * samples
                    diffusion = torch.sqrt(beta_t)

                    # Euler-Maruyama step
                    z = torch.randn_like(samples) if i < steps - 1 else torch.zeros_like(samples)
                    # samples = samples - dt * (drift + diffusion ** 2 * score) + diffusion * torch.sqrt(dt) * z
                    samples = samples - dt * (drift + diffusion ** 2 * score) + diffusion * np.sqrt(dt) * z

                    # Corrector step (optional)
                    if num_corrector_steps > 0 and i < steps - 1:
                        samples = self.corrector_step(samples, sigma_next, num_corrector_steps, snr)

            all_samples.append(samples.cpu())
            del samples
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        samples = torch.cat(all_samples, dim=0)

        if denormalize:
            samples_np = samples.numpy()
            return torch.tensor(self.scaler.inverse_transform(samples_np), dtype=torch.float32)

        return samples

    def denormalize(self, samples: torch.Tensor) -> torch.Tensor:
        """Denormalize samples to original scale."""
        samples_np = samples.cpu().numpy()
        return torch.tensor(self.scaler.inverse_transform(samples_np), dtype=torch.float32)

