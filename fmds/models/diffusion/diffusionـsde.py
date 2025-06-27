import torch
import torch.nn as nn
import torch.optim as optim
import gc

from fmds.models.diffusion.stein_score_network import SteinScoreNetwork
class DiffusionSDE:
    def __init__(self, data,
                 input_dimension: int = 32,
                 hidden_dimension: int = 64,
                 number_hidden_layers: int = 3,
                 output_dimension: int = 32,
                 epochs:int=2500, lr: float =1e-3, noise_levels: int =10):
        self.data = data
        self.epochs = epochs
        self.lr = lr
        self.noise_levels = noise_levels
        self.score_net = SteinScoreNetwork(input_dimension=input_dimension,
                                           hidden_dimension=hidden_dimension,
                                           number_hidden_layers=number_hidden_layers,
                                           output_dimension=output_dimension    )
        self.optimizer = optim.Adam(self.score_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def fit(self):
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            sigma = torch.rand(1).item() * 0.5
            noise = sigma * torch.randn_like(self.data)
            noisy_data = self.data + noise
            score = self.score_net(x=noisy_data)
            loss = self.criterion(score, -noise / (sigma + 1e-8))  # Add small epsilon for stability
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

    def predictor_corrector(self, samples, step_size, num_corrector_steps):
        step_size_tensor = torch.tensor(step_size, device=samples.device, dtype=samples.dtype)
        for _ in range(num_corrector_steps):  # Fixed syntax error
            noise = torch.randn_like(samples)
            score = self.score_net(samples)
            samples = samples + step_size * score + torch.sqrt(step_size_tensor) * noise
            score = self.score_net(samples)
            samples = samples + step_size * score
        return samples

    def forward_diffusion_batch(self, data_points, number_of_steps=100):
        """Vectorized forward diffusion for better performance"""
        noisy_data = data_points.clone()
        dt = 1.0 / number_of_steps

        for step in range(number_of_steps):
            t = (step + 1) * dt
            sigma = t * 0.5
            noise = sigma * torch.randn_like(noisy_data) * torch.sqrt(torch.tensor(dt))
            noisy_data = noisy_data + noise

        return noisy_data

    def forward_diffusion(self, data_point, number_of_steps=100):
        """Single sample forward diffusion"""
        noisy_data_point = data_point.clone()
        dt = 1.0 / number_of_steps

        for step in range(number_of_steps):
            t = (step + 1) * dt
            sigma = t * 0.5
            noise = sigma * torch.randn_like(noisy_data_point) * torch.sqrt(torch.tensor(dt))
            noisy_data_point = noisy_data_point + noise

        return noisy_data_point

    def generate(self,
                 data=None,
                 n_samples=1000,
                 steps=1000,
                 step_size=0.01,  # Fixed type annotation
                 num_corrector_steps=5,
                 batch_size=100  # Add batching to prevent memory issues
                 ):

        all_samples = []

        # Process in batches to avoid memory issues
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            current_batch_size = batch_end - batch_start

            if data is None:
                samples = torch.randn(current_batch_size, 32)
            else:
                print(f'Generating noise from forward diffusion for batch {batch_start // batch_size + 1}')

                # Vectorized sampling and forward diffusion
                random_indices = torch.randint(0, data.size(0), (current_batch_size,))
                random_rows = data[random_indices]
                samples = self.forward_diffusion_batch(random_rows, number_of_steps=100)

            # Ensure samples are on the right device and dtype
            if hasattr(self.score_net, 'parameters'):
                device = next(self.score_net.parameters()).device
                samples = samples.to(device)

            step_size_tensor = torch.tensor(step_size, device=samples.device, dtype=samples.dtype)

            # Reverse diffusion process
            for step in range(steps):
                with torch.no_grad():  # Save memory during generation
                    noise = torch.randn_like(samples)
                    score = self.score_net(samples)
                    samples = samples + step_size * score + torch.sqrt(step_size_tensor) * noise
                    samples = self.predictor_corrector(samples, step_size, num_corrector_steps)

                # Optional: add gradient clipping to prevent explosion
                samples = torch.clamp(samples, -10, 10)

                # Clear cache periodically to prevent memory buildup
                if step % 100 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            all_samples.append(samples.cpu())  # Move to CPU to save GPU memory

            # Force garbage collection
            del samples
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_samples, dim=0)