import torch
from tqdm import tqdm
import torch.nn as nn


class Indi_cond:
    """
    The Indi_cond class implements the diffusion-like process for image generation of the InDI paper.
    It allows sampling new images based on a given model and input parameters.
    
    Attributes:
        noise_steps (int): Number of timesteps in the diffusion process.
        img_size (int): Size of the images.
        img_channel (int): Number of channels in the images.
        device (str): Device to use for computations (e.g., 'cuda' for GPU).
    """
    def __init__(self, img_size=256, img_channel=1, device="cuda"):
        self.img_channel = img_channel
        self.img_size = img_size
        self.device = device

    def sample_timesteps(self, n):
        """
        Samples random timesteps for the diffusion process.
    
        Args:
            n (int): Number of timesteps to sample.
    
        Returns:
            torch.Tensor: A tensor of randomly sampled timesteps.
        """
        return torch.randint(size=(n,))

    def indisample(self, x, model, steps):
        with torch.no_grad():
            for t in tqdm(torch.linspace(1, 0, steps + 1, device='cuda')[:-1]):
                if x.shape[0] > 1:
                    t = t[:, None, None, None]
                else:
                    t = torch.tensor([t], device='cuda')
                # print(t)
                predicted_peak = model(x, t)
                fct = 1 / (steps * t)
                x = (1 - fct) * x + fct * predicted_peak
            # print(y.shape)
            return x
