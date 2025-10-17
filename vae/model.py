from pathlib import Path
from typing import Optional, Union

import torch

DEVICE = None
"""Device in which to run the model."""

# Choose the device to run the model: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class VAE(torch.nn.Module):
    """Variational Autoencoder (VAE) model.

    Args:
        torch (Module): PyTorch module.
    """

    def __init__(
        self,
        latent_dim: int,
        n_components: int = 10,
        beta: float = 1e-3,
    ) -> None:
        """Initialize the VAE model.

        Args:
            latent_dim (int): Dimension of the latent space.
            n_components (int): Number of components for the Gaussian Mixture Model.
            beta (float): Weight for the KL divergence loss term.
        """

        self.latent_dim: int = latent_dim
        self.n_components: int = n_components
        self.beta: float = beta
        self.gmm = None

        super(VAE, self).__init__()

        # Encoder: Convolutional layers that compress the image
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.Flatten(),
        )

        # Layers for the mean and log-variance of the latent space
        self.fc_mu = torch.nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = torch.nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_input = torch.nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (128, 8, 8)),
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.Sigmoid(),  # Output values between [0, 1]
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor: ...

    def build_L(self, L_params: torch.Tensor, latent_dim: int) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed image, mean, and log-variance.
        """
        # Encode the input
        x = self.encoder(x)

        # Get the mean and log-variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterize to get the latent vector
        z = self.reparameterize(mu, logvar)

        # Decode the latent vector
        # (First expand it to match the decoder input size)
        z_expanded = self.decoder_input(z)
        reconstructed = self.decoder(z_expanded)

        return reconstructed, mu, logvar

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        latent_dim: int,
        n_components: int = 10,
        device: torch.device = DEVICE,
        map_location: Optional[Union[torch.device, str]] = None,
    ) -> "VAE":
        """Instantiate a CVAE and load the weights stored in ``checkpoint_path``.

        Args:
            checkpoint_path: Location of the ``.pth`` file produced with ``torch.save``.
            latent_dim: Latent dimensionality used when the model was trained.
            n_components: Number of components for the class-conditional GMM. Defaults to 10.
            device: Device where the model should be placed after loading.
            map_location: Optional override for ``torch.load``'s ``map_location``.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` does not exist.

        Returns:
            CVAE: Model with the restored parameters, ready for inference.
        """

        # Ensure checkpoint path is a Path object
        checkpoint_path = Path(checkpoint_path)

        # Ensure the checkpoint file exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

        # Default map_location to device if not provided
        if map_location is None:
            map_location = device

        # Load the state dictionary fro the checkpoint
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model = cls(
            latent_dim=latent_dim,
            n_components=n_components,
        )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model
