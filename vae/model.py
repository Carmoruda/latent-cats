from pathlib import Path
from typing import Optional, Union

import torch
from torch.nn import functional as F

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
        self.fc_L_params = torch.nn.Sequential(
            torch.nn.Linear(128 * 8 * 8, latent_dim * (latent_dim + 1) // 2),
            torch.nn.LayerNorm(latent_dim * (latent_dim + 1) // 2),
        )

        # Decoder
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

    def reparameterize(
        self, mu: torch.Tensor, L_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample latent vectors using the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian, shape `(batch, latent_dim)`.
            L_params (torch.Tensor): Packed lower-triangular entries used to build the Cholesky factor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Sampled latent vectors of shape `(batch, latent_dim)` and the corresponding Cholesky factor `L`.
        """

        # Build the Cholesky factor L from the packed parameters
        L = self.build_L(L_params)
        eps = torch.randn(mu.shape[0], self.latent_dim, 1, device=mu.device)
        z = mu.unsqueeze(-1) + torch.bmm(L, eps)

        return z.squeeze(-1), L

    def build_L(self, L_params: torch.Tensor) -> torch.Tensor:
        """Construct the lower-triangular Cholesky factor from flattened parameters.

        Args:
            L_params (torch.Tensor): Tensor of shape `(batch, latent_dim * (latent_dim + 1) // 2)` containing the packed lower-triangular entries for each sample.

        Returns:
            torch.Tensor: Batch of lower-triangular matrices `L` with positive diagonal, shape `(batch, latent_dim, latent_dim)`.
        """
        # Construct the lower-triangular matrix
        batch_size = L_params.size(0)
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=L_params.device)

        # Get the indices for the lower-triangular part
        row_idx, col_idx = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)
        L[:, row_idx, col_idx] = L_params

        # Enforce a positive diagonal for numerical stability.
        diag_idx = torch.arange(self.latent_dim)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx])

        return L

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed image, mean, and Cholensky factor.
        """
        # Encode the input
        x = self.encoder(x)

        # Get the mean and log-variance
        mu = self.fc_mu(x)
        L_params = self.fc_L_params(x)

        # Reparameterize to get the latent vector
        z, L = self.reparameterize(mu, L_params)

        # Decode the latent vector
        # (First expand it to match the decoder input size)
        reconstructed = self.decoder(z)

        return reconstructed, mu, L

    def loss(
        self, reconstructed_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, L: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute VAE reconstruction and KL losses with a Cholesky covariance.

        Args:
            recon_x (torch.Tensor): Decoder outputs with the same shape as `x`.
            x (torch.Tensor): Ground truth inputs the VAE is trying to reconstruct.
            mu (torch.Tensor): Mean of the latent Gaussian, shape `(batch, latent_dim)`.
            L (torch.Tensor): Lower-triangular Cholesky factor of the covariance, shape `(batch, latent_dim, latent_dim)`.
            beta (float, optional): Scaling factor applied to the KL term. Defaults to 0.01.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean-squared reconstruction loss and beta-scaled KL divergence loss.
        """
        recon_loss = F.mse_loss(reconstructed_x, x, reduction="mean")

        diag_L = torch.diagonal(L, dim1=1, dim2=2)
        log_diag_L = torch.log(diag_L)
        log_det_sigma = 2 * torch.sum(log_diag_L, dim=1)

        trace_sigma = torch.sum(L.pow(2), dim=(1, 2))
        squared_mean = torch.sum(mu.pow(2), dim=1)
        latent_dim = mu.size(1)

        kl_div = 0.5 * (trace_sigma + squared_mean - latent_dim - log_det_sigma)
        kl_loss = torch.mean(kl_div)

        return recon_loss, (kl_loss * self.beta)

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
