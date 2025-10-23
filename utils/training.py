from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, cast

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from vae import VAE, VAEDataset

TRAIN_FRACTION: float = 0.8
"""Fraction of the dataset to use for training."""

VAL_FRACTION: float = 0.1
"""Fraction of the dataset to use for validation."""


def prepare_dataloader(
    data_dir: Path,
    *,
    length: Optional[int] = None,
    image_size: int = 64,
    batch_size: int,
    seed: Optional[int] = None,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare the dataset and dataloaders for training, validation, and testing.

    Args:
        data_dir (Path): Path to the data directory.
        length (Optional[int], optional): Length of the dataset. Defaults to None.
        image_size (int): Size to which images will be resized.
        batch_size (int): Batch size for the dataloaders.
        seed (Optional[int], optional): Random seed for data splitting. Defaults to None.
        download (bool, optional): Whether to download the dataset if not found. Defaults to False.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and testing.
    """

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = VAEDataset(
        "https://www.kaggle.com/api/v1/datasets/download/borhanitrash/cat-dataset",
        dataset_path=str(data_dir),
        extracted_folder="cats",
        transform=transform,
        download=download,
        delete_extracted=True,
    )

    if length:
        dataset = _balance_dataset(dataset, length, seed=seed)

    def split_dataset(
        dataset: Dataset, *, generator: torch.Generator
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Split the dataset into train, validation, and test sets.

        Args:
            dataset (Dataset): The dataset to split.
            generator (torch.Generator): The random number generator to use for splitting.

        Returns:
            tuple[Dataset, Dataset, Dataset]: The train, validation, and test datasets.
        """

        # Calculate size of the dataset
        dataset_size = len(cast(Sized, dataset))

        # Calculate sizes for train, val, test splits
        train_size = int(TRAIN_FRACTION * dataset_size)
        val_size = int(VAL_FRACTION * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Split the dataset
        splits = random_split(dataset, [train_size, val_size, test_size], generator=generator)

        return tuple(splits)  # type: ignore

    split_generator = torch.Generator()

    if seed:
        split_generator.manual_seed(seed)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, generator=split_generator)

    loader_generator = torch.Generator()

    if seed:
        loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, generator=loader_generator
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, generator=loader_generator
    )

    return train_loader, val_loader, test_loader


def run_training_pipeline(
    config: Mapping[str, Any],
    *,
    data_dir: Path,
    output_dir: Path,
    report_callback: Optional[Callable[[dict[str, float]], None]] = None,
    device: Optional[torch.device] = None,
) -> tuple[VAE, DataLoader, DataLoader, DataLoader]:
    """Run the training pipeline for the VAE model.

    Args:
        config (Mapping[str, Any]): Configuration parameters for training.
        data_dir (Path): Path to the data directory.
        output_dir (Path): Path to the output directory.
        report_callback (Optional[Callable[[dict[str, float]], None]], optional): Callback function to report metrics. Defaults to None.
        device (Optional[torch.device], optional): Device to run the training on. Defaults to None.

    Returns:
        tuple[VAE, DataLoader, DataLoader, DataLoader]: The trained VAE model and the data loaders.
    """

    # Check that directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the configuration
    configuration = dict(config)
    batch_size = int(configuration.get("batch_size", 64))
    seed = configuration.get("seed", None)
    latent_dim = int(configuration.get("latent_dim", 20))
    beta = float(configuration.get("beta", 1e-3))
    n_components = int(configuration.get("n_components", 10))
    learning_rate = float(configuration.get("lr", 1e-3))
    epochs = int(configuration.get("epochs", 50))
    loss_function = configuration.get("loss_function", F.mse_loss)
    download = bool(configuration.get("download", False))
    image_size = int(configuration.get("image_size", 128))

    # Create the model
    model = VAE(
        latent_dim=latent_dim,
        n_components=n_components,
        beta=beta,
    )

    model_device = _get_device(model, device)
    model.to(model_device)

    # Prepare the dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloader(
        data_dir, image_size=image_size, batch_size=batch_size, seed=seed, download=download
    )

    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    # -- Training loop --
    for epoch in range(epochs):
        _, val_history = train_model(
            model,
            loss_function,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            epoch=epoch,
            num_epochs=epochs,
            device=model_device,
        )

        if report_callback:
            metrics = {"val_loss": val_history[-1]}
            report_callback(metrics)

    # Fit gmm
    gmm(model, train_dataloader, n_samples=1000)

    get_reconstructions(
        model,
        val_dataloader,
        root_path=output_dir,
        file_name=output_dir / "reconstructions.png",
        device=model_device,
    )

    return model, train_dataloader, val_dataloader, test_dataloader


def train_model(
    model: VAE,
    loss_function: Optional[Callable[..., torch.Tensor]],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    *,
    epoch: int,
    num_epochs: int,
    device: Optional[torch.device] = None,
) -> tuple[list[float], list[float]]:
    """Train the VAE model.
    Args:
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
        epoch (int, optional): Number of epoch in training.
        num_epochs (int, optional): Total number of epochs for training. Defaults to 10.
    Returns:
        tuple[list[float], list[float]]: Histories of training and validation losses.
    """

    model_device = _get_device(model, device)
    model.to(model_device)

    if loss_function is None:
        loss_function = F.mse_loss

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0
    total_train_samples = 0

    for images in train_dataloader:
        images = images.to(model_device)

        # Forward pass
        reconstructed_images, mu, L = model(images)

        # Calculate loss
        reconstruction_loss, kl_loss = model.loss(reconstructed_images, images, mu, L)
        loss = (reconstruction_loss) + (kl_loss)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_train_loss += loss.item() * batch_size
        total_train_samples += batch_size

    avg_train_loss = total_train_loss / max(total_train_samples, 1)
    train_loss_history.append(avg_train_loss)

    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0

    with torch.no_grad():
        for images in val_dataloader:
            images = images.to(model_device)

            # Forward pass
            reconstructed_images, mu, L = model(images)

            # Calculate loss
            reconstruction_loss, kl_loss = model.loss(reconstructed_images, images, mu, L)
            val_loss = (reconstruction_loss) + (kl_loss)

            batch_size = images.size(0)
            total_val_loss += val_loss.item() * batch_size
            total_val_samples += batch_size

    avg_val_loss = total_val_loss / max(total_val_samples, 1)
    val_loss_history.append(avg_val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    # Update the learning rate scheduler
    scheduler.step(avg_val_loss)

    return train_loss_history, val_loss_history


def generate_images(
    model: VAE,
    *,
    root_path: Path,
    file_name: Path = Path("generated_images.png"),
    num_images: int = 5,
    device: Optional[torch.device] = None,
) -> None:
    """Generate new images by sampling from the latent space.

    Args:
        model (VAE): The VAE model to use.
        root_path (Path): The rooth path for saving the generated images.
        file_name (Path, optional): The file name for saving the generated images. Defaults to Path("generated_images.png").
        num_images (int, optional): The number of images to generate. Defaults to 5.
        device (Optional[torch.device], optional): The device to use for generating images. Defaults to None.
    """

    # Get the device of the model
    model_device = _get_device(model, device)
    model.eval()

    # Create directory to save generated images if it doesn't exist
    root_path.mkdir(parents=True, exist_ok=True)

    # Create full file path to save the generated images
    file_name = root_path / file_name
    file_name.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # If the model has a GMM, sample from it
        if hasattr(model, "gmm") and model.gmm is not None:
            samples, _ = model.gmm.sample(num_images)
            # Convert samples to torch tensor
            z = torch.from_numpy(samples).float().to(model_device)
        else:
            # Sample from standard normal distribution
            z = torch.randn(num_images, model.latent_dim).to(model_device)

        z_expanded = model.decoder_input(z)
        generated_images = model.decoder(z_expanded)

    _plot_generated_images(generated_images, file_name)


def get_reconstructions(
    model: VAE,
    dataloader: DataLoader,
    *,
    root_path: Path,
    num_images: int = 8,
    file_name: Path = Path("reconstructions.png"),
    device: Optional[torch.device] = None,
) -> None:
    """Get reconstructions of images from the dataloader.

    Args:
        model (VAE): The VAE model to use.
        dataloader (DataLoader): DataLoader for the dataset.
        root_path (Path): The root path for saving the reconstructed images.
        num_images (int, optional): Number of images to reconstruct. Defaults to 8.
        file_name (Path, optional): File name to save the reconstructed images. Defaults to Path("reconstructions.png").
        device (Optional[torch.device], optional): The device to use for reconstruction. Defaults to None.
    """

    # Get the device of the model
    model_device = _get_device(model, device)
    model.eval()

    # Create directory to save reconstructed images if it doesn't exist
    root_path.mkdir(parents=True, exist_ok=True)

    # Create full file path to save the reconstructed images
    file_name = root_path / file_name
    file_name.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Get a batch of images from the dataloader
        images = next(iter(dataloader)).to(model_device)
        images = images[:num_images]

        # Reconstruct the images using the VAE
        reconstructed_images, _, _ = model(images)

    _plot_reconstructed_images(reconstructed_images, file_name)


def sample_from_latent_space(mu, L, device):
    """
    Sample latent vectors z from the multivariate Gaussian distribution
    defined by the encoder outputs (μ, L).

    Args:
        mu (torch.Tensor): Latent means of shape (N, D)
        L (torch.Tensor): Cholesky factors of the covariance matrices, shape (N, D, D)
        device (str): 'cpu' or 'cuda'

    Returns:
        torch.Tensor: Sampled latent vectors z of shape (N, D)
    """
    epsilon = torch.randn_like(mu).to(device)  # Standard Gaussian noise ε ~ N(0, I)
    z = mu + torch.bmm(L, epsilon.unsqueeze(-1)).squeeze(-1)  # Reparameterization: z = μ + L·ε
    return z


def gmm(model: VAE, train_dataloader: DataLoader, n_samples: int) -> None:
    """
    Fit a Gaussian Mixture Model (GMM) to the latent representations.

    Each photo in the dataset is treated as one Gaussian component in the GMM,
    parameterized by (μ_i, Σ_i = L_i L_iᵀ).

    Args:
        model (VAE): The trained VAE model.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        n_samples (int): Number of latent samples to generate from the GMM.

    Returns:
        torch.Tensor: Latent samples of shape (n_samples, D).
    """
    device = _get_device(model)
    model.to(device)

    mu, L = _get_latent_representations(model, train_dataloader)

    N, D = mu.shape

    # Convert tensors to NumPy arrays for sklearn compatibility
    mu_np = mu.detach().cpu().numpy()
    L_np = L.detach().cpu().numpy()

    # Compute full covariance matrices: Σ_i = L_i * L_iᵀ
    cov_np = np.matmul(L_np, np.transpose(L_np, (0, 2, 1)))

    # Add small regularization to the diagonal for numerical stability
    eps = 1e-6
    cov_np += np.eye(D)[None, :, :] * eps

    # Initialize Gaussian Mixture Model with N components (one per photo)
    gmm = GaussianMixture(n_components=N, covariance_type="full")
    gmm.weights_ = np.ones(N) / N  # Equal weights for all components
    gmm.means_ = mu_np  # Component means
    gmm.covariances_ = cov_np  # Full covariances per component

    model.gmm = gmm  # Store GMM in the model for later use


def _get_device(model: torch.nn.Module, override: Optional[torch.device] = None) -> torch.device:
    """Get the device of the model parameters or use the override if provided.

    Args:
        model (torch.nn.Module): The model to check.
        override (Optional[torch.device]): An optional device to use instead.

    Returns:
        torch.device: The device of the model parameters or the override.
    """
    # If an override device is provided, use it
    if override is not None:
        return override

    # Otherwise, get the device of the model parameters
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")  # Default to CPU if model has no parameters


def _balance_dataset(
    dataset: Dataset,
    target_length: int,
    *,
    seed: Optional[int],
) -> Dataset:
    """
    Balance the dataset to the target length by random sampling without replacement.

    Args:
        dataset (Dataset): The dataset to balance.
        target_length (int): The target length for the balanced dataset.
        seed (Optional[int]): A seed for the random number generator.

    Returns:
        Dataset: A balanced dataset.
    """

    # Get the current size of the dataset
    data_size = len(cast(Sized, dataset))

    # If the dataset is smaller than or equal to the target length,
    # return it as is (no balancing needed)
    if data_size <= target_length:
        return dataset

    # Create a random generator for reproducibility
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Randomly select indices to create a balanced subset
    indices = torch.randperm(data_size, generator=generator)[:target_length].tolist()

    # Return the balanced subset
    return Subset(dataset, indices)


def _plot_generated_images(generated_images: torch.Tensor, file_name: Path) -> None:
    """Plot and save generated images.

    Args:
        generated_images (torch.Tensor): Tensor of generated images.
        file_name (Path): Path to save the plotted images.
    """

    import matplotlib.pyplot as plt

    num_images = generated_images.size(0)

    # Plot the generated images
    _, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(generated_images[i].cpu().squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Generated {i + 1}")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    print(f"Generated images saved to {file_name}")


def _plot_reconstructed_images(reconstructed_images: torch.Tensor, file_name: Path) -> None:
    """Plot and save reconstructed images.

    Args:
        reconstructed_images (torch.Tensor): Tensor of reconstructed images.
        file_name (Path): Path to save the plotted images.
    """

    import matplotlib.pyplot as plt

    num_images = reconstructed_images.size(0)

    # Plot the reconstructed images
    _, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(reconstructed_images[i].cpu().squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Reconstructed {i + 1}")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    print(f"Reconstructed images saved to {file_name}")


def _get_latent_representations(model, train_dataloader):
    model_device = _get_device(model)

    model.to(model_device)
    model.eval()

    with torch.no_grad():
        all_mu = []
        all_L = []

        for images in train_dataloader:
            images = images.to(model_device)

            # Forward pass
            _, mu, L = model(images)

            all_mu.append(mu.cpu())
            all_L.append(L.cpu())

        all_mu_tensor = torch.cat(all_mu, dim=0)
        all_L_tensor = torch.cat(all_L, dim=0)

    return all_mu_tensor, all_L_tensor
