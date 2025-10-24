from pathlib import Path
from typing import Any, Mapping

import torch.nn.functional as F
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from utils import load_config_from_yaml, seed_everything
from utils import training as training_utils
from vae import VAE
from vae.model import DEVICE


def train_vae(config: Mapping[str, Any], report=True) -> tuple[VAE, DataLoader, DataLoader]:
    """ "Train a VAE model based on the provided configuration.

    Args:
        config (Mapping[str, Any]): Configuration dictionary for training.
        report (bool, optional): Whether to report intermediate results. Defaults to True.

    Returns:
        tuple[VAE, DataLoader, DataLoader]: The trained model and the dataloaders.
    """

    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"

    report_callback = train.report if report else None

    model, train_dataloader, _, test_dataloader = training_utils.run_training_pipeline(
        config,
        data_dir=data_dir,
        output_dir=output_dir,
        report_callback=report_callback,
        device=DEVICE,
    )

    return model, test_dataloader, train_dataloader


def hyperparameter_tuning() -> None:
    """Perform hyperparameter tuning for the VAE model."""

    # Define the search space for hyperparameters
    search_space = {
        "lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
        "latent_dim": tune.grid_search([64, 128, 256]),
        "epochs": tune.grid_search([10, 50, 100]),
    }

    # Scheduler to stop bad trials early
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2,
    )

    # Set up the Tuner
    tuner = tune.Tuner(
        train_vae,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=1,  # Will run 27 trials due to grid search
        ),
        param_space=search_space,
    )

    # Run the hyperparameter tuning
    results = tuner.fit()

    # Get the best result
    best_result = results.get_best_result(metric="loss", mode="min")

    print("\n" * 2)
    print("-" * 50)
    print(" " * 15 + "Best Hyperparameters")
    print("-" * 50)
    print(f" * Best trial config: {best_result.config}")
    if best_result.metrics is not None and "loss" in best_result.metrics:
        print(f" * Best trial final validation loss: {best_result.metrics['loss']}")
    else:
        print(" * Best trial final validation loss: Not available")

    # You can now retrain the model with the best hyperparameters or load the best checkpoint
    # For example:
    # best_model = VAE(latent_dim=best_result.config["latent_dim"])
    # best_checkpoint = torch.load(best_result.checkpoint.to_air_checkpoint().path + "/checkpoint.pt")
    # best_model.load_state_dict(best_checkpoint["model_state"])
    print("-" * 50)
    print("\n")


if __name__ == "__main__":
    config_dir = Path(__file__).parent / "configs/local.yaml"
    config = load_config_from_yaml(config_dir, overrides=None)

    # Adjust paths to be absolute
    data_dir = Path(__file__).parent / config.data_dir
    output_dir = Path(__file__).parent / config.output_dir
    config = config.with_updates(data_dir=data_dir, output_dir=output_dir)

    config.ensure_directories()

    if config.seed:
        seed_everything(config.seed)

    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"

    model, test_loader, train_loader = train_vae(
        {
            "lr": config.learning_rate,
            "latent_dim": config.latent_dim,
            "epochs": config.epochs,
            "loss_function": F.mse_loss,
            "beta": config.beta,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "download": config.download,
            "n_components": config.n_components,
            "image_size": config.image_size,
        },
        False,
    )

    # hyperparameter_tuning()

    training_utils.generate_images(model, root_path=output_dir)
