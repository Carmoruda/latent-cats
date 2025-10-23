import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
"""Image file extensions supported by the dataset."""


class VAEDataset(Dataset):
    """Dataset of images for VAE.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset class.

    Attributes:
        url (str): URL to download the dataset from Kaggle.
        dataset_path (Path): Path to the dataset directory.
        transform (transforms.Compose, optional): Transformations to apply to the images.
    """

    def __init__(
        self,
        url: str,
        dataset_path: str,
        extracted_folder: str = "default",
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
        delete_extracted: bool = True,
    ) -> None:
        """Initialize the dataset. Assigns URL, dataset path, optional
        transformations, downloads the dataset and lists image files.

        Args:
            url (string): URL to the Kaggle dataset download.
            dataset_path (string): Directory for the dataset.
            extracted_folder (str, optional): The folder name inside the extracted zip where images are located. Defaults to "default".
            transform (transforms.Compose, optional): Optional transform to be applied on a sample. Defaults to None.
            download (bool, optional): Whether to download the dataset. Defaults to False.
            delete_extracted (bool, optional): Whether to delete the extracted files after moving images.
        """

        self.url = url
        self.dataset_path = Path(dataset_path)
        self.transform = transform

        # Download and extract the dataset if specified
        if download:
            self.download(extracted_folder, delete_extracted)

        # Create dataset directory if it doesn't exist
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # List all image files in the dataset directory and store their names if they are images
        self.images = sorted(
            file.name
            for file in self.dataset_path.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        number = len(self.images)

        if number <= 0:
            raise ValueError(
                f"No images found in '{self.dataset_path}'. Please ensure that the download and extraction were successful and that there are images in the folder."
            )

        return number

    def __getitem__(self, index) -> Image.Image:
        """Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Image.Image: The requested image.
        """
        # Handle case where index is a tensor
        # (A tensor is a multi-dimensional array used in PyTorch for data representation)
        if torch.is_tensor(index):
            index = index.item()

        # Ensure index is an integer
        index = int(index)

        # Get the image name for the sample
        img_name = self.images[index]
        img_path = self.dataset_path / img_name

        # Load the image and ensure the file handle is closed promptly
        with Image.open(img_path) as image:
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
            else:
                image = image.copy()  # Detach image from file handle

        return image

    def download(self, extracted_folder, delete_extracted) -> None:
        """Download the dataset from the Kaggle URL and extract it.

        Args:
            extracted_folder (str): Path to the folder where the dataset will be extracted.
            delete_extracted (bool, optional): Whether to delete the extracted folder after moving images. Defaults to True.

        Raises:
            RuntimeError: If the dataset cannot be downloaded or extracted.
        """

        print("Downloading and extracting dataset...")

        output_zip = Path("dataset.zip")
        extract_to = Path(".")

        # Download the zip file
        response = requests.get(self.url, stream=True, timeout=30)

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Failed to download dataset from {self.url}. HTTP Error: {e}"
            ) from e

        # Write the zip file to disk
        with output_zip.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        # Extract the zip file
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        # Move dataset to the final location
        extracted_root = (extract_to / extracted_folder).resolve()
        data_dst = self.dataset_path.resolve()

        # Search for the folder containing images inside 'extracted'
        data_src: Optional[Path] = None
        for root, _, files in os.walk(extracted_root):
            if any(file.lower().endswith(IMAGE_EXTENSIONS) for file in files):
                data_src = Path(root)
                break

        if data_src is None:
            raise RuntimeError(
                f"No image files found in the extracted dataset at '{extracted_root}'."
            )

        # Create destination directory if it doesn't exist
        data_dst.mkdir(parents=True, exist_ok=True)

        # Move all images to data_dst, de-duplicating names when needed
        for src_file in data_src.iterdir():
            if not src_file.is_file():
                continue

            dst_file = data_dst / src_file.name
            if dst_file.exists():
                dst_file = self._resolve_duplicate_path(dst_file)
                print(f"File '{src_file.name}' already exists. Renaming to '{dst_file.name}'.")

            shutil.move(str(src_file), str(dst_file))

        # Delete the extracted folder and zip file to clean up
        if delete_extracted:
            shutil.rmtree(extracted_root, ignore_errors=True)
            output_zip.unlink(missing_ok=True)

        print(f"Dataset downloaded and extracted successfully to {data_dst}")

    @staticmethod
    def _resolve_duplicate_path(destination: Path) -> Path:
        """Generate a non-conflicting path by appending a numeric suffix.

        Args:
            destination (Path): The original file path.
        """

        counter = 1
        stem = destination.stem
        suffix = destination.suffix
        parent = destination.parent

        candidate = destination

        while candidate.exists():
            candidate = parent / f"{stem}_{counter}{suffix}"
            counter += 1

        return candidate
