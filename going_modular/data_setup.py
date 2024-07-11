
"""
Contains functionality for creating PyTorch DataLoader's for
image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

# Function to create dataloaders
def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns them into
    PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        transform (transforms.Compose): torchvision transforms to perform on training and testing data.
        batch_size (int): Number of samples per batch in each of the DataLoaders.
        num_workers (int, optional): Number of workers per DataLoader. Default is os.cpu_count().

    Returns:
        tuple: A tuple of (train_dataloader, test_dataloader, class_names).
               Where class_names is a list of the target classes.

    Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir="path/to/train_dir",
            test_dir="path/to/test_dir",
            transform=some_transform,
            batch_size=32,
            num_workers=4
        )
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(
        root=train_dir,       # target folder of images
        transform=transform,  # transforms to perform on data (images)
        target_transform=None # transforms to perform on labels (if necessary)
    )

    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )

    # Get class names as a list
    class_names = train_data.classes

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size, # how many samples per batch?
        num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
        shuffle=True,           # shuffle the data?
        pin_memory=True         # stores data in GPU instead of fetching from CPU every time
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,          # don't usually need to shuffle testing data
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
