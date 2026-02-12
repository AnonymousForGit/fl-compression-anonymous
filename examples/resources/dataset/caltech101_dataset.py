import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from appfl.misc.data import (
    Dataset,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
)


def get_caltech101(
    num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs
):
    """
    Return the Caltech101 dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    :param partition_strategy: data partitioning strategy ("iid", "class_noniid", "dirichlet_noniid")
    """
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"
    # Define transforms for Caltech101 (resize to 224x224 for standard models)
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Download the full dataset
    full_dataset = torchvision.datasets.Caltech101(
        root=dir, 
        download=True, 
        transform=transform
    )

    # Split into train and test (80/20 split)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    
    train_data_raw, test_data_raw = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Obtain the test dataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        data, label = test_data_raw[idx]
        test_data_input.append(data.tolist())
        test_data_label.append(label)
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # Partition the training dataset
    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = dirichlet_noniid_partition(
            train_data_raw, num_clients, **kwargs
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    return train_datasets[client_id], test_dataset