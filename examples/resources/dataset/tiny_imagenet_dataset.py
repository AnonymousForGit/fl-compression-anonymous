import os
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import zipfile

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset

from appfl.misc.data import plot_distribution


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TINY_IMAGENET_DEFAULT_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


class TinyImageNetDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.classes = self._load_classes()
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.classes)}
        self.samples, self.targets = self._build_samples()

    def _load_classes(self) -> List[str]:
        wnids_file = self.root / "wnids.txt"
        if wnids_file.exists():
            with open(wnids_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        train_dir = self.root / "train"
        return sorted(
            [p.name for p in train_dir.iterdir() if p.is_dir()]
        )

    def _build_samples(self) -> Tuple[List[Path], List[int]]:
        if self.split == "train":
            return self._build_train_samples()
        if self.split == "val":
            return self._build_val_samples()
        raise ValueError(f"Unsupported split: {self.split}")

    def _build_train_samples(self) -> Tuple[List[Path], List[int]]:
        samples: List[Path] = []
        targets: List[int] = []
        train_dir = self.root / "train"

        for wnid in self.classes:
            class_img_dir = train_dir / wnid / "images"
            if not class_img_dir.exists():
                continue
            for img_path in sorted(class_img_dir.glob("*.JPEG")):
                samples.append(img_path)
                targets.append(self.class_to_idx[wnid])
        return samples, targets

    def _build_val_samples(self) -> Tuple[List[Path], List[int]]:
        samples: List[Path] = []
        targets: List[int] = []
        val_dir = self.root / "val"
        ann_file = val_dir / "val_annotations.txt"
        img_dir = val_dir / "images"

        if not ann_file.exists():
            raise FileNotFoundError(f"Missing Tiny-ImageNet val annotation file: {ann_file}")

        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) < 2:
                    continue
                img_name, wnid = fields[0], fields[1]
                if wnid not in self.class_to_idx:
                    continue
                img_path = img_dir / img_name
                if img_path.exists():
                    samples.append(img_path)
                    targets.append(self.class_to_idx[wnid])

        return samples, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.targets[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _download_tiny_imagenet(
    dataset_dir: Path, download_url: str, keep_archive: bool = False
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / "tiny-imagenet-200.zip"
    extracted_root = dataset_dir / "tiny-imagenet-200"

    if (extracted_root / "train").exists() and (extracted_root / "val").exists():
        return extracted_root

    print(f"Downloading Tiny-ImageNet from {download_url} to {archive_path} ...")
    urllib.request.urlretrieve(download_url, archive_path)

    print(f"Extracting {archive_path} ...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    if not keep_archive and archive_path.exists():
        archive_path.unlink()

    if not (extracted_root / "train").exists() or not (extracted_root / "val").exists():
        raise FileNotFoundError(
            f"Downloaded archive extracted, but expected train/val not found under {extracted_root}."
        )

    return extracted_root


def _resolve_tiny_imagenet_root(
    dataset_dir: str,
    auto_download: bool = True,
    download_url: str = TINY_IMAGENET_DEFAULT_URL,
    keep_archive: bool = False,
) -> Path:
    provided_root = Path(dataset_dir)
    repo_root = Path(__file__).resolve().parents[3]
    auto_candidates = [
        provided_root,
        provided_root / "tiny-imagenet-200",
        repo_root / "examples" / "datasets" / "RawData",
        repo_root / "examples" / "datasets" / "RawData" / "tiny-imagenet-200",
        repo_root / "datasets" / "RawData",
        repo_root / "datasets" / "RawData" / "tiny-imagenet-200",
    ]

    checked = []
    for candidate in auto_candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if (candidate / "train").exists() and (candidate / "val").exists():
            return candidate

    if auto_download:
        try:
            return _download_tiny_imagenet(
                dataset_dir=provided_root.resolve(),
                download_url=download_url,
                keep_archive=keep_archive,
            )
        except Exception as e:
            checked_paths = "\n".join(f"- {path}" for path in dict.fromkeys(checked))
            raise FileNotFoundError(
                "Tiny-ImageNet auto-download failed.\n"
                f"Reason: {e}\n"
                "Checked paths:\n"
                f"{checked_paths}\n"
                "You can manually place the dataset in one of the paths above, "
                "or set `dataset_kwargs.download_url` / `dataset_kwargs.dataset_dir`."
            ) from e

    checked_paths = "\n".join(f"- {path}" for path in dict.fromkeys(checked))
    raise FileNotFoundError(
        "Tiny-ImageNet not found. Checked paths:\n"
        f"{checked_paths}\n"
        "Expected directory layout: <root>/train and <root>/val "
        "(or <root>/tiny-imagenet-200/train and val)."
    )


def _iid_partition_subset(
    train_dataset: TinyImageNetDataset, num_clients: int, seed: int
) -> List[Subset]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [Subset(train_dataset, split.tolist()) for split in splits]


def _class_noniid_partition_subset(
    train_dataset: TinyImageNetDataset,
    num_clients: int,
    classes_per_client: int,
    visualization: bool,
    output_dirname: str,
    output_filename: str,
    seed: int,
) -> List[Subset]:
    rng = np.random.default_rng(seed)
    labels = np.array(train_dataset.targets, dtype=np.int64)
    classes = np.array(sorted(set(labels.tolist())), dtype=np.int64)
    num_classes = len(classes)
    classes_per_client = max(1, min(classes_per_client, num_classes))

    label_indices: Dict[int, np.ndarray] = {}
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        label_indices[int(cls)] = cls_indices

    client_classes: List[set] = []
    for _ in range(num_clients):
        sampled = rng.choice(classes, size=classes_per_client, replace=False)
        client_classes.append(set(int(c) for c in sampled.tolist()))

    # Ensure every class appears in at least one client.
    for cls in classes:
        cls_int = int(cls)
        if not any(cls_int in assigned for assigned in client_classes):
            client_classes[int(rng.integers(0, num_clients))].add(cls_int)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    sample_matrix = np.zeros((num_classes, num_clients), dtype=np.int64)
    class_to_row = {int(cls): row for row, cls in enumerate(classes.tolist())}

    for cls in classes:
        cls_int = int(cls)
        assigned_clients = [
            cid for cid, assigned in enumerate(client_classes) if cls_int in assigned
        ]
        cls_splits = np.array_split(label_indices[cls_int], len(assigned_clients))
        for cid, cls_part in zip(assigned_clients, cls_splits):
            cls_indices = cls_part.tolist()
            client_indices[cid].extend(cls_indices)
            sample_matrix[class_to_row[cls_int], cid] = len(cls_indices)

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    if visualization:
        classes_samples = [len(label_indices[int(cls)]) for cls in classes]
        if len(classes_samples) > 20:
            print(
                "Skipping data distribution plot for Tiny-ImageNet: "
                "plot_distribution currently supports up to 20 classes."
            )
        else:
            plot_distribution(
                num_clients=num_clients,
                classes_samples=classes_samples,
                sample_matrix=sample_matrix,
                output_dirname=output_dirname,
                output_filename=output_filename,
            )

    return [Subset(train_dataset, indices) for indices in client_indices]


def _dirichlet_noniid_partition_subset(
    train_dataset: TinyImageNetDataset,
    num_clients: int,
    alpha: float,
    visualization: bool,
    output_dirname: str,
    output_filename: str,
    seed: int,
) -> List[Subset]:
    rng = np.random.default_rng(seed)
    labels = np.array(train_dataset.targets, dtype=np.int64)
    classes = np.array(sorted(set(labels.tolist())), dtype=np.int64)
    num_classes = len(classes)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    sample_matrix = np.zeros((num_classes, num_clients), dtype=np.int64)

    for row, cls in enumerate(classes):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = np.floor(proportions * len(cls_indices)).astype(int)
        counts[-1] = len(cls_indices) - np.sum(counts[:-1])

        start = 0
        for cid, count in enumerate(counts.tolist()):
            end = start + count
            if count > 0:
                client_indices[cid].extend(cls_indices[start:end].tolist())
            sample_matrix[row, cid] = count
            start = end

    # Prevent empty client datasets to avoid trainer runtime failures.
    sizes = [len(indices) for indices in client_indices]
    largest_client = int(np.argmax(sizes))
    for cid, size in enumerate(sizes):
        if size == 0 and len(client_indices[largest_client]) > 1:
            moved = client_indices[largest_client].pop()
            client_indices[cid].append(moved)

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    if visualization:
        classes_samples = [int(np.sum(labels == cls)) for cls in classes]
        if len(classes_samples) > 20:
            print(
                "Skipping data distribution plot for Tiny-ImageNet: "
                "plot_distribution currently supports up to 20 classes."
            )
        else:
            plot_distribution(
                num_clients=num_clients,
                classes_samples=classes_samples,
                sample_matrix=sample_matrix,
                output_dirname=output_dirname,
                output_filename=output_filename,
            )

    return [Subset(train_dataset, indices) for indices in client_indices]


def get_tiny_imagenet(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    **kwargs,
):
    """
    Return Tiny-ImageNet train/val datasets for a given client.
    """
    dataset_dir = kwargs.get(
        "dataset_dir", os.path.join(os.getcwd(), "datasets", "RawData")
    )
    dataset_root = _resolve_tiny_imagenet_root(
        dataset_dir=dataset_dir,
        auto_download=bool(kwargs.get("auto_download", True)),
        download_url=kwargs.get("download_url", TINY_IMAGENET_DEFAULT_URL),
        keep_archive=bool(kwargs.get("keep_archive", False)),
    )
    seed = int(kwargs.get("seed", 42))

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_data_raw = TinyImageNetDataset(
        root=str(dataset_root), split="train", transform=transform
    )
    val_dataset = TinyImageNetDataset(
        root=str(dataset_root), split="val", transform=transform
    )

    if partition_strategy == "iid":
        train_datasets = _iid_partition_subset(train_data_raw, num_clients, seed)
    elif partition_strategy == "class_noniid":
        train_datasets = _class_noniid_partition_subset(
            train_dataset=train_data_raw,
            num_clients=num_clients,
            classes_per_client=int(kwargs.get("classes_per_client", 20)),
            visualization=bool(kwargs.get("visualization", False)),
            output_dirname=kwargs.get("output_dirname", "./output"),
            output_filename=kwargs.get("output_filename", "visualization.pdf"),
            seed=seed,
        )
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = _dirichlet_noniid_partition_subset(
            train_dataset=train_data_raw,
            num_clients=num_clients,
            alpha=float(kwargs.get("alpha", 0.5)),
            visualization=bool(kwargs.get("visualization", False)),
            output_dirname=kwargs.get("output_dirname", "./output"),
            output_filename=kwargs.get("output_filename", "visualization.pdf"),
            seed=seed,
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    if not (0 <= client_id < len(train_datasets)):
        raise ValueError(
            f"client_id {client_id} out of range for {len(train_datasets)} clients."
        )

    return train_datasets[client_id], val_dataset
