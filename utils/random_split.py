from typing import List
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from utils.helper_functions import normalize_ratios

def stratified_random_split(ds: torch.utils.data.Dataset, parts: List[float], targets: List[int]) -> List[torch.utils.data.Dataset]:
    """
    Perform a stratified random split on the dataset.

    Args:
        ds: PyTorch dataset to split.
        parts: List of proportions that sum to 1.
        targets: List of labels corresponding to dataset samples.

    Returns:
        List of PyTorch datasets corresponding to the splits.
    """
    total_length = len(ds)

    # Normalize ratios
    parts = normalize_ratios(parts)

    lengths = list(map(lambda p: int(p * total_length), parts))
    left_over = total_length - sum(lengths)
    lengths[0] += left_over  # Adjust first split to account for leftover

    indices = list(range(total_length))
    train_indices, temp_indices, _, temp_targets = train_test_split(
        indices, targets, test_size=(1 - parts[0]), stratify=targets, random_state=42
    )
    val_size = parts[1] / (parts[1] + parts[2])
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_targets, test_size=(1 - val_size), stratify=temp_targets, random_state=42
    )

    return [Subset(ds, train_indices), Subset(ds, val_indices), Subset(ds, test_indices)]
