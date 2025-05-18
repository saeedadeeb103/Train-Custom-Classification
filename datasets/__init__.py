from typing import List
from torch.utils.data import Dataset
import os

from .image_dataset import CustomDataset
from .audio_dataset import EmodbDataset
from .ctc_audio_dataclass import CTCEmodbDataset
from .TESS_Dataset import TESSRawWaveformDataset, MSPPodcastDataset

__dataset_mapper__ = {
    "image": CustomDataset,
    "emodb": EmodbDataset,
    "CTCemodb": CTCEmodbDataset,
    "TESSDataset": TESSRawWaveformDataset,
    "MSPPodcastDataset": MSPPodcastDataset
}

def list_datasets() -> List[str]:
    return sorted(__dataset_mapper__.keys())

def get_dataset_by_name(dataset: str, *args, **kwargs) -> Dataset:
    assert dataset in __dataset_mapper__, (
        f"Dataset '{dataset}' not found in the mapper. Available datasets: {list_datasets()}"
    )

    dataset_class = __dataset_mapper__[dataset]

    # Define required arguments for each dataset
    required_args = {
        "MSPPodcastDataset": ["labels_path", "audio_dir"],
        "TESSDataset": ["root_path"],
        "emodb": ["root_path"],
        "CTCemodb": ["root_path"],
        "image": ["root_path"],
    }

    # Check for required arguments
    dataset_required = required_args.get(dataset, [])
    missing_args = [
        arg for arg in dataset_required if arg not in kwargs and not args
    ]
    if missing_args:
        raise ValueError(
            f"Dataset '{dataset}' requires {dataset_required}, but {missing_args} are missing."
        )

    # Validate partition_path for MSPPodcastDataset if provided
    if dataset == "MSPPodcastDataset" and "partition_path" in kwargs:
        partition_path = kwargs["partition_path"]
        if partition_path and not os.path.exists(partition_path):
            raise ValueError(
                f"Provided 'partition_path' for MSPPodcastDataset does not exist: {partition_path}"
            )

    try:
        return dataset_class(*args, **kwargs)
    except TypeError as e:
        raise ValueError(
            f"Invalid arguments provided for dataset '{dataset}'. Error: {str(e)}"
        )
