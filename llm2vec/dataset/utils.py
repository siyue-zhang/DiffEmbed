from ..dataset import E5Data
from ..dataset import E5Mix
from ..dataset import Wiki1M
from ..dataset import MSMARCO
from ..dataset import ReasonIR
from ..dataset import E5Custom

def load_dataset(dataset_name, split="validation", file_path=None, **kwargs):
    """
    Loads a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Split of the dataset to load.
        file_path (str): Path to the dataset file.
    """

    dataset_mapping = {
        "E5": E5Data,
        "Wiki1M": Wiki1M,
        "E5Mix": E5Mix,
        "MSMARCO": MSMARCO,
        "ReasonIR": ReasonIR,
        "E5Custom": E5Custom
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")

    if split not in ["train", "validation", "test"]:
        raise NotImplementedError(f"Split {split} not supported.")

    return dataset_mapping[dataset_name](split=split, file_path=file_path, **kwargs)
