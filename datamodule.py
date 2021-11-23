from typing import Optional
from os import cpu_count
from dataclasses import dataclass

import torch
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
from omegaconf import MISSING
from speechcorpusy.interface import ConfCorpus


@dataclass
class ConfLoader:
    """Configuration of data loader.
    Args:
        batch_size: Number of datum in a batch
        num_workers: Number of data loader worker
        pin_memory: Use data loader pin_memory
    """
    batch_size: int = MISSING
    num_workers: Optional[int] = MISSING
    pin_memory: Optional[bool] = MISSING

class DataLoaderPerformance:
    """PyTorch DataLoader performance configs.
    All attributes which affect performance of [torch.utils.data.DataLoader][^DataLoader] @ v1.6.0
    [^DataLoader]:https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    num_workers: int
    pin_memory: bool

    def __init__(self, num_workers: Optional[int], pin_memory: Optional[bool]) -> None:
        """Default: num_workers == cpu_count & pin_memory == True
        """

        # Design Note:
        #   Current performance is for single GPU training.
        #   cpu_count() is not appropriate under the multi-GPU condition.

        if num_workers is None:
            c = cpu_count()
            num_workers = c if c is not None else 0
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory if pin_memory is not None else True


@dataclass
class ConfData:
    """Configuration of data.
    """
    adress_archive_root = MISSING
    corpus: ConfCorpus = ConfCorpus(
        root="{..adress_archive_root}"
    )
    dataset: ConfDataset = ConfDataset()
    loader: ConfLoader = ConfLoader()

class ZR19enDataModule(LightningDataModule):
    """PL-DataModule of LJSpeech wave & mel.
    """
    def __init__(self, conf: ConfData):
        # Design Notes: Dataset independent
        #   DataModule's responsibility is about data loader.
        #   Dataset handle corpus, preprocessing and datum sampling.

        super().__init__()
        self.conf = conf
        self.loader_perf = DataLoaderPerformance(conf.loader.num_workers, conf.loader.pin_memory)

    def prepare_data(self) -> None:
        """Prepare data in dataset.
        """

        ZR19MulawMelSpkDataset(True, self.conf.dataset)

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets and batch sizes.
        """

        if stage == "fit" or stage is None:
            dataset_full_train = ZR19MulawMelSpkDataset(True, self.conf.dataset)
            dataset_full_not_train = ZR19MulawMelSpkDataset(False, self.conf.dataset)

            n_full = len(dataset_full_train)
            # N-3 utterances, fixed-length.
            self.dataset_train, _ = random_split(
                dataset_full_train, [n_full - 3, 3], generator=torch.Generator().manual_seed(42)
            )
            # 3 utterances, variable-length. Generator enable consistent split.
            _, self.dataset_val = random_split(
                dataset_full_not_train, [n_full - 3, 3], generator=torch.Generator().manual_seed(42)
            )
            self.batch_size_val = 1

        if stage == "test" or stage is None:
            self.dataset_test = ZR19MulawMelSpkDataset(False, conf=self.conf.dataset)
            self.batch_size_test = self.conf.loader.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.conf.loader.batch_size,
            shuffle=True,
            num_workers=self.loader_perf.num_workers,
            pin_memory=self.loader_perf.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_val,
            num_workers=self.loader_perf.num_workers,
            pin_memory=self.loader_perf.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_test,
            num_workers=self.loader_perf.num_workers,
            pin_memory=self.loader_perf.pin_memory,
        )


class JVSjaDataModule(LightningDataModule):
    """PL-DataModule of JVS wave & mel.
    """
    def __init__(self, conf: ConfData):
        # Design Notes: Dataset independent
        #   DataModule's responsibility is about data loader.
        #   Dataset handle corpus, preprocessing and datum sampling.

        super().__init__()
        self.conf = conf
        self.loader_perf = DataLoaderPerformance(conf.loader.num_workers, conf.loader.pin_memory)

    def prepare_data(self) -> None:
        """Prepare data in dataset.
        """

        JVSMulawMelSpkDataset(True, self.conf.dataset)

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets and batch sizes.
        """

        if stage == "fit" or stage is None:
            dataset_full_train = JVSMulawMelSpkDataset(True, conf=self.conf.dataset)
            dataset_full_not_train = JVSMulawMelSpkDataset(False, conf=self.conf.dataset)

            n_full = len(dataset_full_train)
            # N-3 utterances, fixed-length.
            self.dataset_train, _ = random_split(
                dataset_full_train, [n_full - 3, 3], generator=torch.Generator().manual_seed(42)
            )
            # 3 utterances, variable-length. Generator enable consistent split.
            _, self.dataset_val = random_split(
                dataset_full_not_train, [n_full - 3, 3], generator=torch.Generator().manual_seed(42)
            )
            self.batch_size_val = 1

        if stage == "test" or stage is None:
            self.dataset_test = JVSMulawMelSpkDataset(False, conf=self.conf.dataset)
            self.batch_size_test = self.conf.loader.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.conf.loader.batch_size,
            shuffle=True,
            num_workers=self.loader_perf.num_workers,
            pin_memory=self.loader_perf.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_val,
            num_workers=self.loader_perf.num_workers,
            pin_memory=self.loader_perf.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_test,
            num_workers=self.loader_perf.num_workers,
            pin_memory=self.loader_perf.pin_memory,
        )