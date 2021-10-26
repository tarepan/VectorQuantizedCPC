from enum import Enum
from typing import Optional
import os
from datetime import timedelta
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cloud_io import get_filesystem
from omegaconf import MISSING

from model import Encoder
from vocoder import ConfVocoderModel, VocoderModel


class Profiler(Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


@dataclass
class ConfTrainer:
    """Configuration of trainer.
    Args:
        max_epochs: Number of maximum training epoch
        val_interval_epoch: Interval epoch between validation
        profiler: Profiler setting
    """
    max_epochs: int = MISSING
    val_interval_epoch: int = MISSING
    profiler: Optional[Profiler] = MISSING

@dataclass
class ConfCkptLog:
    """Configuration of checkpointing and logging.
    """
    dir_root: str = MISSING
    name_exp: str  = MISSING
    name_version: str  = MISSING

@dataclass
class ConfTrainVocoder:
    """Configuration of train.
    """
    model: ConfVocoderModel = ConfVocoderModel()
    trainer: ConfTrainer = ConfTrainer()
    ckpt_log: ConfCkptLog = ConfCkptLog()


def train_vocoder(conf: ConfTrainVocoder, datamodule: LightningDataModule, encoder: Encoder) -> None:
    """Train vocoder
    """

    ckpt_and_logging = CheckpointAndLogging(
        conf.ckpt_log.dir_root,
        conf.ckpt_log.name_exp,
        conf.ckpt_log.name_version
    )

    # setup
    model = VocoderModel(conf.model, encoder)

    # Save checkpoint as `last.ckpt` every 15 minutes.
    ckpt_cb = ModelCheckpoint(
        train_time_interval=timedelta(minutes=15),
        save_last=True,
        save_top_k=0,
    )

    trainer = pl.Trainer(
        gradient_clip_val=1,
        gpus=1 if torch.cuda.is_available() else 0,
        auto_select_gpus=True,
        precision=16,
        max_epochs=conf.trainer.max_epochs,
        check_val_every_n_epoch=conf.trainer.val_interval_epoch,
        # logging/checkpointing
        resume_from_checkpoint=ckpt_and_logging.resume_from_checkpoint,
        default_root_dir=ckpt_and_logging.default_root_dir,
        logger=pl_loggers.TensorBoardLogger(
            ckpt_and_logging.save_dir, ckpt_and_logging.name, ckpt_and_logging.version
        ),
        callbacks=[ckpt_cb],
        # reload_dataloaders_every_epoch=True,
        profiler=conf.trainer.profiler,
        progress_bar_refresh_rate=30
    )

    # training
    trainer.fit(model, datamodule=datamodule)


class CheckpointAndLogging:
    """Generate path of checkpoint & logging.
    {dir_root}/
        {name_exp}/
            {name_version}/
                checkpoints/
                    {name_ckpt} # PyTorch-Lightning Checkpoint. Resume from here.
                hparams.yaml
                events.out.tfevents.{xxxxyyyyzzzz} # TensorBoard log file.
    """

    # [PL's Trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-class-api)
    default_root_dir: Optional[str]
    resume_from_checkpoint: Optional[str]
    # [PL's TensorBoardLogger](https://pytorch-lightning.readthedocs.io/en/stable/logging.html#tensorboard)
    save_dir: str
    name: str
    version: str
    # [PL's ModelCheckpoint](https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)
    # dirpath: Implicitly inferred from `default_root_dir`, `name` and `version` by PyTorch-Lightning

    def __init__(
        self,
        dir_root: str,
        name_exp: str = "default",
        name_version: str = "version_-1",
        name_ckpt: str = "last.ckpt",
    ) -> None:

        path_ckpt = os.path.join(dir_root, name_exp, name_version, "checkpoints", name_ckpt)

        # PL's Trainer
        self.default_root_dir = dir_root
        self.resume_from_checkpoint = path_ckpt if get_filesystem(path_ckpt).exists(path_ckpt) else None

        # TB's TensorBoardLogger
        self.save_dir = dir_root
        self.name = name_exp
        self.version = name_version
