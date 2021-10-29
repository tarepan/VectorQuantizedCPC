from typing import List, Tuple
from dataclasses import dataclass

from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from omegaconf import MISSING

from model import Encoder 
from network_vocoder import ConfVocoder, Vocoder


@dataclass
class ConfOptim:
    """Configuration of optimizer.
    Args:
        learning_rate: Optimizer learning rate
        sched_milestones: LR shaduler decay epoch/step
        sched_gamma: LR shaduler decay rate
    """
    learning_rate: float = MISSING
    sched_milestones: List[int] = MISSING
    sched_gamma: float = MISSING

@dataclass
class ConfVocoderModel:
    """Configuration of VocoderSystem.

    Args:
        sampling_rate: Output audio sampling rate
    """
    sampling_rate: int = MISSING
    n_speakers: int = MISSING
    network: ConfVocoder = ConfVocoder(n_speakers="${..n_speakers}")
    optim: ConfOptim = ConfOptim()

class VocoderModel(pl.LightningModule):
    """Vocoder model of VQ-CPC/RNNMS.
    """

    def __init__(self, conf: ConfVocoderModel, encoder: Encoder):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.model = Vocoder(conf.network)
        self.encoder = encoder
        self.spk_increment: int = 5

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], _):
        """Supervised learning.
        """

        audio_series, mel_series, speaker = batch

        # This could be done in preprocessing for Vocoder.
        # No optimizer register -> automatic no_grad (maybe)
        _, _, indice_series = self.encoder.encode(mel_series)

        # Vocoding -> CE loss
        energy_series: Tensor = self.model(audio_series[:, :-1], indice_series, speaker)
        loss = F.cross_entropy(energy_series.transpose(1, 2), audio_series[:, 1:])

        self.log('loss', loss)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        """full length needed & padding is not good (long white seems to be not good for RNN) => cannot batch (batch=1)
        """

        _, mel_series, speaker = batch
        _, _, indice_series = self.encoder.encode(mel_series)
        wave_linear_reconst = self.model.generate(indice_series, speaker)
        spk_target_vc = (speaker + self.spk_increment) % self.conf.n_speakers
        wave_linear_vc = self.model.generate(indice_series, spk_target_vc)

        idx_spk_src = speaker.item()
        idx_spk_tgt = spk_target_vc.item()
        # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)
        # add_audio(tag: str, snd_tensor: Tensor(1, L), global_step: Optional[int] = None, sample_rate: int = 44100)
        self.logger.experiment.add_audio(
            f"spk_{idx_spk_src}",
            wave_linear_reconst,
            global_step=self.global_step,
            sample_rate=self.conf.sampling_rate,
        )
        self.logger.experiment.add_audio(
            f"{idx_spk_src}_to_{idx_spk_tgt}",
            wave_linear_vc,
            global_step=self.global_step,
            sample_rate=self.conf.sampling_rate,
        )
        return {"val_loss": 0}

    def configure_optimizers(self):
        """Set up a optimizer
        """
        conf = self.conf.optim

        optim = Adam(self.model.parameters(), lr=conf.learning_rate)
        sched = {
            "scheduler": MultiStepLR(optim, conf.sched_milestones, conf.sched_gamma),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }
