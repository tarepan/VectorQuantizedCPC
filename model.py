from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Tuple

from omegaconf.omegaconf import MISSING
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F


def repeat_modules(atom_gen: Callable[[], List[nn.Module]], n_repeat: int) -> List[nn.Module]:
    return list(chain.from_iterable([atom_gen() for _ in range(0, n_repeat)]))


@dataclass
class ConfEncoder:
    """
    Args:
        in_channels
        channels
        n_embeddings
        z_dim
        c_dim
    """
    in_channels: int = MISSING
    channels: int = MISSING
    n_embeddings: int = MISSING
    z_dim: int = MISSING
    c_dim: int = MISSING

class Encoder(nn.Module):
    def __init__(self, conf: ConfEncoder):
        """Encode spectrogram to discrete latent representation.
        
        Model: Spec-Conv1d/k4s2-LN-ReLU-[FC-LN-ReLU]x4-FC-VQ + LSTM
        
        Args:
        """
        super(Encoder, self).__init__()
        # Conv1d/k4s2
        self.conv = nn.Conv1d(conf.in_channels, conf.channels, 4, 2, 1, bias=False)
        # Segmental FC layer: LN-ReLU-[FC-LN-ReLU]x4-FC
        # Change name break PyTorch checkpoint. Be careful. `seg_fc` is better name in future.
        self.encoder = nn.Sequential(
            nn.LayerNorm(conf.channels),
            nn.ReLU(True),
            *repeat_modules(lambda : [
                nn.Linear(conf.channels, conf.channels, bias=False),
                nn.LayerNorm(conf.channels),
                nn.ReLU(True)
            ], 4),
            nn.Linear(conf.channels, conf.z_dim),
        )
        self.codebook = VQEmbeddingEMA(conf.n_embeddings, conf.z_dim)
        self.rnn = nn.LSTM(conf.z_dim, conf.c_dim, batch_first=True)

    def encode(self, mel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode spectrogram.

        Returns:
            (z, c, indices): latent vector series, context vector series, latent index series
        """
        z = self.conv(mel)
        # [Batch, Feature, Time] => [Batch, Time, Feature] => (FC in Feature dim)
        z = self.encoder(z.transpose(1, 2))
        z, indices = self.codebook.encode(z)
        c, _ = self.rnn(z)
        return z, c, indices

    def forward(self, mels: Tensor):
        z = self.conv(mels)
        z = self.encoder(z.transpose(1, 2))
        z, loss, perplexity = self.codebook(z)
        c, _ = self.rnn(z)
        return z, c, loss, perplexity


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


@dataclass
class ConfCPC:
    n_prediction_steps: int = MISSING
    n_speakers_per_batch: int = MISSING
    n_utterances_per_speaker: int = MISSING
    n_negatives: int = MISSING
    z_dim: int = MISSING
    c_dim: int = MISSING

class CPCLoss(nn.Module):
    def __init__(self, n_speakers_per_batch, n_utterances_per_speaker, n_prediction_steps, n_negatives, z_dim, c_dim):
        super(CPCLoss, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps // 2
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c):
        length = z.size(1) - self.n_prediction_steps

        z = z.reshape(
            self.n_speakers_per_batch,
            self.n_utterances_per_speaker,
            -1,
            self.z_dim
        )
        c = c[:, :-self.n_prediction_steps, :]

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, :, k:length + k, :]

            Wc = self.predictors[k-1](c)
            Wc = Wc.view(
                self.n_speakers_per_batch,
                self.n_utterances_per_speaker,
                -1,
                self.z_dim
            )

            batch_index = torch.randint(
                0, self.n_utterances_per_speaker,
                size=(
                    self.n_utterances_per_speaker,
                    self.n_negatives
                ),
                device=z.device
            )
            batch_index = batch_index.view(
                1, self.n_utterances_per_speaker, self.n_negatives, 1
            )

            seq_index = torch.randint(
                1, length,
                size=(
                    self.n_speakers_per_batch,
                    self.n_utterances_per_speaker,
                    self.n_negatives,
                    length
                ),
                device=z.device
            )
            seq_index += torch.arange(length, device=z.device)
            seq_index = torch.remainder(seq_index, length)

            speaker_index = torch.arange(self.n_speakers_per_batch, device=z.device)
            speaker_index = speaker_index.view(-1, 1, 1, 1)

            z_negatives = z_shift[speaker_index, batch_index, seq_index, :]

            zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2)

            f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1)
            f = f.view(
                self.n_speakers_per_batch * self.n_utterances_per_speaker,
                self.n_negatives + 1,
                -1
            )

            labels = torch.zeros(
                self.n_speakers_per_batch * self.n_utterances_per_speaker, length,
                dtype=torch.long, device=z.device
            )

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies


@dataclass
class ConfModel:
    encoder: ConfEncoder = ConfEncoder()
    cpc: ConfCPC = ConfCPC()
