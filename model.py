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
        """
        Args:
            mels (Batch=Spk*Utt, Freq, T_clipped): Mel-spectrogram
        Returns:
            (Acoustic Unit, Context, vq_loss, perplexity)
        """
        # (Batch, Freq, Time) => (Batch, Feature, Time)
        z = self.conv(mels)
        # SegFC: (Batch, Feature, Time) => (Batch, Time, Feature) => (Batch, Time, Feature)
        z = self.encoder(z.transpose(1, 2))
        # ? (Batch, Time, Feature) => (Batch, Time, Feature)
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
        """
        Args:
            x (Batch, Time, Feature): Feature time-series
        """

        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)
        # Quantization?
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
    def __init__(
        self,
        n_speakers_per_batch: int,
        n_utterances_per_speaker: int,
        n_prediction_steps: int,
        n_negatives: int,
        z_dim: int,
        c_dim: int
    ):
        """
        Args:
            n_speakers_per_batch: Number of speaker per batch
            n_utterances_per_speaker: Number of utterance per speaker in a batch
            n_prediction_steps: Number of CPC prediction step
            n_negatives: Number of contrastive negatives
            z_dim: Dimension of acoustic unit
            c_dim: Dimension of context
        """
        super(CPCLoss, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps // 2
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        # Linear predictor for each future steps
        # todo: Check `n_prediction_steps`, half of predictors not used?
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z: Tensor, c: Tensor):
        """
        Execute Contrastive Predictive Coding.

        All uttrances are used for anchor, except for last n_pred frame (no positives).
        It results in `Spk * Utt * (MelClip-n_pred) * n_pred` positives per batch.
        Args:
            z (Spk*Utt, Time, FeatureZ): Acoustic Unit
            c (Spk*Utt, Time, FeatureC): Context
        """
        # L = Time - self.n_prediction_steps
        length = z.size(1) - self.n_prediction_steps

        # (Spk*Utt, Time, FeatureZ) => (Spk, Utt, Time, FeatureZ)
        z = z.reshape(
            self.n_speakers_per_batch,
            self.n_utterances_per_speaker,
            -1,
            self.z_dim
        )
        # Context which have enough number of future z (k steps)
        # (Spk*Utt, Time, FeatureC) => (Spk*Utt, Time=L, FeatureC)
        c = c[:, :-self.n_prediction_steps, :]

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            # ========================== CPC@t+k ==========================

            # ============ Positive series ============
            # Positives for +k future predictions
            # (Spk, Utt, Time=L, FeatureZ)
            z_shift = z[:, :, k:length + k, :]
            # ============ /Positive series ===========

            # ============= Anchor series =============
            # Anchor: t+k Future Prediction of t=[0, L-1]
            # `Wc`: linear transformation `W` of context `c`
            # (Spk*Utt, Time=L, FeatureC) => (Spk*Utt, Time=L, FeatureZ) => (Spk, Utt, Time=L, FeatureZ)
            Wc = self.predictors[k-1](c).view(
                self.n_speakers_per_batch,
                self.n_utterances_per_speaker,
                -1,
                self.z_dim
            )
            # ============= /Anchor series ============

            # ============ Negatives series ===========
            # Sample negatives by array indexing.
            # Tricky access through 'multi-dimensional arrays with multi-dimensional index arrays'
            # [numpy](https://numpy.org/doc/stable/user/basics.indexing.html#indexing-multi-dimensional-arrays)

            # Speaker Index: *within-speaker* negative sampling
            #   (Spk,   1,   1, 1) reshaped from [0, 1, ..., n_spk-1]
            speaker_index = torch.arange(self.n_speakers_per_batch, device=z.device).view(-1, 1, 1, 1)

            # Utterance index: "negative examples drawn from other utterances" from original paper
            #   Each utterances work as a positive, and one positive has `Neg` negatives.
            #     Current implementation could draw 'themselfs', is it negligible practically?
            #       False 'super hard negative' could influence strongly, is't it?
            #   (  1, Utt, Neg, 1) random sampled from [0, n_utt)
            utt_index = torch.randint(
                0, self.n_utterances_per_speaker,
                size=(self.n_utterances_per_speaker, self.n_negatives),
                device=z.device
            ).view(1, self.n_utterances_per_speaker, self.n_negatives, 1)

            # Seq index: Position in a utterance for a negative sample
            #   (Spk, Utt, Neg, L) random sampled from [1, length) + some shift
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
            # Add [0, 1, ..., length-1] to last dim
            seq_index += torch.arange(length, device=z.device)
            # %length: some value out of range by `[1, length) + 0~(length-1)`
            seq_index = torch.remainder(seq_index, length)

            # Negatives      (Spk, Utt, Neg, Time, FeatureZ)
            #   Generate new tensor by indexing.
            #   Neg dim is introduced and FeatureZ is kept with partial indexing.
            #   Below is index list.
            #       z_shift        (Spk, Utt, Time=L, FeatureZ)
            #       speaker_index  (Spk,   1,   1, 1): Reshaped [0, 1, ..., Spk-1]
            #       utt_index      (1,   Utt, Neg, 1): Random [0, n_utt)
            #       seq_index:     (Spk, Utt, Neg, L): Random [1, length) then shift 0~L-1
            z_negatives = z_shift[speaker_index, utt_index, seq_index, :]
            # ============ /Negatives series ==========

            # Concatenate positive series and negatives series
            # (Spk, Utt, 1+Neg, Time=L, FeatureZ)
            zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2)

            # Dot-product Similarities
            # (Spk, Utt, 1+Neg, Time=L, FeatureZ) => (Spk, Utt, 1+Neg, Time=L) => (Spk*Utt, 1+Neg, Time=L)
            f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1).view(
                self.n_speakers_per_batch * self.n_utterances_per_speaker,
                self.n_negatives + 1,
                -1
            )

            # Index of positive, it is always `0` because of cat(pos, negs)
            # (Spk*Utt, L) = 0
            labels = torch.zeros(
                self.n_speakers_per_batch * self.n_utterances_per_speaker, length,
                dtype=torch.long, device=z.device
            )

            # Similarities => Final InfoNCE Loss
            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels
            accuracy = torch.mean(accuracy.float())

            # Stack results of t+k
            losses.append(loss)
            accuracies.append(accuracy.item())
            # ========================== /CPC@t+k =========================

        loss = torch.stack(losses).mean()
        return loss, accuracies


@dataclass
class ConfModel:
    encoder: ConfEncoder = ConfEncoder()
    cpc: ConfCPC = ConfCPC()
