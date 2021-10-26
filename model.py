from dataclasses import dataclass
import math

from omegaconf.omegaconf import MISSING
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from rnnms.networks.vocoder import ConfRNNMSVocoder, RNNMSVocoder


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
        # LN-ReLU-[FC-LN-ReLU]x4-FC
        self.encoder = nn.Sequential(
            nn.LayerNorm(conf.channels),
            nn.ReLU(True),
            #
            nn.Linear(conf.channels, conf.channels, bias=False),
            nn.LayerNorm(conf.channels),
            nn.ReLU(True),
            #
            nn.Linear(conf.channels, conf.channels, bias=False),
            nn.LayerNorm(conf.channels),
            nn.ReLU(True),
            #
            nn.Linear(conf.channels, conf.channels, bias=False),
            nn.LayerNorm(conf.channels),
            nn.ReLU(True),
            #
            nn.Linear(conf.channels, conf.channels, bias=False),
            nn.LayerNorm(conf.channels),
            nn.ReLU(True),
            #
            nn.Linear(conf.channels, conf.z_dim),
        )
        self.codebook = VQEmbeddingEMA(conf.n_embeddings, conf.z_dim)
        self.rnn = nn.LSTM(conf.z_dim, conf.c_dim, batch_first=True)

    def encode(self, mel):
        """Encode spectrogram.

        Returns:
            (z, c, indices): latent vector series, context vector series, latent index series
        """
        z = self.conv(mel)
        z = self.encoder(z.transpose(1, 2))
        z, indices = self.codebook.encode(z)
        c, _ = self.rnn(z)
        return z, c, indices

    def forward(self, mels):
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


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


@dataclass
class ConfVocoder:
    """
    Args:
        in_channels: Dimension of latent vector z (NOT codebook size)
        n_speakers: Number of speakers
        speaker_embedding_dim: Dimension of speaker embedding
        bits_mu_law: Depth of quantized bit
    """
    in_channels: int = MISSING
    n_speakers: int = MISSING
    speaker_embedding_dim: int = MISSING
    bits_mu_law: int = MISSING
    rnnms: ConfRNNMSVocoder = ConfRNNMSVocoder()

class Vocoder(nn.Module):
    """Independently-trained vocoder conditioned on discrete VQ-CPC output.
    
    Model is bidirectional_PreNet + WaveRNN (=RNN_MS).
    """
    def __init__(self, conf: ConfVocoder):
        """
        """
        super(Vocoder, self).__init__()

        # (discrete) Code/Id => (continuous) embedding space
        self.code_embedding = nn.Embedding(512, conf.in_channels)
        self.speaker_embedding = nn.Embedding(conf.n_speakers, conf.speaker_embedding_dim)

        # Input feature toward RNN_MS: Content_embedding + Speaker_embedding
        conf.rnnms.dim_i_feature = conf.in_channels + conf.speaker_embedding_dim
        self.rnnms = RNNMSVocoder(conf.rnnms)

    def forward(self, x: Tensor, z: Tensor, speaker: Tensor):
        """Forward a content representation sequence at once with teacher observation sequence for AR.
        
        Args:
            x: μ-law encoded observation sequence for AR teacher signal
            z: Index series of discrete content representation for conditioning
            speaker: Speaker ID (discrete value)
        
        Returns:
            Energy distribution of `bits` bit μ-law value
        """
        
        # Content embedding and upsampling
        z_embed = self.code_embedding(z)
        # (Batch, Time, Embed_z) => (Batch, Embed_z, 2*Time) => (Batch, 2*Time, Embed_z)
        z_embed_up: Tensor = F.interpolate(z_embed.transpose(1, 2), scale_factor=2).transpose(1, 2)
        # Speaker embedding and upsampling
        spk_embed: Tensor = self.speaker_embedding(speaker)
        # Time-directional copy (keep Batch/dim0 & Embed/dim2 by `-1` flag)
        # (Batch, Embed_spk) => (Batch, 1, Embed_spk) => (Batch, 2*Time, Embed_spk)
        spk_embed_up = spk_embed.unsqueeze(1).expand(-1, z_embed_up.size(1), -1)

        # Input to RNN_MS
        latent_series = torch.cat((z_embed_up, spk_embed_up), dim=-1)

        return self.rnnms(x, latent_series)

    def generate(self, z: Tensor, speaker: Tensor):
        """Generate utterances from a batch of (latent_code, speaker_index)
        """

        # Content embedding and upsampling
        z_embed = self.code_embedding(z)
        z_embed_up: Tensor = F.interpolate(z_embed.transpose(1, 2), scale_factor=2).transpose(1, 2)
        # Speaker embedding and upsampling
        spk_embed = self.speaker_embedding(speaker)
        spk_embed_up = spk_embed.unsqueeze(1).expand(-1, z_embed_up.size(1), -1)

        # Input to RNN_MS
        z = torch.cat((z_embed_up, spk_embed_up), dim=-1)

        return self.rnnms.generate(z)

@dataclass
class ConfModel:
    encoder: ConfEncoder = ConfEncoder()
    cpc: ConfCPC = ConfCPC()
    vocoder: ConfVocoder = ConfVocoder()
