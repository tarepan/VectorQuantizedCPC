from dataclasses import dataclass

from omegaconf.omegaconf import MISSING
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from rnnms.networks.vocoder import ConfRNNMSVocoder, RNNMSVocoder


@dataclass
class ConfVocoder:
    """
    Args:
        size_i_codebook: Size of input discrete codebook
        dim_i_embedding: Dimension of embedded input
        n_speakers: Number of speakers
        dim_speaker_embedding: Dimension of speaker embedding
    """
    size_i_codebook: int = MISSING
    dim_i_embedding: int = MISSING
    n_speakers: int = MISSING
    dim_speaker_embedding: int = MISSING
    rnnms: ConfRNNMSVocoder = ConfRNNMSVocoder()

class Vocoder(nn.Module):
    """Independently-trained vocoder conditioned on discrete VQ-CPC output.
    
    Network is bidirectional_PreNet + WaveRNN (=RNN_MS).
    """
    def __init__(self, conf: ConfVocoder):
        """
        """
        super(Vocoder, self).__init__()

        # (discrete) latent_code/speaker_id => (continuous) embedding space
        self.code_embedding = nn.Embedding(conf.size_i_codebook, conf.dim_i_embedding)
        self.speaker_embedding = nn.Embedding(conf.n_speakers, conf.dim_speaker_embedding)
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

        latent_series = torch.cat((z_embed_up, spk_embed_up), dim=-1)
        return self.rnnms(x, latent_series)

    def generate(self, z: Tensor, speaker: Tensor):
        """Generate utterances from a batch of (latent_code, speaker_index)
        """

        # Content/Speaker embedding and upsampling
        z_embed = self.code_embedding(z)
        z_embed_up: Tensor = F.interpolate(z_embed.transpose(1, 2), scale_factor=2).transpose(1, 2)
        spk_embed = self.speaker_embedding(speaker)
        spk_embed_up = spk_embed.unsqueeze(1).expand(-1, z_embed_up.size(1), -1)
        z_spk_series = torch.cat((z_embed_up, spk_embed_up), dim=-1)
        return self.rnnms.generate(z_spk_series)
