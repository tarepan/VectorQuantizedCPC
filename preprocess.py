from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import librosa
import scipy
import numpy as np
from omegaconf import MISSING

import numpy.typing as npt
ND_FP32 = npt.NDArray[np.float32]
ND_LONG = npt.NDArray[np.int32]


# We could use librosa's preemphasis: [librosa.effects.preemphasis](https://librosa.org/doc/main/generated/librosa.effects.preemphasis.html)
def preemphasis(x: ND_FP32, preemph: float) -> ND_FP32:
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x: ND_FP32, mu: int) -> ND_LONG:
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y: ND_LONG, mu: int) -> ND_FP32:
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


@dataclass
class ConfPreprocessing:
    """Configuration of preprocessing.
    """
    sr: int = MISSING
    n_fft: int = MISSING
    n_mels: int = MISSING
    fmin: int = MISSING
    preemph: float = MISSING
    top_db: int = MISSING
    hop_length: int = MISSING
    win_length: int = MISSING
    bits: int = MISSING


def wave_to_mu_mel(wave: ND_FP32, conf: ConfPreprocessing) -> Tuple[ND_LONG, ND_FP32]:
    """
    Convert a waveform into μ-law waveform and mel spectrogram.

    Args:
        wave: Target waveform
        conf: Configuration of preprocessing
    """
    # Scale adjustment: [?, ?] -> (-1, +1)
    wave = wave / np.abs(wave).max() * 0.999

    # Preemphasis -> melspectrogram -> log-mel spec -> ? -> μ-law
    mel: ND_FP32 = librosa.feature.melspectrogram(preemphasis(wave, conf.preemph),
                                         sr=conf.sr,
                                         n_fft=conf.n_fft,
                                         n_mels=conf.n_mels,
                                         hop_length=conf.hop_length,
                                         win_length=conf.win_length,
                                         fmin=conf.fmin,
                                         power=1)
    logmel: ND_FP32 = librosa.amplitude_to_db(mel, top_db=conf.top_db)
    logmel: ND_FP32 = logmel / conf.top_db + 1.

    mulaw = mulaw_encode(wave, mu=2**conf.bits)

    return mulaw, logmel


def process_to_mel_mu(
    path_i_wav: Path,
    path_o_mel: Path,
    path_o_mulaw: Path,
    conf: ConfPreprocessing
) -> None:
    """
    Preprocess specified audio file into mel-spectrogram and μ-law waveform file.
    """

    # Load
    wave: ND_FP32 = librosa.load(path_i_wav, sr=conf.sr)[0]

    # Process
    mu_law, spec = wave_to_mu_mel(wave, conf)

    # Save
    path_o_mulaw.parent.mkdir(parents=True, exist_ok=True)
    path_o_mel.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_o_mulaw, mu_law)
    np.save(path_o_mel, spec)
