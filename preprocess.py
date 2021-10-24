from pathlib import Path
import json
from dataclasses import dataclass

import librosa
import scipy
import numpy as np
from omegaconf import MISSING

from config import load_conf, ConfGlobal


# We could use librosa's preemphasis: [librosa.effects.preemphasis](https://librosa.org/doc/main/generated/librosa.effects.preemphasis.html)
def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
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

def wave_to_mu_mel(wave: np.ndarray, conf: ConfPreprocessing) -> (np.ndarray, np.ndarray):
    """
    Convert a waveform into μ-law waveform and mel spectrogram.

    Args:
        wave: Target waveform
        conf: Configuration of preprocessing
    """
    # Scale adjustment: [?, ?] -> (-1, +1)
    wave = wave / np.abs(wave).max() * 0.999

    # Preemphasis -> melspectrogram -> log-mel spec -> ? -> μ-law
    mel = librosa.feature.melspectrogram(preemphasis(wave, conf.preemph),
                                         sr=conf.sr,
                                         n_fft=conf.n_fft,
                                         n_mels=conf.n_mels,
                                         hop_length=conf.hop_length,
                                         win_length=conf.win_length,
                                         fmin=conf.fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=conf.top_db)
    logmel = logmel / conf.top_db + 1

    wave = mulaw_encode(wave, mu=2**conf.bits)

    return wave, logmel


def process_to_mel_mu(path_i_wav: Path, path_o_mel: Path, path_o_mulaw: Path, cfg: ConfGlobal):
    """
    Preprocess specified audio file into mel-spectrogram and μ-law waveform file.
    """

    # datasets/2019
    out_dir = Path("datasets") / cfg.dataset.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        # datasets/2019/english/{"train"|"test"}
        split_path = out_dir / cfg.dataset.language / split
        # `train.json` | `test.json`
        with open(split_path.with_suffix(".json")) as file:
            metadata = json.load(file)
            wave, _ = librosa.load(path_i_wav, sr=conf.preprocessing.sr)
            mu_law, spec = wave_to_mu_mel(wave, cfg.preprocessing)
            np.save(path_o_mulaw, mu_law)
            np.save(path_o_mel, spec)


if __name__ == "__main__":
    conf = load_conf()
    preprocess_dataset(conf)
