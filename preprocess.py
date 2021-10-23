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


def process_wav(wav: np.ndarray, conf: ConfPreprocessing) -> (np.ndarray, np.ndarray):
    """
    Args:
        wav: Target waveform
        conf: Configuration of preprocessing
    """
    # Scale adjustment: [?, ?] -> (-1, +1)
    wav = wav / np.abs(wav).max() * 0.999

    # Preemphasis -> melspectrogram -> log-mel spec -> ? -> μ-law
    mel = librosa.feature.melspectrogram(preemphasis(wav, conf.preemph),
                                         sr=conf.sr,
                                         n_fft=conf.n_fft,
                                         n_mels=conf.n_mels,
                                         hop_length=conf.hop_length,
                                         win_length=conf.win_length,
                                         fmin=conf.fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=conf.top_db)
    logmel = logmel / conf.top_db + 1

    wav = mulaw_encode(wav, mu=2**conf.bits)

    return wav, logmel


def preprocess_dataset(cfg: ConfGlobal):

    # zerospeech/2019
    in_dir = Path(cfg.in_dir)
    # datasets/2019
    out_dir = Path("datasets") / cfg.dataset.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        # datasets/2019/english/{"train"|"test"}
        split_path = out_dir / cfg.dataset.language / split
        # `train.json` | `test.json`
        with open(split_path.with_suffix(".json")) as file:
            metadata = json.load(file)
            for in_path, start, duration, out_path in metadata:
                # in_path
                # start: (maybe) volume>0 start time - BUT ALL DATA IS START=0
                # duration: (maybe) effective duration
                # out_path

                # Load uttered part of a .wav file
                # zerospeech/2019/{in_path}
                # e.g. wav_path = "zerospeech/2019/english/train/unit/S015_0361841101"
                wav_path = in_dir / in_path
                wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=conf.preprocessing.sr,
                                    offset=start, duration=duration)
                mu_law, spec = process_wav(wav, cfg.preprocessing)
                # Output:
                # datasets/2019/{out_path}
                # e.g. out_path = "datasets/2019/english/train/S015/S015_0361841101"
                out_path = out_dir / out_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path.with_suffix(".wav.npy"), mu_law)
                np.save(out_path.with_suffix(".mel.npy"), spec)
        print(f"{split} preprocessed.")


if __name__ == "__main__":
    conf = load_conf()
    preprocess_dataset(conf)
