import hydra
from hydra import utils
from pathlib import Path
import librosa
import scipy
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass

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


def process_wav(wav_path, out_path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    """
    Args:
        wav_path: Path of target waveform without `.wav` suffix
        out_path: Path of output file without suffix, used for several outputs
        sr=160000
        preemph=0.97
        n_fft=2048
        n_mels=80
        hop_length=160
        win_length=400
        fmin=50
        top_db=80
        bits=8
        offset=0.0: Audio load offset, used only by `librosa.load`
        duration=None: Audio load duration, used only by `librosa.load`
    """
    # Load uttered part of a .wav file
    wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=sr,
                          offset=offset, duration=duration)
    # Scale adjustment: [?, ?] -> (-1, +1)
    wav = wav / np.abs(wav).max() * 0.999

    # Preemphasis -> melspectrogram -> log-mel spec -> ? -> μ-law
    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    wav = mulaw_encode(wav, mu=2**bits)

    # Output:
    # xxxx/
    #   name_of_wave.wav.npy (wav)
    #   name_of_wave.mel.npy (logmel)
    np.save(out_path.with_suffix(".wav.npy"), wav)
    np.save(out_path.with_suffix(".mel.npy"), logmel)

    # Returns used for reporting
    return out_path, logmel.shape[-1]


@dataclass
class ConfPreprocessing:
    """Configuration of preprocessing.
    """
    sr: int = 16000
    n_fft: int = 2048
    n_mels: int = 80
    fmin: int = 50
    preemph: int = 0.97
    top_db: int = 80
    hop_length: int = 160
    win_length: int = 400
    bits: int = 8
@hydra.main(config_path="config/preprocessing.yaml")
def preprocess_dataset(cfg):
    conf_preprocessing = ConfPreprocessing()

    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    # datasets/2019
    out_dir = Path(utils.to_absolute_path("datasets")) / str(cfg.dataset.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    for split in ["train", "test"]:
        futures = []
        # datasets/2019/english/{"train"|"test"}
        split_path = out_dir / cfg.dataset.language / split
        with open(split_path.with_suffix(".json")) as file:
            metadata = json.load(file)
            for in_path, start, duration, out_path in metadata:
                # in_path
                # start: Used only as `offset` in `preprocess_wav`
                # duration: Used only as `duration` in `preprocess_wav`
                # out_path

                # Specify pathes from dataset path and matadata
                wav_path = in_dir / in_path
                out_path = out_dir / out_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                futures.append(executor.submit(
                    partial(process_wav, wav_path, out_path, **cfg.preprocessing,
                            offset=start, duration=duration)))

        results = [future.result() for future in tqdm(futures)]

        print(f"{split} preprocessed.")

if __name__ == "__main__":
    preprocess_dataset()
