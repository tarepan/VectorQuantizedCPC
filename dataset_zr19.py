from pathlib import Path
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass
import pickle

from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset
from corpuspy.components.archive import hash_args, try_to_acquire_archive_contents, save_archive
from omegaconf import MISSING
import numpy as np
from numpy import load
import librosa

from ZR19 import ConfCorpus, ItemIdZR19, ZR19
from preprocess import ConfPreprocessing, process_to_mel_mu

import numpy.typing as npt
ND_FP32 = npt.NDArray[np.float32]
ND_LONG = npt.NDArray[np.int32]


def get_dataset_item_id_list(dir_dataset: Path) -> Path:
    """Get μ-law discrete waveform item path in dataset.
    """
    return dir_dataset / "id_list.pickle.bin"

def get_dataset_mulaw_path(dir_dataset: Path, item_id: ItemIdZR19) -> Path:
    """Get μ-law discrete waveform item path in dataset.
    """
    return dir_dataset / item_id.subtype / item_id.speaker / "mulaws" / f"{item_id.utterance_name}.mulaw.npy"


def get_dataset_mel_path(dir_dataset: Path, item_id: ItemIdZR19) -> Path:
    """Get mel-spec item path in dataset.
    """
    return dir_dataset / item_id.subtype / item_id.speaker / "mels" / f"{item_id.utterance_name}.mel.npy"


Datum_ZR19 = Tuple[LongTensor, FloatTensor, int]

@dataclass
class ConfDataset:
    """Configuration of dataset.
    Args:
        adress_data_root: Root adress of data
        clip_length_mel: Clipping length with mel frame unit.
        mel_stft_stride: hop length of mel-spectrogram STFT.
    """
    adress_data_root: Optional[str] = MISSING
    clip_length_mel: int = MISSING
    mel_stft_stride: int = MISSING
    corpus: ConfCorpus = ConfCorpus(mirror_root="${..adress_data_root}")
    preprocess: ConfPreprocessing = ConfPreprocessing(hop_length="${..mel_stft_stride}")

class ZR19MulawMelSpkDataset(Dataset[Datum_ZR19]):
    """Audio mu_law_wave/mel_spec/speaker_index dataset from ZR19 corpus.
    """
    def __init__(self, train: bool, conf: ConfDataset):
        """
        Args:
            train: train_dataset if True else validation/test_dataset.
            conf: Configuration of this dataset.
        """

        # Design Notes:
        #   Dataset archive name:
        #     Dataset contents differ based on argument,
        #     so archive should differ when arguments differ.
        #     It is guaranteed by name by argument hash.

        # Store parameters.
        self.conf = conf
        self._train = train

        self._corpus = ZR19(conf.corpus)
        arg_hash = hash_args(conf.preprocess.sr, conf.preprocess.bits, conf.preprocess.hop_length, conf.clip_length_mel)
        archive_name = f"{arg_hash}.zip"

        archive_root = conf.adress_data_root
        # Directory to which contents are extracted and archive is placed
        # if adress is not provided.
        local_root = Path(".")/"tmp"/"ZR19_mel_mulaw"
        
        # Archive: placed in given adress (conf) or default adress (local dataset directory)
        adress_archive_given = f"{archive_root}/datasets/ZR19/{archive_name}" if archive_root else None
        adress_archive_default = str(local_root/"archive"/archive_name)
        adress_archive = adress_archive_given or adress_archive_default

        # Contents: contents are extracted in local dataset directory
        self._path_contents = local_root/"contents"/arg_hash

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(adress_archive, self._path_contents)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents, adress_archive)
            print("Dataset contents was generated and archive was saved.")
        else:
            # Load item_id list
            with open(get_dataset_item_id_list(self._path_contents), mode='rb') as f:
                self._ids: List[ItemIdZR19] = pickle.load(f)
        self._speakers: List[str] = list(set(map(lambda item_id: item_id.speaker, self._ids)))

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus.get_contents()

        # Data selection based on audio length
        self._ids: List[ItemIdZR19] = [
            item_id for item_id in self._corpus.get_identities()
            if len(librosa.load(self._corpus.get_item_path(item_id), sr=self.conf.preprocess.sr)[0])
            >
            (self.conf.clip_length_mel + 2) * self.conf.mel_stft_stride
        ]
        # Save the item_id list
        path_id_list = get_dataset_item_id_list(self._path_contents)
        path_id_list.parent.mkdir(parents=True, exist_ok=True)
        with open(path_id_list, mode='wb') as f:
            pickle.dump(self._ids, f)

        print("Preprocessing...")
        for id in self._ids:
            path_i_wav = self._corpus.get_item_path(id)
            path_o_mulaw = get_dataset_mulaw_path(self._path_contents, id)
            path_o_mel = get_dataset_mel_path(self._path_contents, id)
            process_to_mel_mu(path_i_wav, path_o_mel, path_o_mulaw, self.conf.preprocess)
        print("Preprocessed.")

    def _load_datum(self, id: ItemIdZR19) -> Datum_ZR19:

        # (freq, T_mel)
        mel: ND_FP32 = load(get_dataset_mel_path(self._path_contents, id))
        # (T_mel * hop_length,)
        mulaw: ND_LONG = load(get_dataset_mulaw_path(self._path_contents, id))
        # Index ∈ [0, N_spk - 1]
        speaker = self._speakers.index(id.speaker)

        if self._train:
            # Time-directional random clipping
            # Waveform could be padded during STFT,
            # so waveform could a little short at last mel frame.
            start = random.randint(0, mel.shape[-1] - self.conf.clip_length_mel - 1 - 1)

            # Mel-spectrogram clipping
            start_mel = start
            end_mel = start + self.conf.clip_length_mel
            # (T_mel, freq) -> (clip_length_mel, freq)
            mel_clipped = mel[:, start_mel : end_mel]

            # Waveform clipping
            start_mulaw = self.conf.mel_stft_stride * start_mel
            end_mulaw = self.conf.mel_stft_stride * end_mel + 1
            # (T_mel * hop_length,) -> (clip_length_mel * hop_length,)
            mulaw_clipped = mulaw[start_mulaw : end_mulaw]

            # print(f"mulaw: {mulaw.shape}, cut: {start_mulaw}~{end_mulaw}={end_mulaw - start_mulaw}, mulaw_clipped: {mulaw_clipped.shape}")
            return LongTensor(mulaw_clipped), FloatTensor(mel_clipped), speaker
        else:
            return LongTensor(mulaw), FloatTensor(mel), speaker

    def __getitem__(self, n: int) -> Datum_ZR19:
        """Load n-th datum.
        Args:
            n : The index of the datum to be loaded
        """
        return self._load_datum(self._ids[n])

    def __len__(self) -> int:
        return len(self._ids)