from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle
import random

from torch import FloatTensor, from_numpy
from torch.utils.data import Dataset
from corpuspy.components.archive import hash_args, try_to_acquire_archive_contents, save_archive
from omegaconf import MISSING
import numpy as np
from numpy import load
from tqdm import tqdm
from scipy.io import wavfile

from ZR19 import ConfCorpus, ItemIdZR19, ZR19
from JVS import ItemIdJVS, JVS
from preprocess import ConfPreprocessing, process_to_mel

import numpy.typing as npt
ND_FP32 = npt.NDArray[np.float32]
ND_LONG = npt.NDArray[np.int32]


def dataset_adress(
    root_adress: Optional[str],
    corpus_name: str,
    dataset_type: str,
    preprocess_args,
    ) -> Tuple[str, Path]:
    """Path of dataset archive file and contents directory.

    Args:
        root_adress: Optional[str],
        corpus_name: str,
        dataset_type: str,
        preprocess_args,
    Returns: [archive file adress, contents directory path]
    """
    # Design Notes:
    #   Why not `Path` object? -> Archive adress could be remote url
    #
    # Original Data (corpus) / Prepared Data (dataset) / Transformation (preprocss)
    #   If use different original data, everything change.
    #   Original item can be transformed into different type of data.
    #   Even if data type is same, value could be changed by processing parameters.
    #
    # Directory structure:
    #     datasets/{corpus_name}/{dataset_type}/
    #         archive/{preprocess_args}.zip
    #         contents/{preprocess_args}/{actual_data_here}

    # Contents: Placed under default local directory
    contents_root = local_root = "./tmp"
    # Archive: Placed under given adress or default local directory
    archive_root = root_adress or local_root

    rel_dataset = f"datasets/{corpus_name}/{dataset_type}"
    archive_file = f"{archive_root}/{rel_dataset}/archive/{preprocess_args}.zip"
    contents_dir = f"{contents_root}/{rel_dataset}/contents/{preprocess_args}"
    return archive_file, contents_dir


def len_wav(path: Path, target_sr: int) -> int:
    """Length of waveform in given sampling rate.
    
    1sec 16kHz => L=16000
    1sec 24kHz w/ 16kHz resampling => L=16000
    """
    sr_raw, wave = wavfile.read(path)
    return int(len(wave) / (sr_raw/target_sr))


def get_path_dataset_ids_per_spk(dir_dataset: Path) -> Path:
    """Get path of item per speaker dict of the dataset.
    """
    return dir_dataset / "ids_per_spk.pickle.bin"

def get_dataset_mel_path(dir_dataset: Path, item_id: ItemIdZR19) -> Path:
    """Get mel-spec item path in dataset.
    """
    return dir_dataset / item_id.subtype / f"{item_id.speaker}" / "mels" / f"{item_id.utterance_name}.mel.npy"


Datum_ZR19CPC = Tuple[FloatTensor, int]

@dataclass
class ConfCPCDataset:
    """Configuration of CPC dataset.
    Args:
        adress_data_root: Root adress of data
        clip_length_mel: Clipping length with mel frame unit.
        mel_stft_stride: hop length of mel-spectrogram STFT.
        n_utterances_per_speaker: Number of utterances per speaker for CPC.
    """
    adress_data_root: Optional[str] = MISSING
    clip_length_mel: int = MISSING
    mel_stft_stride: int = MISSING
    n_utterances_per_speaker: int = MISSING
    corpus: ConfCorpus = ConfCorpus(mirror_root="${..adress_data_root}")
    preprocess: ConfPreprocessing = ConfPreprocessing(hop_length="${..mel_stft_stride}")

class ZR19CPCMelSpkDataset(Dataset[Datum_ZR19CPC]):
    """Audio mel_spec/speaker_index CPC dataset from ZR19 corpus.
    """
    def __init__(self, train: bool, conf: ConfCPCDataset):
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
        arg_hash = hash_args(conf.preprocess.sr, conf.preprocess.bits, conf.preprocess.hop_length, conf.clip_length_mel)
        adress_archive, self._path_contents = dataset_adress(conf.adress_data_root, "ZR19", "cpc_mel_spk", arg_hash)

        # Corpus
        self._corpus = ZR19(conf.corpus)

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(adress_archive, self._path_contents)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents, adress_archive)
            print("Dataset contents was generated and archive was saved.")
        else:
            # Load ids_per_spk dict
            with open(get_path_dataset_ids_per_spk(self._path_contents), mode='rb') as f:
                self._ids_per_spk: Dict[str, List[ItemIdZR19]] = pickle.load(f)
        self._speakers: List[str] = list(self._ids_per_spk.keys())

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus.get_contents()

        # Selection: Audio length => utterance per speaker
        ## Based on audio length
        ids_enough_long: List[ItemIdZR19] = [
            item_id for item_id in self._corpus.get_identities()
            if len_wav(self._corpus.get_item_path(item_id), self.conf.preprocess.sr)
            >
            self.conf.clip_length_mel * self.conf.mel_stft_stride
        ]
        ## Based on number of utterances per speaker
        ids_per_spk_full: Dict[str, List[ItemIdZR19]] = {}
        for item_id in ids_enough_long:
            ids_per_spk_full.setdefault(item_id.speaker, []).append(item_id)
        self._ids_per_spk: Dict[str, List[ItemIdZR19]] = {
            spk: utts for spk, utts in ids_per_spk_full.items()
            if len(utts) >= self.conf.n_utterances_per_speaker
        }

        # Save the ids_per_spk dict
        path_ids_per_spk = get_path_dataset_ids_per_spk(self._path_contents)
        path_ids_per_spk.parent.mkdir(parents=True, exist_ok=True)
        with open(path_ids_per_spk, mode='wb') as f:
            pickle.dump(self._ids_per_spk, f)

        print("Preprocessing...")
        for item_id in tqdm(sum(list(self._ids_per_spk.values()), [])):
            path_i_wav = self._corpus.get_item_path(item_id)
            path_o_mel = get_dataset_mel_path(self._path_contents, item_id)
            process_to_mel(path_i_wav, path_o_mel, self.conf.preprocess)
        print("Preprocessed.")

    def _load_datum(self, spk_id: str) -> Datum_ZR19CPC:
        """
        Returns:
            (FloatTensor[Utt, Freq, T_clipped], Spk: int) from random single speaker
        """

        spk_idx = self._speakers.index(spk_id)
        # Random sampled ids
        uttr_ids = self._ids_per_spk[spk_id]
        item_ids: List[ItemIdZR19] = random.sample(uttr_ids, self.conf.n_utterances_per_speaker)

        if self._train:
            # Stack a clipped mel-spec of utterances from single speaker
            mels: List[ND_FP32] = []
            for item_id in item_ids:
                # Time-directional clipping: (freq, T_mel) -> (freq, clip_length_mel)
                mel: ND_FP32 = load(get_dataset_mel_path(self._path_contents, item_id))
                start = random.randint(0, mel.shape[-1] - self.conf.clip_length_mel)
                mels.append(mel[:, start : start + self.conf.clip_length_mel])
            return from_numpy(np.stack(mels)).float(), spk_idx
        else:
            mel: ND_FP32 = load(get_dataset_mel_path(self._path_contents, item_ids[0]))
            return from_numpy(np.stack([mel])).float(), spk_idx

    def __getitem__(self, n: int) -> Datum_ZR19CPC:
        """Load n-th speaker's data.
        Args:
            n: The index of the speaker to be loaded
        """
        return self._load_datum(self._speakers[n])

    def __len__(self) -> int:
        """Number of speakers
        """
        return len(self._speakers)


def get_dataset_mel_path_JVS(dir_dataset: Path, item_id: ItemIdJVS) -> Path:
    """Get mel-spec item path in dataset.
    """
    return dir_dataset / item_id.subtype / f"{item_id.speaker}" / "mels" / f"{item_id.serial_num}.mel.npy"

Datum_JVSCPC = Tuple[FloatTensor, int]

class JVSCPCMelSpkDataset(Dataset[Datum_JVSCPC]):
    """Audio mel_spec/speaker_index CPC dataset from JVS corpus.
    """
    def __init__(self, train: bool, conf: ConfCPCDataset):
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
        arg_hash = hash_args(conf.preprocess.sr, conf.preprocess.bits, conf.preprocess.hop_length, conf.clip_length_mel)
        adress_archive, self._path_contents = dataset_adress(conf.adress_data_root, "JVS", "cpc_mel_spk", arg_hash)

        # Corpus
        self._corpus = JVS(conf.corpus)

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(adress_archive, self._path_contents)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents, adress_archive)
            print("Dataset contents was generated and archive was saved.")
        else:
            # Load ids_per_spk dict
            with open(get_path_dataset_ids_per_spk(self._path_contents), mode='rb') as f:
                self._ids_per_spk: Dict[int, List[ItemIdJVS]] = pickle.load(f)
        self._speakers: List[int] = list(self._ids_per_spk.keys())

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus.get_contents()

        # Selection: Audio length => utterance per speaker
        ## Based on audio length
        ids_enough_long: List[ItemIdJVS] = [
            item_id for item_id in self._corpus.get_identities()
            if len_wav(self._corpus.get_item_path(item_id), self.conf.preprocess.sr)
            >
            self.conf.clip_length_mel * self.conf.mel_stft_stride
        ]
        ## Based on number of utterances per speaker
        ids_per_spk_full: Dict[int, List[ItemIdJVS]] = {}
        for item_id in ids_enough_long:
            ids_per_spk_full.setdefault(item_id.speaker, []).append(item_id)
        self._ids_per_spk: Dict[int, List[ItemIdJVS]] = {
            spk: utts for spk, utts in ids_per_spk_full.items()
            if len(utts) >= self.conf.n_utterances_per_speaker
        }

        # Save the ids_per_spk dict
        path_ids_per_spk = get_path_dataset_ids_per_spk(self._path_contents)
        path_ids_per_spk.parent.mkdir(parents=True, exist_ok=True)
        with open(path_ids_per_spk, mode='wb') as f:
            pickle.dump(self._ids_per_spk, f)

        print("Preprocessing...")
        for item_id in tqdm(sum(list(self._ids_per_spk.values()), [])):
            path_i_wav = self._corpus.get_item_path(item_id)
            path_o_mel = get_dataset_mel_path_JVS(self._path_contents, item_id)
            process_to_mel(path_i_wav, path_o_mel, self.conf.preprocess)
        print("Preprocessed.")

    def _load_datum(self, spk_id: int) -> Datum_JVSCPC:
        """
        Returns:
            (FloatTensor[Utt, Freq, T_clipped], Spk: int) from random single speaker
        """

        spk_idx = self._speakers.index(spk_id)
        # Random sampled ids
        uttr_ids = self._ids_per_spk[spk_id]
        item_ids: List[ItemIdJVS] = random.sample(uttr_ids, self.conf.n_utterances_per_speaker)

        if self._train:
            # Stack a clipped mel-spec of utterances from single speaker
            mels: List[ND_FP32] = []
            for item_id in item_ids:
                # Time-directional clipping: (freq, T_mel) -> (freq, clip_length_mel)
                mel: ND_FP32 = load(get_dataset_mel_path_JVS(self._path_contents, item_id))
                start = random.randint(0, mel.shape[-1] - self.conf.clip_length_mel)
                mels.append(mel[:, start : start + self.conf.clip_length_mel])
            return from_numpy(np.stack(mels)).float(), spk_idx
        else:
            mel: ND_FP32 = load(get_dataset_mel_path_JVS(self._path_contents, item_ids[0]))
            return from_numpy(np.stack([mel])).float(), spk_idx

    def __getitem__(self, n: int) -> Datum_ZR19CPC:
        """Load n-th speaker's data.
        Args:
            n: The index of the speaker to be loaded
        """
        return self._load_datum(self._speakers[n])

    def __len__(self) -> int:
        """Number of speakers
        """
        return len(self._speakers)
