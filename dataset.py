from pathlib import Path
import json
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class CPCDataset(Dataset):
    def __init__(self, root, n_sample_frames, n_utterances_per_speaker, hop_length, sr):
        """
        Args:
            root: Data root path
            n_sample_frames: Length of mel clip. Used for minimum audio length check and item clip.
            n_utterances_per_speaker: Number of single-speaker's utterances in a batch
            hop_length: STFT hop length
            sr: Audio sampling rate
        """
        self.root = Path(root)
        self.n_sample_frames = n_sample_frames
        self.n_utterances_per_speaker = n_utterances_per_speaker

        # with open(self.root / "speakers.json") as file:
        #     self.speakers = sorted(json.load(file))

        min_duration = n_sample_frames * hop_length / sr
        with open(self.root / "train.json") as file:
            metadata = json.load(file)
        metadata_by_speaker = dict()
        for _, _, duration, out_path in metadata:
            if duration > min_duration:
                # Path of preprocessed .npy
                out_path = Path(out_path)
                speaker = out_path.parent.stem
                metadata_by_speaker.setdefault(speaker, []).append(out_path)
        self.metadata = [
            (k, v) for k, v in metadata_by_speaker.items()
            # Minumum number of utterance of a speaker
            if len(v) >= n_utterances_per_speaker]

    def __len__(self):
        """
        Number of speakers
        """
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Returns:
            (Utterance, Freq, T_clipped) from random single speaker
        """
        # speaker_name, List[Path of utterance]
        speaker, paths = self.metadata[index]

        mels = list()
        paths: List[Path] = random.sample(paths, self.n_utterances_per_speaker)
        # Stack a clipped mel-spec of a utterance
        for path in paths:
            # mel::(Freq, Time)
            mel = np.load((self.root.parent / path).with_suffix(".mel.npy"))
            pos = random.randint(0, mel.shape[1] - self.n_sample_frames)
            mel = mel[:, pos:pos + self.n_sample_frames]
            mels.append(mel)
        mels = np.stack(mels)

        # (Utterance, Freq, T_clipped) from single speaker
        return torch.from_numpy(mels), self.speakers.index(speaker)
