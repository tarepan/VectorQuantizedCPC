from ZR19 import ConfCorpus
from typing import List, NamedTuple
from pathlib import Path

from corpuspy.interface import AbstractCorpus
from corpuspy.helper.contents import get_contents
from corpuspy.helper.forward import forward_from_GDrive
import fsspec


# Subtype = Literal["parallel100"] # >=Python3.8
Subtype = str
subtypes = ["parallel100"]
Speaker = int

class ItemIdJVS(NamedTuple):
    """Identity of Japanese versatile speech / JVS corpus's item.

    Args:
        subtype: Sub-corpus name
        speaker: Speaker ID
        serial_num: Utterance Number
    """
    # Design Note: Audio Length
    #   ItemId do not contain audio length intentionally.
    #   'Effective' length differ in situation by situation.
    #   For example, native file length, non-zero length,
    #     dB-based length, Voice Activity region, etc...
    subtype: Subtype
    speaker: Speaker
    serial_num: int


class JVS(AbstractCorpus[ItemIdJVS]):
    """Zero Resource Speech Challenge 2019 corpus.

    Archive/contents handler of Zero Resource Speech Challenge 2019 corpus.
    Terminology:
        mirror: Mirror archive of the corpus
        contents: Contents extracted from archive
    """

    def __init__(self, conf: ConfCorpus) -> None:
        """Initiate JVS with archive options.
        """

        self.conf = conf

        # Equal to 1st layer directory name of original zip.
        self._corpus_name: str = f"jvs_ver1"
        archive_name = f"{self._corpus_name}.zip"

        self._origin_content_id = "19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt"

        mirror_root = conf.mirror_root
        # Directory to which contents are extracted, and mirror is placed if adress is not provided.
        local_root = Path("./data")

        # Mirror: placed in given adress (conf) or default adress (local corpus directory)
        adress_mirror_given = f"{mirror_root}/corpuses/JVS/{archive_name}" if mirror_root else None
        adress_mirror_default = str(local_root / "corpuses" / "JVS" / "archive" / archive_name)
        self._adress_mirror = adress_mirror_given or adress_mirror_default

        # Contents: contents are extracted in local corpus directory
        self._path_contents = local_root / "corpuses" / "JVS" / "contents"

    def get_contents(self) -> None:
        """Get corpus contents into local.
        """

        get_contents(self._adress_mirror, self._path_contents, self.conf.download, self.forward_from_origin)

    def forward_from_origin(self) -> None:
        """Forward original corpus archive to the mirror adress.
        """

        forward_from_GDrive(self._origin_content_id, self._adress_mirror, 3.29)

    def get_identities(self) -> List[ItemIdJVS]:
        """Get corpus item identities.

        Currently only 'parallel100' is provided.
        Returns:
            Full item identity list.
        """

        # Design notes:
        #   No contents dependency is intentional.
        #   Corpus handler can be used without corpus itself (e.g. Get item identities for a preprocessed dataset).
        #   Hard-coded identity list enable contents-independent identity acquisition.

        ids: List[ItemIdJVS] = [ItemIdJVS("parallel100", spk, utt) for utt in range(1, 101) for spk in range(1, 101)]
        return ids

    def get_item_path(self, id: ItemIdJVS) -> Path:
        """Get path of the item.

        Directory structure:
        jvs001/
            parallel100/
                wav24kHz16bit/
                    VOICEACTRESS100_001.wav
                    ...
                    VOICEACTRESS100_100.wav
            nonpara30/
            falset10/
            whisper10/
        ...
        jvs100/
        
        Args:
            id: Target item identity.
        Returns:
            Path of the specified item.
        """
        root = self._path_contents
        spk_dir = f"jvs{str(id.speaker).zfill(3)}"
        f_name = f"VOICEACTRESS100_{str(id.serial_num).zfill(3)}.wav"
        return root / self._corpus_name / spk_dir / "parallel100" / "wav24kHz16bit" / f_name


def forward_from_general(adress_from: str, forward_to: str) -> None:
    """Forward a file from the adress to specified adress.
    Forward any_adress -> any_adress through fsspec (e.g. local, S3, GCP).
    Args:
        adress_from: Forward origin adress.
        forward_to: Forward distination adress.
    """

    adress_from_with_cache = f"simplecache::{adress_from}"
    forward_to_with_cache = f"simplecache::{forward_to}"

    with fsspec.open(adress_from_with_cache, "rb") as origin:
        print("Forward: Reading from the adress...")
        archive = origin.read()
        print("Forward: Read.")

        print("Forward: Writing to the adress...")
        with fsspec.open(forward_to_with_cache, "wb") as destination:
            destination.write(archive)
        print("Forward: Written.")
