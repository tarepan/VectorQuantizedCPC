from typing import List, NamedTuple, Optional
from pathlib import Path
from dataclasses import dataclass

from corpuspy.interface import AbstractCorpus
from corpuspy.helper.contents import get_contents
from omegaconf import MISSING
import fsspec

from zr19_items_unit import utterances_unit
from zr19_items_voice import utterances_voice


"""
'Zero Resource Speech Challenge 2019 corpus'

This corpus contains 4x2 sub-corpuses.
x2 are Development(en)/Surprise(id), 4 are below.

|          name          |    directory      |                                                                              |
|:----------------------:|:-----------------:|:----------------------------------------------------------------------------:|
| `Train Voice`          | `train/voice/`    | 2 spk (Development) / 1 spk (Surprise)                                       |
| `Train Unit Discovery` | `train/unit/`     | 100 spk (Development), 112 spk (Surprise)                                    |
| `Train Parallel`       | `train/parallel/` | parallel spoken utterances from the Target Voice and from other speakers     |
| `Test`                 | `test/`           | Unseen (not in Train) utterances by unseen speakers (Dev 24 spk, Sur 15 spk) |
"""

# Subtype = Literal["train/parallel/source", "train/parallel/voice", "train/unit", "train/voice", "test"] # >=Python3.8
Subtype = str
subtypes = ["train-parallel-source", "train-parallel-voice", "train-unit", "train-voice", "test"]
Speaker = str

class ItemIdZR19(NamedTuple):
    """Identity of Zero Resource Speech Challenge 2019 corpus's item.

    Args:
        subtype: Sub-corpus name
        speaker: Speaker ID
        serial_num: Utterance Number string
    """
    # Design Note: Audio Length
    #   ItemId do not contain audio length intentionally.
    #   'Effective' length differ in situation by situation.
    #   For example, native file length, non-zero length,
    #     dB-based length, Voice Activity region, etc...
    subtype: Subtype
    speaker: Speaker
    utterance_name: str


@dataclass
class ConfCorpus:
    """Configuration of corpus.
    Args:
        mirror_root: Root adress of corpus mirror, to which original archive is forwarded. If None, use default.
        download: Whether download original corpus or not when requested (e.g. origin->mirror forwarding).
    """
    mirror_root: Optional[str] = MISSING
    download: bool = MISSING

class ZR19(AbstractCorpus[ItemIdZR19]):
    """Zero Resource Speech Challenge 2019 corpus.

    Archive/contents handler of Zero Resource Speech Challenge 2019 corpus.
    Terminology:
        mirror: Mirror archive of the corpus
        contents: Contents extracted from archive
    """

    def __init__(self, conf: ConfCorpus) -> None:
        """Initiate ZR19 with archive options.
        """

        self.conf = conf

        # Equal to 1st layer directory name of original zip.
        self._corpus_name: str = f"english"
        archive_name = f"{self._corpus_name}.tgz"

        self._origin_adress = f"https://download.zerospeech.com/2019/english.tgz"
        # !wget --no-check-certificate 

        mirror_root = conf.mirror_root
        # Directory to which contents are extracted, and mirror is placed if adress is not provided.
        local_root = Path("./data")

        # Mirror: placed in given adress (conf) or default adress (local corpus directory)
        adress_mirror_given = f"{mirror_root}/corpuses/ZR19/{archive_name}" if mirror_root else None
        adress_mirror_default = str(local_root / "corpuses" / "ZR19" / "archive" / archive_name)
        self._adress_mirror = adress_mirror_given or adress_mirror_default

        # Contents: contents are extracted in local corpus directory
        self._path_contents = local_root / "corpuses" / "ZR19" / "contents"

    def get_contents(self) -> None:
        """Get corpus contents into local.
        """

        get_contents(self._adress_mirror, self._path_contents, self.conf.download, self.forward_from_origin)

    def forward_from_origin(self) -> None:
        """Forward original corpus archive to the mirror adress.
        """

        forward_from_general(self._origin_adress, self._adress_mirror)

    def get_identities(self) -> List[ItemIdZR19]:
        """Get corpus item identities.

        Currently, train/unit & train/voice are provided.
        Returns:
            Full item identity list.
        """

        # Design notes:
        #   No contents dependency is intentional.
        #   Corpus handler can be used without corpus itself (e.g. Get item identities for a preprocessed dataset).
        #   Hard-coded identity list enable contents-independent identity acquisition.

        # spk_unit: List[str] = ["S015", "S020", "S021", "S023", "S027", "S031", "S032", "S033",
        #     "S034", "S035", "S036", "S037", "S038", "S039", "S040", "S041", "S042", "S043",
        #     "S044", "S045", "S046", "S047", "S048", "S049", "S050", "S051", "S052", "S053",
        #     "S054", "S055", "S056", "S058", "S059", "S060", "S061", "S063", "S064", "S065",
        #     "S066", "S067", "S069", "S070", "S071", "S072", "S073", "S074", "S075", "S076",
        #     "S077", "S078", "S079", "S080", "S082", "S083", "S084", "S085", "S086", "S087",
        #     "S088", "S090", "S091", "S092", "S093", "S094", "S095", "S096", "S097", "S098",
        #     "S099", "S100", "S101", "S102", "S103", "S104", "S105", "S106", "S107", "S109",
        #     "S110", "S111", "S112", "S113", "S114", "S115", "S116", "S117", "S118", "S119",
        #     "S120", "S121", "S122", "S123", "S125", "S126", "S127", "S128", "S129", "S131",
        #     "S132", "S133"]
        # spk_voice: List[str] = ["V001","V002"]
        # spk_para_src: List[str] = ?
        # spk_para_vic: List[str] = ["V001","V002"]
        # spk_test: List[str] = ?

        ids: List[ItemIdZR19] = []
        for item in utterances_unit:
                ids.append(ItemIdZR19("train-unit", item[0:4], item))
        for item in utterances_voice:
                ids.append(ItemIdZR19("train-voice", item[0:4], item))
        return ids

    def get_item_path(self, id: ItemIdZR19) -> Path:
        """Get path of the item.

        Directory structure:
        train/
            parallel/
                source/
                    S000_0000000000.wav
                voice/
                    V001_0000000000.wav
                    V002_0000000000.wav
            unit/
                S000_0000000000.wav
            voice/
                V001_0000000000.wav
                V002_0000000000.wav
        test/
            S000_0000000000.wav
        
        Args:
            id: Target item identity.
        Returns:
            Path of the specified item.
        """
        root = self._path_contents
        return root / self._corpus_name / Path(id.subtype.replace('-', '/')) / f"{id.utterance_name}.wav"


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