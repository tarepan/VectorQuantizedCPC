from model import ConfModel
from typing import Callable, List, TypeVar, Union
from dataclasses import dataclass

from omegaconf import OmegaConf, SCMode, MISSING

from dataset_zr19 import ConfDataset


CONF_DEFAULT_STR = """
dataset:
    adress_data_root: null
    clip_length_mel: ${training.vocoder.sample_frames}
    mel_stft_stride: 160
    corpus:
        download: false
    preprocess:
        sr: 16000
        n_fft: 2048
        n_mels: 80
        fmin: 50
        preemph: 0.97
        top_db: 80
        # hop_length: local sync
        win_length: 400
        bits: 8
model:
    encoder:
        in_channels: ${dataset.preprocess.n_mels}
        channels: 512
        n_embeddings: 512
        z_dim: 64
        c_dim: 256
    cpc:
        n_prediction_steps: ${training.cpc.n_prediction_steps}
        n_speakers_per_batch: ${training.cpc.n_speakers_per_batch}
        n_utterances_per_speaker: ${training.cpc.n_utterances_per_speaker}
        n_negatives: ${training.cpc.n_negatives}
        z_dim: ${model.encoder.z_dim}
        c_dim: ${model.encoder.c_dim}
    vocoder:
        in_channels: ${model.encoder.z_dim}
        dim_voc_latent: 256
        upsampling_t: ${dataset.preprocess.hop_length}
        bits_mu_law: ${dataset.preprocess.bits}
        n_speakers: 102
        # todo: Fix n_speakers dependency. Now this is not hardcoded.
        speaker_embedding_dim: 64
        bidirectional: true
        mu_embedding_dim: 256
        rnn_channels: 896
        fc_channels: 256
training:
    cpc:
        sample_frames: 128
        n_speakers_per_batch: 8
        n_utterances_per_speaker: 8
        n_prediction_steps: 12
        n_negatives: 17
        n_epochs: 22000
        scheduler:
            warmup_epochs: 150
            initial_lr: 1e-5
            max_lr: 4e-4
            gamma: 0.25
            milestones:
                - 20000
        checkpoint_interval: 500
        n_workers: 8
        log_interval: 10
    vocoder:
        batch_size: 32
        sample_frames: 32
        n_steps: 160000
        optimizer_lr: 4e-4
        scheduler_milestones:
            - 50000
            - 75000
            - 100000
            - 125000
        scheduler_gamma: 0.5
        checkpoint_interval: 5000
        n_workers: 8
resume: NoResume
checkpoint_dir: checkpoints/vqcpc/version1
cpc_checkpoint: checkpoints/cpc/english2019/model.ckpt-22000.pt
vocoder_checkpoint: checkpoints/vocoder/english2019/version1/model.ckpt-xxxxxx.pt
save_auxiliary: False
in_dir: zerospeech/2019
out_dir: results/z2019en
synthesis_list: ./target_vc.json
"""


@dataclass
class ConfTrainCPCSched:
    warmup_epochs: int = MISSING
    initial_lr: float = MISSING
    max_lr: float = MISSING
    gamma: float = MISSING
    milestones: List[int] = MISSING


@dataclass
class ConfTrainCPC:
    sample_frames: int = MISSING
    n_speakers_per_batch: int = MISSING
    n_utterances_per_speaker: int = MISSING
    n_prediction_steps: int = MISSING
    n_negatives: int = MISSING
    n_epochs: int = MISSING
    scheduler: ConfTrainCPCSched = ConfTrainCPCSched()
    checkpoint_interval: int = MISSING
    n_workers: int = MISSING
    log_interval: int = MISSING    


@dataclass
class ConfTrainVocoder:
    batch_size: int = MISSING
    sample_frames: int = MISSING
    n_steps: int = MISSING
    optimizer_lr: float = MISSING
    scheduler_milestones: List[int] = MISSING
    scheduler_gamma: float = MISSING
    checkpoint_interval: int = MISSING
    n_workers: int = MISSING


@dataclass
class ConfTraining:
    cpc: ConfTrainCPC = ConfTrainCPC()
    vocoder: ConfTrainVocoder = ConfTrainVocoder()


@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        resume: Resume training from `resume` path
        checkpoint_dir:
        cpc_checkpoint:
        vocoder_checkpoint:
        save_auxiliary:
        in_dir:
        out_dir:
        synthesis_list:
    """
    resume: str = MISSING
    checkpoint_dir: str = MISSING
    cpc_checkpoint: str = MISSING
    vocoder_checkpoint: str = MISSING
    save_auxiliary: bool = MISSING
    in_dir: str = MISSING
    out_dir: str = MISSING
    synthesis_list: str = MISSING
    dataset: ConfDataset = ConfDataset()
    model: ConfModel = ConfModel()
    training: ConfTraining = ConfTraining()


def conf_default() -> ConfGlobal:
    """Default global configuration.
    """
    return OmegaConf.merge(
        OmegaConf.structured(ConfGlobal),
        OmegaConf.create(CONF_DEFAULT_STR)
    )

T = TypeVar('T')
def gen_load_conf(gen_conf_default: Callable[[], T], ) -> Callable[[], T]:
    """Generate 'Load configuration type-safely' function.
    Priority: CLI args > CLI-specified config yaml > Default
    Args:
        gen_conf_default: Function which generate default structured config
    """

    def generated_load_conf() -> T:
        default = gen_conf_default()
        cli = OmegaConf.from_cli()
        extends_path = cli.get("path_extend_conf", None)
        if extends_path:
            extends = OmegaConf.load(extends_path)
            conf_final = OmegaConf.merge(default, extends, cli)
        else:
            conf_final = OmegaConf.merge(default, cli)

        # Design Note -- OmegaConf instance v.s. DataClass instance --
        #   OmegaConf instance has runtime overhead in exchange for type safety.
        #   Configuration is constructed/finalized in early stage,
        #   so config is eternally valid after validation in last step of early stage.
        #   As a result, we can safely convert OmegaConf to DataClass after final validation.
        #   This prevent (unnecessary) runtime overhead in later stage.
        #
        #   One demerit: No "freeze" mechanism in instantiated dataclass.
        #   If OmegaConf, we have `OmegaConf.set_readonly(conf_final, True)`

        # [todo]: Return both dataclass and OmegaConf because OmegaConf has export-related utils.

        # `.to_container()` with `SCMode.INSTANTIATE` resolve interpolations and check MISSING.
        # It is equal to whole validation.
        return OmegaConf.to_container(conf_final, structured_config_mode=SCMode.INSTANTIATE)

    return generated_load_conf

load_conf = gen_load_conf(conf_default)
"""Load configuration type-safely.
"""