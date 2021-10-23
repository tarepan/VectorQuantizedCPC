from typing import Callable, TypeVar, Union
from dataclasses import dataclass

from omegaconf import OmegaConf, SCMode, MISSING

from preprocess import ConfPreprocessing


CONF_DEFAULT_STR = """
dataset:
    dataset: 2019
    language: english
    path: 2019/english
    n_speakers: 102
preprocessing:
    sr: 16000
    n_fft: 2048
    n_mels: 80
    fmin: 50
    preemph: 0.97
    top_db: 80
    hop_length: 160
    win_length: 400
    bits: 8
model:
    encoder:
        in_channels: ${preprocessing.n_mels}
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
        conditioning_channels: 128
        n_speakers: ${dataset.n_speakers}
        speaker_embedding_dim: 64
        mu_embedding_dim: 256
        rnn_channels: 896
        fc_channels: 256
        bits: ${preprocessing.bits}
        hop_length: ${preprocessing.hop_length}
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
        optimizer:
            lr: 4e-4
        scheduler:
            milestones:
                - 50000
                - 75000
                - 100000
                - 125000
            gamma: 0.5
        checkpoint_interval: 5000
        n_workers: 8
resume: False
checkpoint_dir: checkpoints/vqcpc/version1
cpc_checkpoint: checkpoints/cpc/english2019/model.ckpt-22000.pt
vocoder_checkpoint: checkpoints/vocoder/english2019/version1/model.ckpt-xxxxxx.pt
save_auxiliary: False
in_dir: zerospeech/2019
out_dir: results/z2019en
synthesis_list: ./target_vc.json
"""


@dataclass
class ConfDataset:
    dataset: str = MISSING
    language: str = MISSING
    path: str = MISSING
    n_speakers: int = MISSING

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
    resume: Union[bool, str] = MISSING
    checkpoint_dir: str = MISSING
    cpc_checkpoint: str = MISSING
    vocoder_checkpoint: str = MISSING
    save_auxiliary: bool = MISSING
    in_dir: str = MISSING
    out_dir: str = MISSING
    synthesis_list: str = MISSING
    preprocessing: ConfPreprocessing = ConfPreprocessing()
    dataset: ConfDataset = ConfDataset()


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