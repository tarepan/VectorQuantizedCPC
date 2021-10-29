from typing import Callable, List, TypeVar
from dataclasses import dataclass

from omegaconf import OmegaConf, SCMode, MISSING

from dataset_zr19 import ConfDataset
from train_vocoder import ConfTrainVocoder
from model import ConfModel
from datamodule import ConfData


CONF_DEFAULT_STR = """
seed: 13
sampling_rate: 16000
bit_mulaw: 8
dim_mel_freq: 80
size_latent_codebook: 512 
dim_latent: 64
dim_cpc_context: 256
cpc_checkpoint: checkpoints/cpc/english2019/model.ckpt-22000.pt
vocoder_checkpoint: checkpoints/vocoder/english2019/version1/model.ckpt-xxxxxx.pt
save_auxiliary: False
synthesis_list: ./target_vc.json
model:
    encoder:
        in_channels: ${dim_mel_freq}
        channels: 512
        n_embeddings: ${size_latent_codebook}
        z_dim: ${dim_latent}
        c_dim: ${dim_cpc_context}
    cpc:
        n_prediction_steps: ${training.cpc.n_prediction_steps}
        n_speakers_per_batch: ${training.cpc.n_speakers_per_batch}
        n_utterances_per_speaker: ${training.cpc.n_utterances_per_speaker}
        n_negatives: ${training.cpc.n_negatives}
        z_dim: ${dim_latent}
        c_dim: ${dim_cpc_context}
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
training_vocoder:
    model:
        sampling_rate: ${sampling_rate}
        # todo: Fix n_speakers dependency. Now this is not hardcoded.
        n_speakers: 102
        network:
            size_i_codebook: ${size_latent_codebook}
            dim_i_embedding: ${dim_latent}
            dim_speaker_embedding: 64
            rnnms:
                dim_voc_latent: 256
                bits_mu_law: ${bit_mulaw}
                upsampling_t: ${data.dataset.preprocess.hop_length}
                prenet:
                    num_layers: 2
                    bidirectional: true
                wave_ar:
                    size_i_embed_ar: 256
                    size_h_rnn: 896
                    size_h_fc: 256
        optim:
            learning_rate: 4e-4
            sched_milestones:
                - 50000
                - 75000
                - 100000
                - 125000
            sched_gamma: 0.5
    trainer:
        max_epochs: 540
        val_interval_epoch: 10
        profiler: null
    ckpt_log:
        dir_root: vqcpc_vocoder
        name_exp: default
        name_version: version_-1
data:
    loader:
        batch_size: 32
        num_workers: 8
        pin_memory: null
    dataset:
        adress_data_root: null
        clip_length_mel: 32
        mel_stft_stride: 160
        corpus:
            download: false
        preprocess:
            sr: ${sampling_rate}
            n_fft: 2048
            n_mels: ${dim_mel_freq}
            fmin: 50
            preemph: 0.97
            top_db: 80
            # hop_length: local sync
            win_length: 400
            bits: ${bit_mulaw}
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
class ConfTraining:
    cpc: ConfTrainCPC = ConfTrainCPC()


@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        seed: Random seed
        sampling_rate: Audio sampling rate
        bit_mulaw: Bit depth of mu-law signal
        dim_mel_freq: Dimension of mel-spectrogram's frequency
        size_latent_codebook: Codebook size of latent (Number of index)
        dim_latent: Dimension of latent vector
        cpc_checkpoint: CPC encoder checkpoint
        vocoder_checkpoint: RNNMS vocoder checkpoint
        save_auxiliary:
        synthesis_list:
    """
    seed: int = MISSING
    sampling_rate: int = MISSING
    bit_mulaw: int = MISSING
    dim_mel_freq: int = MISSING
    size_latent_codebook: int = MISSING 
    dim_latent: int = MISSING
    dim_cpc_context: int = MISSING
    cpc_checkpoint: str = MISSING
    vocoder_checkpoint: str = MISSING
    save_auxiliary: bool = MISSING
    synthesis_list: str = MISSING
    model: ConfModel = ConfModel()
    training: ConfTraining = ConfTraining()
    training_vocoder: ConfTrainVocoder = ConfTrainVocoder()
    data: ConfData = ConfData()

def conf_default() -> ConfGlobal:
    """Default global configuration.
    """
    return OmegaConf.merge(
        OmegaConf.structured(ConfGlobal),
        OmegaConf.create(CONF_DEFAULT_STR)
    )

def conf_programatic(conf: ConfGlobal) -> ConfGlobal:
    """
    """
    # Target: `conf.model.vocoder.rnnms.dim_i_feature`
    conf_voc = conf.training_vocoder.model.network
    conf_voc.rnnms.dim_i_feature = conf_voc.dim_i_embedding + conf_voc.dim_speaker_embedding

    # PlaceHolder

    return conf

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
            conf_merged = OmegaConf.merge(default, extends, cli)
        else:
            conf_merged = OmegaConf.merge(default, cli)
        conf_final = conf_programatic(conf_merged)

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

if __name__ == "__main__":
    print(load_conf())