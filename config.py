yaml_str = """
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
        bidirectional: true
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
"""

# Required and specific to some step
## train_vocoder
resume: False
cpc_checkpoint: ???
checkpoint_dir: ???
## Train_cpc
resume: False
checkpoint_dir: ???
## preprocessing
in_dir: ???
## encode
checkpoint: ???
out_dir: ???
save_auxiliary: False
## convert
synthesis_list: ???
in_dir: ???
out_dir: ???
cpc_checkpoint: ???
vocoder_checkpoint: ???

@dataclass
class ConfEncode:
    common: ConfAllStep = ConfAllStep()
    checkpoint: str = MISSING
    out_dir: str = MISSING
    save_auxiliary: False

# Requiredで入力させたい、かつ特定stepでしか使わない
# => 全stepで入力させるのは無駄. でもall-in-one confにMISSINGで置いておくと入力無しでerror.
# => dataclassのextendで各stepごとにconfのsubclass設定、そこではMISSINGが適切
#
