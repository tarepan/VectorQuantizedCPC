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
        n_prediction_steps: ${training.n_prediction_steps}
        n_speakers_per_batch: ${training.n_speakers_per_batch}
        n_utterances_per_speaker: ${training.n_utterances_per_speaker}
        n_negatives: ${training.n_negatives}
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
cpc_checkpoint: ???
checkpoint_dir: ???
"""