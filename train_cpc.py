from itertools import chain
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ZR19CPCMelSpkDataset
from scheduler import WarmupScheduler
from model import Encoder, CPCLoss
from config import load_conf, ConfGlobal


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def save_checkpoint(encoder, cpc, optimizer, scheduler, epoch, checkpoint_dir: Path):
    """Save model and learning states.

    Number in filename indicates `epoch`.
    """

    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "cpc": cpc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model.ckpt-{epoch}.pt"
    torch.save(checkpoint_state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path.stem}")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def train_model(cfg: ConfGlobal):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    checkpoint_dir = Path(cfg.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    encoder = Encoder(**cfg.model.encoder)
    cpc = CPCLoss(**cfg.model.cpc)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encoder.to(device)
    cpc.to(device)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    conf_sched = conf.training.cpc.scheduler
    # Adam & (original) WarmupScheduler
    optimizer = optim.Adam(
        chain(encoder.parameters(), cpc.parameters()),
        lr=conf_sched.initial_lr)
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=conf_sched.warmup_epochs,
        initial_lr=conf_sched.initial_lr,
        max_lr=conf_sched.max_lr,
        milestones=conf_sched.milestones,
        gamma=conf_sched.gamma)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if cfg.resume != "scratch":
        print(f"Resume checkpoint from: {cfg.resume}:")
        resume_path = cfg.resume
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        cpc.load_state_dict(checkpoint["cpc"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 1
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Item: (Utterance, Freq, T_clipped) from single speaker
    dataset = ZR19CPCMelSpkDataset(cfg.data.dataset.cpc)

    # Batch: (Speaker, Utterance, Freq, T_clipped)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.cpc.n_speakers_per_batch,
        shuffle=True,
        num_workers=1, pin_memory=True,
        drop_last=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for epoch in range(start_epoch, cfg.training.cpc.n_epochs + 1):
        ################################ epoch ################################
        if epoch % cfg.training.cpc.log_interval == 0 or epoch == start_epoch:
            average_cpc_loss = average_vq_loss = average_perplexity = 0
            average_accuracies = np.zeros(cfg.training.cpc.n_prediction_steps // 2)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i, (mels, _) in enumerate(tqdm(dataloader), 1):
            ############################# step ############################
            # mels::(Speaker, Utterance, Freq, T_clipped)
            mels = mels.to(device)
            # As batch of clipped mel-spectrogram (No distinguish between speakers and utterances)
            # (Spk*Utt, Freq, T_clipped)
            mels = mels.view(
                cfg.training.cpc.n_speakers_per_batch * cfg.training.cpc.n_utterances_per_speaker,
                cfg.dim_mel_freq,
                -1
            )

            optimizer.zero_grad()

            # (Spk*Utt, Freq, T_clipped) => (Spk*Utt, Time, Feature)
            z, c, vq_loss, perplexity = encoder(mels)
            cpc_loss, accuracy = cpc(z, c)
            loss = cpc_loss + vq_loss

            loss.backward()
            optimizer.step()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # For Logging
            average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i
            average_accuracies += (np.array(accuracy) - average_accuracies) / i
            ############################ /step ############################
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Count is epoch-based
        scheduler.step()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Logging
        if epoch % cfg.training.cpc.log_interval == 0 and epoch != start_epoch:
            ## TB
            # writer.add_scalar("cpc_loss/train", average_cpc_loss, epoch)
            # writer.add_scalar("vq_loss/train", average_vq_loss, epoch)
            # writer.add_scalar("perplexity/train", average_perplexity, epoch)
            ## console
            print("epoch:{}, cpc loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
                  .format(epoch, cpc_loss, average_vq_loss, average_perplexity))
            print(100 * average_accuracies)
        # Checkpointing
        if epoch % cfg.training.cpc.checkpoint_interval == 0 and epoch != start_epoch:
            save_checkpoint(
                encoder, cpc, optimizer,
                scheduler, epoch, checkpoint_dir)
        ############################### /epoch ################################
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    conf = load_conf()
    train_model(conf)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
