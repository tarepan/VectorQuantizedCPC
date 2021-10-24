from pathlib import Path

from tqdm import tqdm
import soundfile
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_zr19 import ZR19MulawMelSpkDataset
from model import Encoder, Vocoder
from config import load_conf, ConfGlobal


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def save_checkpoint(decoder, optimizer, scheduler, step, checkpoint_dir):
    """Save model and learning states.
    
    Number in filename indicates global `step`.
    """
    
    checkpoint_state = {
        "vocoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def train_model(conf: ConfGlobal):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    checkpoint_dir = Path(conf.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    encoder = Encoder(conf.model.encoder)
    vocoder = Vocoder(conf.model.vocoder)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encoder.to(device)
    vocoder.to(device)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Adam, MultiStepLR
    optimizer = optim.Adam(
        vocoder.parameters(),
        lr=conf.training.vocoder.optimizer_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=conf.training.vocoder.scheduler_milestones,
        gamma=conf.training.vocoder.scheduler_gamma)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if conf.resume is not "NoResume":
        print("Resume checkpoint from: {}:".format(conf.resume))
        checkpoint = torch.load(conf.resume, map_location=lambda storage, _: storage)
        vocoder.load_state_dict(checkpoint["vocoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step: int = checkpoint["step"]
    else:
        global_step: int = 0
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # If encoder is used in preprocessing for vocoder, this move to other files.
    print("Resume cpc encoder from: {}:".format(conf.cpc_checkpoint))
    checkpoint = torch.load(conf.cpc_checkpoint, map_location=lambda storage, _: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    # /

    # Dataset preparation
    dataset = ZR19MulawMelSpkDataset(True, conf.dataset)
    dataset_val = ZR19MulawMelSpkDataset(False, conf.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=conf.training.vocoder.batch_size,
        shuffle=True,
        num_workers=conf.training.vocoder.n_workers,
        pin_memory=True,
        drop_last=True)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False)

    dir_sample = Path("./out_sample")
    dir_sample.mkdir(parents=True, exist_ok=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    n_epochs = conf.training.vocoder.n_steps // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for epoch in range(start_epoch, n_epochs + 1):
        ################################ epoch ################################
        average_loss = 0
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
            ############################# step ############################
            # tqdm report 'step' number in progress bar

            # Setup
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)
            optimizer.zero_grad()

            # This could be done in preprocessing for Vocoder. Isn't it?
            with torch.no_grad():
                _, _, indices = encoder.encode(mels)
            # /

            # Vocoding -> CE loss
            output = vocoder(audio[:, :-1], indices, speakers)
            loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])

            # Backward / Gradient clipping / Optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vocoder.parameters(), 1)
            optimizer.step()
            scheduler.step()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            average_loss += (loss.item() - average_loss) / i
            global_step += 1
            # Checkpointing
            #   global step based, not epoch based
            if global_step % conf.training.vocoder.checkpoint_interval == 0:
                save_checkpoint(
                    vocoder, optimizer, scheduler, global_step, checkpoint_dir)
            ############################# step ############################
        # Logging
        print("epoch:{}, loss:{:.3E}".format(epoch, average_loss))
        # Sample generation
        # Difference mel-spec length, so batch_size=1
        #   5 min/epoch in P100
        if epoch % 12 == 0:
            dl_v = iter(dataloader_val)
            for i in range(0, 3):
                _, mels, speakers = next(dl_v)
                mels, speakers = mels.to(device), speakers.to(device)
                with torch.no_grad():
                    _, _, indices = encoder.encode(mels)
                    # output::float
                    output = vocoder.generate(indices, speakers)
                soundfile.write(f"{str(dir_sample)}/No{i}_step{global_step}.wav", output[0], conf.dataset.preprocess.sr)
        ############################### /epoch ################################
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    conf = load_conf()
    train_model(conf)
