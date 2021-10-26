import torch
import pytorch_lightning as pl
import torchaudio

from config import load_conf
from model import Encoder
from datamodule import ZR19enDataModule
from train_vocoder import train_vocoder


if __name__ == "__main__":
    conf = load_conf()

    pl.seed_everything(conf.seed)
    torchaudio.set_audio_backend("sox_io")

    # Dataset
    datamodule = ZR19enDataModule(conf.data)

    # Pre-trained Encoder
    encoder = Encoder(conf.model.encoder)
    encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Resume cpc encoder from: {}:".format(conf.cpc_checkpoint))
    checkpoint = torch.load(conf.cpc_checkpoint, map_location=lambda storage, _: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    # Run training
    train_vocoder(conf.training_vocoder, datamodule, encoder)
