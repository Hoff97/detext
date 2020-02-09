from detext.server.ml.train_classifier import train_classifier

import torch


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_classifier(device=device)
