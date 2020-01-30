from detext.server.util.train import train_classifier


def run():
    train_classifier(device='cuda')
