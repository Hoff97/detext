import scripts.download as download
import scripts.import_data as import_data
import scripts.train_augment as train_augment
import scripts.upload as upload


def run():
    download.run()
    import_data.run()
    train_augment.run()
    upload.run()
