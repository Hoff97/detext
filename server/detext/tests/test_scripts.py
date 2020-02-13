from django.test import TestCase

import scripts.import_data as import_data
import scripts.import_model as import_model
import scripts.test as test
import scripts.test_onnx as test_onnx
import scripts.train as train
import scripts.train_augment as train_augment
import scripts.train_classifier as train_classifier
from detext.server.util.transfer import data_to_file


class ScriptTest(TestCase):
    def test_test_script(self):
        test.run()

    def test_test_onnx_script(self):
        test_onnx.run()

    def test_train_classifier_script(self):
        train_classifier.run()

    def test_train_script(self):
        train.run(num_epochs=1, device="cpu")

    def test_train_augment(self):
        train_augment.run(num_epochs=1, device="cpu")

    def test_import_model(self):
        train_augment.run(num_epochs=1, device="cpu")
        import_model.run()

    def test_import_data_works(self):
        data = data_to_file()
        f = open('download.pth', 'wb')
        f.write(data.getvalue())
        f.close()
        import_data.run()
