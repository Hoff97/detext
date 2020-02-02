from django.test import TestCase

import scripts.test as test
import scripts.test_onnx as test_onnx
import scripts.train_classifier as train_classifier
import scripts.train as train


class ScriptTest(TestCase):
    def test_test_script(self):
        test.run()

    def test_test_onnx_script(self):
        test_onnx.run()

    def test_train_classifier_script(self):
        train_classifier.run()

    def test_train_script(self):
        train.run(num_epochs=1)
