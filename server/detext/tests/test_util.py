from django.test import TestCase

from detext.server.ml.models.mobilenet import MobileNet


class UtilTest(TestCase):
    def test_can_export_and_load_mobilenet(self):
        model = MobileNet(features=20)
        f = 'model.pth'

        model.to_file(f)
        model_2 = MobileNet.from_file(f)

        self.assertEquals(model.num_features, model_2.num_features)
