from django.test import TestCase
from PIL import Image

from detext.server.ml.models.mobilenet import MobileNet
from detext.server.ml.util.util import augment_image


class UtilTest(TestCase):
    def test_augment_image(self):
        img = Image.open('res/test/alpha/alpha.png')
        augment_image(img)

    def test_can_export_and_load_mobilenet(self):
        model = MobileNet(features=20)
        f = 'model.pth'

        model.to_file(f)
        model_2 = MobileNet.from_file(f)

        self.assertEquals(model.num_features, model_2.num_features)
