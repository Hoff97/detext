from django.test import TestCase

from detext.server.ml.models.mobilenet import MobileNet, TestTimeDropout

import torch


class MlTest(TestCase):
    def test_mobilenet_can_predict_uncertainty(self):
        model = MobileNet(features=20, estimate_variane=True)
        inp = torch.randn(1, 3, 224, 224, device="cpu")
        result = model(inp)

        self.assertEquals(list(result.shape), [2, 1, 20])

    def test_mobilenet_can_use_testtime_dropout(self):
        model = MobileNet(features=20, test_time_dropout=True)
        model = model.eval()

        inp = torch.randn(1, 3, 224, 224, device="cpu")
        result1 = model(inp)
        result2 = model(inp)

        self.assertFalse(torch.all(result1.eq(result2)))

    def test_test_time_dropout(self):
        with self.assertRaises(ValueError, msg='Dropout shouldnt accept p<0'):
            TestTimeDropout(p=-1)

        with self.assertRaises(ValueError, msg='Dropout shouldnt accept p>1'):
            TestTimeDropout(p=2)
