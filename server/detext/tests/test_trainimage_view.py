import base64
import io

from django.test import TestCase
from PIL import Image
from rest_framework import status

from detext.server.models import TrainImage
from detext.tests.util.auth import AuthTestCase


class MathSymbolViewTest(AuthTestCase, TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_create_works(self):
        item = TrainImage.objects.get(pk=1)

        image = Image.open(io.BytesIO(item.image))
        image = image.convert("RGBA")

        bt = base64.encodebytes(image.tobytes()).decode('UTF-8')
        response = self.client.post(f'/api/image/', {
            'image': bt,
            'width': image.width,
            'height': image.height,
            'symbol': item.symbol.id
        }, **self.auth_headers)

        self.assertEquals(response.status_code, status.HTTP_201_CREATED)

    def test_dist_works(self):
        response = self.client.get(f'/api/image/dist/')

        self.assertEquals(response.status_code, status.HTTP_200_OK)

    def test_dist_works_with_args(self):
        response = self.client.get(f'/api/image/dist/?log&width=10&height=10')

        self.assertEquals(response.status_code, status.HTTP_200_OK)
