import base64
import io
import json
from datetime import datetime

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from PIL import Image
from rest_framework import status
from rest_framework.test import APIClient

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
