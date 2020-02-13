from django.test import TestCase
from rest_framework import status

import scripts.train_augment as train_augment
from detext.server.util.transfer import get_upload_json
from detext.tests.util.auth import AuthTestCase


class MathSymbolViewTest(AuthTestCase, TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_can_get_most_recent(self):
        response = self.client.get('/api/model/latest/')
        self.assertEquals(response.status_code, status.HTTP_200_OK)

    def test_train_requires_login(self):
        response = self.client.post('/api/model/train/')
        self.assertEquals(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_train_works(self):
        response = self.client.post('/api/model/train/', {
            'epochs': 1
        }, **self.auth_headers)
        self.assertEquals(response.status_code, status.HTTP_200_OK)

    def test_train_no_epochs(self):
        response = self.client.post('/api/model/train/', **self.auth_headers)
        self.assertEquals(response.status_code, status.HTTP_200_OK)

    def test_upload_requires_login(self):
        train_augment.run(num_epochs=1, device="cpu")
        json = get_upload_json('test_augment.pth')
        response = self.client.post('/api/model/', json)
        self.assertEquals(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_upload_works_with_login(self):
        train_augment.run(num_epochs=1, device="cpu")
        json = get_upload_json('test_augment.pth')
        response = self.client.post('/api/model/', json,
                                    content_type='application/json',
                                    **self.auth_headers)
        self.assertEquals(response.status_code, status.HTTP_200_OK)
