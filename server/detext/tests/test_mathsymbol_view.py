import json
from datetime import datetime

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from rest_framework import status

from detext.server.models import MathSymbol

from rest_framework.test import APIClient

from detext.tests.util.auth import AuthTestCase

class MathSymbolViewTest(AuthTestCase, TestCase):
    def setUp(self):
        super().setUp()

        self.o = MathSymbol.objects.create(name = 'Test', description = 'Test', timestamp = datetime.now())

    def tearDown(self):
        super().tearDown()
        self.o.delete()

    def test_can_get_math_symbol(self):
        response = self.client.get('/api/symbol/')
        self.assertEquals(response.status_code, status.HTTP_200_OK)

    def test_delete_requires_login(self):
        response = self.client.delete(f'/api/symbol/{self.o.id}/')
        self.assertEquals(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_can_delete(self):
        response = self.client.delete(f'/api/symbol/{self.o.id}/', **self.auth_headers)
        self.assertEquals(response.status_code, status.HTTP_204_NO_CONTENT)

        with self.assertRaises(MathSymbol.DoesNotExist):
            MathSymbol.objects.get(pk=self.o.id)

    def test_create_works(self):
        response = self.client.post(f'/api/symbol/', {
            'name': 'test2',
            'description': 'test2',
            'latex': 'string',
            'image': 'abcd',
            'timestamp': datetime.now()
        }, **self.auth_headers)

        self.assertEquals(response.status_code, status.HTTP_201_CREATED)

    def test_create_requires_name(self):
        response = self.client.post(f'/api/symbol/', {
            'description': 'test2',
            'latex': 'string',
            'image': 'abcd',
            'timestamp': datetime.now()
        }, **self.auth_headers)

        self.assertEquals(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_update_requires_login(self):
        response = self.client.put(f'/api/symbol/{self.o.id}/', {
            'name': 'test2',
            'description': 'test2',
            'latex': 'string',
            'image': 'abcd',
            'timestamp': datetime.now()
        })

        self.assertEquals(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_update_works(self):
        response = self.client.put(f'/api/symbol/{self.o.id}/', {
            'name': 'test2',
            'description': 'test2',
            'latex': 'string',
            'timestamp': datetime.now()
        }, content_type='application/json', **self.auth_headers)

        self.assertEquals(response.status_code, status.HTTP_200_OK)

    def test_update_should_not_contain_image(self):
        response = self.client.put(f'/api/symbol/{self.o.id}/', {
            'name': 'test2',
            'description': 'test2',
            'latex': 'string',
            'image': 'abcd',
            'timestamp': datetime.now()
        }, content_type='application/json', **self.auth_headers)

        self.assertEquals(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_update_image_should_work(self):
        response = self.client.put(f'/api/symbol/{self.o.id}/image/', {
            'image': 'abcd'
        }, content_type='application/json', **self.auth_headers)

        self.assertEquals(response.status_code, status.HTTP_200_OK)

    def test_update_image_should_require_login(self):
        response = self.client.put(f'/api/symbol/{self.o.id}/image/', {
            'image': 'abcd'
        }, content_type='application/json')

        self.assertEquals(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_update_image_should_require_valid_pk(self):
        response = self.client.put(f'/api/symbol/200/image/', {
            'image': 'abcd'
        }, content_type='application/json', **self.auth_headers)

        self.assertEquals(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_update_image_should_require_image(self):
        response = self.client.put(f'/api/symbol/200/image/', {
        }, content_type='application/json', **self.auth_headers)

        self.assertEquals(response.status_code, status.HTTP_400_BAD_REQUEST)