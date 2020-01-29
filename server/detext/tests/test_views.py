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
        self.assertEquals(response.status_code, 200)

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