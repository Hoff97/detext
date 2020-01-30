import json

from django.contrib.auth.models import User


class AuthTestCase:
    def setUp(self):
        self.user = User.objects.create_superuser(username='testuser',
                                                  email='testemail@a.c',
                                                  password='12345')

        self.login()

    def tearDown(self):
        self.user.delete()

    def login(self):
        response = self.client.post('/api/api-token-auth/', {
            'username': 'testuser',
            'password': '12345'
        })
        self.token = json.loads(response.content)['token']
        self.auth_headers = {
            'HTTP_AUTHORIZATION': f'Token {self.token}'
        }
