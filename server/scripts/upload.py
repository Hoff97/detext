import requests

from detext.server.util.transfer import get_upload_json


def run():
    file_name = "test_augment.pth"
    url = 'http://localhost:8000/api/model/'
    token = ''

    headers = {'Authorization': f'Token {token}'}

    json = get_upload_json(file_name)

    response = requests.post(url, headers=headers, json=json)

    print(response)
