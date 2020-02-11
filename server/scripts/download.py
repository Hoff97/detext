import requests


def run():
    url = 'http://localhost:8000/api/image/download/'
    token = ''

    headers = {'Authorization': f'Token {token}'}

    response = requests.get(url, headers=headers)

    print(response)

    f = open('download.pth', 'wb')
    f.write(response.content)
    f.close()
