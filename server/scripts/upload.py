import requests

from detext.server.ml.models.mobilenet import MobileNet

from pathlib import Path

import base64


def run():
    file_name = "test_augment.pth"
    url = 'http://localhost:8000/api/model/'
    token = ''

    headers = {'Authorization': f'Token {token}'}

    pytorch = Path(file_name).read_bytes()
    model = MobileNet.from_file(file_name)
    model.eval()
    byte_arr = model.to_onnx()

    json = {
        "pytorch": base64.b64encode(pytorch).decode('utf-8'),
        "onnx": base64.b64encode(byte_arr.getvalue()).decode('utf-8')
    }

    response = requests.post(url, headers=headers, json=json)

    print(response)
