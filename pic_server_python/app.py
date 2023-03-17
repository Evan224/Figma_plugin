from flask import Flask, send_file, request
from flask_cors import CORS
import os
import io
import numpy as np
from PIL import Image
import time

DEFAULT_URL = 'http://localhost:5000'

app = Flask(__name__)
CORS(app)


@app.route('/picture/<string:pic_name>')
def serve_picture(pic_name):
    filename = f'{pic_name}.jpg'
    response = send_file(f'../mock/ui_imgs/{filename}', mimetype='image/jpg')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


@app.route('/json/<string:pic_name>')
def serve_json(pic_name):
    filename = f'{pic_name}.json'
    response = send_file(f'../mock/json/{filename}', mimetype='json')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


@app.route('/api/imageList', methods=['GET'])
def image_list():
    static_path = '../mock/ui_imgs'
# use os.listdir() to get a list of all filenames in the folder
    filenames = os.listdir(static_path)
    # use a list comprehension to filter out non-image filenames
    image_filenames = [filename for filename in filenames if filename.endswith(
        '.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')]
    print(image_filenames)
    return {'imageList': image_filenames}


@app.route('/api/image', methods=['POST'])
def image():
    # get the input data
    file = request.files['image']

    # save the file to the server's file system
    file.save('test.png')

    # create a response JSON object and return it
    response_data = {'result': 'success'}
    return response_data


if __name__ == '__main__':
    app.run()
