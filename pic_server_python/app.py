from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import os
import io
import numpy as np
from PIL import Image
import time
import json

DEFAULT_URL = 'http://127.0.0.1:5000'

app = Flask(__name__)
CORS(app)

json_dicts = {

}


@app.route('/picture/<string:pic_name>')
def serve_picture(pic_name):
    filename = f'{pic_name}.jpg'
    response = send_file(f'../mock/ui_imgs/{filename}', mimetype='image/jpg')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


@app.route('/element/<string:pic_name>')
def serve_element(pic_name):
    filename = f'{pic_name}.png'
    response = send_file(
        f'../mock/element_imgs/{filename}', mimetype='image/png')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


@app.route('/json/<string:pic_name>')
def serve_json(pic_name):
    if pic_name in json_dicts:
        print('cache hit')
        return json_dicts[pic_name]
    filename = f'{pic_name}.json'
    with open(f'../mock/json/{filename}', 'rb') as f:
        file_contents = json.load(f)
        json_dicts[pic_name] = file_contents
    response = send_file(json_dicts[pic_name], mimetype='json')
    print(response, type(response))
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return jsonify(json_dicts[pic_name])


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
