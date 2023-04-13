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
    response = send_file(f'./static/UI/{filename}', mimetype='image/jpg')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response

@app.route('/background/<string:pic_name>')
def serve_background(pic_name):

    filename = f'{pic_name}.jpg'
    response = send_file(f'./static/background/{filename}', mimetype='image/jpg')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


@app.route('/element/<string:pic_name>/<int:element_id>')
def serve_element(pic_name, element_id):
    filename = f'element_{element_id}.jpg'
    response = send_file(
        f'./static/elements/{pic_name}/{filename}', mimetype='image/jpg')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


@app.route('/json/<string:pic_name>')
def serve_json(pic_name):
    filename = f'{pic_name}.json'
    with open(f'./static/json/{filename}', 'rb') as f:
        file_contents = json.load(f)
        json_dicts[pic_name] = file_contents
    response = send_file(json_dicts[pic_name], mimetype='json')
    print(response, type(response))
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return jsonify(json_dicts[pic_name])

@app.route('/partialjson/<string:pic_name>')
def serve_partial_json(pic_name):
    filename = f'{pic_name}.json'
    with open(f'./static/InitJSON/{filename}', 'rb') as f:
        file_contents = json.load(f)
        json_dicts[pic_name] = file_contents
    response = send_file(json_dicts[pic_name], mimetype='json')
    print(response, type(response))
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return jsonify(json_dicts[pic_name])


@app.route('/recommend', methods=['POST'])
def recommend_json():
    # Get the JSON file from the request
    data = request.get_json()


    time.sleep(1)
    # Return the modified JSON file
    return jsonify(data)

# get the list of image filenames in the static folder
@app.route('/api/imageList', methods=['GET'])
def image_list():
    static_path = './static/UI_temp'
    # use os.listdir() to get a list of all filenames in the folder
    filenames = os.listdir(static_path)
    # use a list comprehension to filter out non-image filenames
    image_filenames = [filename for filename in filenames if filename.endswith(
        '.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')]

    print(image_filenames)

    return {'imageList': image_filenames}

@app.route('/api/elementList/<name>', methods=['GET'])
def element_list(name):
    static_path = './static/elements/'+name
    # use os.listdir() to get a list of all filenames in the folder
    filenames = os.listdir(static_path)

    return {'imageList': filenames}

@app.route('/target', methods=['POST'])
def target():
    # print the body of the request
    print(request.get_json())
    # mock the target algorithm
    # time.sleep(2)
    # send the hello world response
    response = {'result': 'success'}
    return jsonify(response), 200, {'Access-Control-Allow-Origin': '*'}

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
