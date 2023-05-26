from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import os
import io
import numpy as np
from PIL import Image
import time
import base64
import json
from figma_function import figma_fn
from loop_figma_function import figma_fn as loop_figma_fn


DEFAULT_URL = 'http://127.0.0.1:5000'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


json_dicts = {

}


@app.route('/partial_picture/<string:pic_name>')
def serve_partial_picture(pic_name):
    filename = f'{pic_name}'
    print(f"./static/partial_picture/{filename}")
    if os.path.exists(f'./static/partial_picture/{filename}'):
        print("file exists====================================================")
        response = send_file(
            f'./static/partial_picture/{filename}', mimetype='image/jpg')
    else:
        response = send_file(
            f'./static/background/{filename}', mimetype='image/jpg')

    response_data = {
        'image_url': f"{DEFAULT_URL}/static/partial_picture/{filename}"
    }
    response = jsonify(response_data)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS

    return response


@app.route('/picture/<string:pic_name>')
def serve_picture(pic_name):

    filename = f'{pic_name}.jpg'
    response = send_file(f'./static/UI/{filename}', mimetype='image/jpg')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


@app.route('/background/<string:pic_name>')
def serve_background(pic_name):

    filename = f'{pic_name}.jpg'
    response = send_file(
        f'./static/background/{filename}', mimetype='image/jpg')
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
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return jsonify(json_dicts[pic_name])


@app.route('/partialjson/<string:pic_name>')
def serve_partial_json(pic_name):
    filename = f'{pic_name}.json'
    with open(f'./static/InitJSON/{filename}', 'rb') as f:
        file_contents = json.load(f)
        json_dicts[pic_name] = file_contents
    response = send_file(json_dicts[pic_name], mimetype='json')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return jsonify(json_dicts[pic_name])


@app.route('/recommend', methods=['POST'])
def recommend_json():
    # Get the JSON file from the request

    data = request.get_json()

    result = loop_figma_fn(data["data"], data["name"])

    return jsonify(result)


@app.route('/recommend/mock', methods=['POST'])
def recommend_json_mock():
    # Get the JSON file from the request

    data = request.get_json()

    ui = data["name"]
    with open(f'./static/json/{ui}.json', 'rb') as f:
        file_contents = json.load(f)

    current_elements = data["data"]["elements"]
    all_elements = file_contents["elements"]

    # get all missing elements
    missing_elements = []
    for element in all_elements:
        if element not in current_elements:
            missing_elements.append(element)

    for element in missing_elements:
        if int(element["id"]) < 6:
            element["level"] = "high"
        elif int(element["id"]) < 11:
            element["level"] = "medium"
        else:
            element["level"] = "low"

    file_contents["target"] = missing_elements
    return jsonify(file_contents)

# get the list of image filenames in the static folder


@app.route('/api/imageList', methods=['GET'])
def image_list():

    # set the path to the static folder

    static_path = './static/UI_temp'
    # use os.listdir() to get a list of all filenames in the folder
    filenames = os.listdir(static_path)
    # use a list comprehension to filter out non-image filenames
    image_filenames = [filename for filename in filenames if filename.endswith(
        '.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')]
    response = jsonify({'imageList': image_filenames})
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


@app.route('/api/elementList/<name>', methods=['GET'])
def element_list(name):
    static_path = './static/elements/'+name
    # use os.listdir() to get a list of all filenames in the folder
    filenames = os.listdir(static_path)

    return {'imageList': filenames}


@app.route('/api/mock/libraryElements', methods=['GET'])
def library_mock_elements():
    static_path = './static/library_elements'
    # use os.listdir() to get a list of all filenames in the folder
    filenames = os.listdir(static_path)
    # use a list comprehension to filter out non-image filenames
    image_filenames = [filename for filename in filenames if filename.endswith(
        '.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')]
    response = jsonify({'imageList': image_filenames})

    return response


@app.route('/target', methods=['POST'])
def target():
    # Get the JSON file from the request
    data = request.get_json()

    result, state = figma_fn(data)

    return jsonify(result)


@app.route('/dotted/mock', methods=['POST'])
def dotted_mock():
    # Get the JSON file from the request
    data = request.get_json()
    # based on the current data, return the type, position, width, and height of the dotted line randomly
    result = {
        'type': 'image',
        'x': np.random.randint(100, 500),
        'y': np.random.randint(100, 500),
        'width': np.random.randint(0, 100),
        'height': np.random.randint(0, 100)
    }

    return jsonify(result)


@app.route('/api/image', methods=['POST'])
def image():
    # get the input data
    file = request.files['image']

    # save the file to the server's file system
    file.save('test.png')

    # create a response JSON object and return it
    response_data = {'result': 'success'}
    return response_data


@app.route('/api/element_image', methods=['POST'])
def create_partical_iamge():
    # get the input data
    data = request.get_json()
    ui = data["name"]
    element_list = data["element_list"]
    filename = f'{ui}.jpg'
    base_image = Image.open(f'./static/background/{filename}')
    for element in element_list:
        element_image = Image.open(
            f'./static/elements/{ui}/element_{element["id"]}.jpg')
        element_image = element_image.convert('RGBA')
        base_image.paste(
            element_image, (element["left"], element["top"]), element_image)

    # Create an in-memory file object
    new_image_path = f"./static/partial_picture/{filename}"
    base_image.save(new_image_path)
    timestamp = int(time.time())

    # Return the new image
    response_data = {
        'image_url': f"{DEFAULT_URL}/static/partial_picture/{filename}?t={timestamp}"
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run()
