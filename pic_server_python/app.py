from flask import Flask, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/picture/<int:pic_id>')
def serve_picture(pic_id):
    filename = f'pic{pic_id}.png'
    response = send_file(f'static/{filename}', mimetype='image/png')
    response.headers['Access-Control-Allow-Origin'] = '*'  # CORS
    return response


if __name__ == '__main__':
    app.run()
