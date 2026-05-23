from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'status': 'success', 'message': 'Test server is running'})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)