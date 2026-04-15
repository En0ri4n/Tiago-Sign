from flask import Flask, request, jsonify
from logging import getLogger, basicConfig, INFO

basicConfig(level=INFO)
LOGGER = getLogger('werkzeug')
LOGGER.name = '[REST-API]'
port=32563
app = Flask('app')

@app.route('/api', methods=['POST'])
def App():
    data = request.json
    LOGGER.info(f'Data received from client: {data}')

    return jsonify({
            'status': 'success',
            'data': data
        })

app.run(port=port)
LOGGER.info(f'RestAPI started on port {port}')

