from flask import Flask, request, jsonify
from flask_cors import CORS

import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

SAVE_ANNO_DIR = "./data/annotations/"
SAVE_EVAL_DIR = "./data/evaluations"
if not os.path.exists(SAVE_ANNO_DIR):
    os.makedirs(SAVE_ANNO_DIR)
if not os.path.exists(SAVE_EVAL_DIR):
    os.makedirs(SAVE_EVAL_DIR)

@app.route('/save-json', methods=['POST'])
def save_json():
    
    raw_data = request.get_json(force=True)
    
    print(type(raw_data))
    
    single_data = raw_data.split('\n')
    
    for data in single_data:
        print(json.loads(data)['original'])
        
    return jsonify({"message": "JSON data saved successfully"}), 200
 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

