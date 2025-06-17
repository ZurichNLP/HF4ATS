from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pytz
from datetime import datetime

app = Flask(__name__)
CORS(app)


@app.route('/save-json/', methods=['POST'])
def save_json():

    zurich_tz = pytz.timezone("Europe/Zurich")
    time_stamp = datetime.now(zurich_tz).strftime("%Y-%m-%d-%H-%M-%S")
    
    try:
        data = request.get_json(force=True)
        split_data = data.split('\n')
       
        user_id = json.loads(split_data[0])['userID']
        phase_id = json.loads(split_data[0])['phaseID']
        dataset_id = json.loads(split_data[0])['datasetID']
        
        SAVE_ANNO_DIR = "./data/annotations/" + user_id + "/"
        SAVE_EVAL_DIR = "./data/evaluations/" + user_id + "/"
        
        if not os.path.exists(SAVE_ANNO_DIR):
            os.makedirs(SAVE_ANNO_DIR)
        if not os.path.exists(SAVE_EVAL_DIR):
            os.makedirs(SAVE_EVAL_DIR)
        
        file_name = ""
        if phase_id == "anno":
            file_name = os.path.join(SAVE_ANNO_DIR, f"labeled_{user_id}_{phase_id}_{dataset_id}_{time_stamp}.jsonl")
            
        elif phase_id == "eval":
            file_name = os.path.join(SAVE_EVAL_DIR, f"labeled_{user_id}_{phase_id}_{dataset_id}_{time_stamp}.jsonl")
            
        for split in split_data:
            json_line = json.loads(split)
            with open(file_name, 'a') as f:
                f.write(json.dumps(json_line) + '\n')
        
        return jsonify({"message": "JSON data saved successfully"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(ssl_context=('cert.pem', 'key.pem'), host='0.0.0.0', port=5000)

