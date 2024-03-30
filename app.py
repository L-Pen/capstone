from flask import Flask, render_template, request, jsonify, make_response
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/file-input", methods=['POST'])
def input_file():
    if request.method == 'POST':
        response = make_response("lol")
        response.status_code = 200
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        response.headers["Access-Control-Allow-Methods"] = ["POST"]
        response.headers["Access-Control-Allow-Headers"] = ["Origin", "X-Requested-With", "Content-Type", "Accept", "Authorization"]
        try:
            bytes_np_dec = request.files.get("binary_data").read();
            file_path = "./test.h5"
            with open(file_path, "wb") as file:
                file.write(bytes_np_dec)
            data = pd.read_hdf(file_path, key="df")
            data = data.loc[(data.index.get_level_values('CellType') == 'OT-1')]
            print(data);
            print("Done")
        except Exception as e:
            print(e)
        return response

@app.route("/setup-model-environment", methods=['POST'])
def setup_model_environment():
    return



if __name__ == '__main__':
    app.run(debug=True, port=8001)