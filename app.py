import pygal as pygal
from flask import Flask, render_template, request, jsonify
import json
import csv
from flask import Flask
from flask import render_template
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

def get_csv():
    csv_path = './result_datasets/recom_format.csv'
    csv_file = open(csv_path, 'r')
    csv_obj = csv.DictReader(csv_file)
    csv_list = list(csv_obj)
    return csv_list

@app.route("/")
def index():
    template = 'index.html'
    object_list = get_csv()
    return render_template(template, object_list=object_list)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload():
   if request.method == 'POST':
      f = request.files['file']
      csv_path='./{}'.format(f.filename)
      # print(f.filename)
      f.save(secure_filename(f.filename))
      data= np.genfromtxt(os.path.abspath(f.filename) , delimiter=',')
      graph = pygal.Line()
      csv_file = open(csv_path, 'r')
      csv_obj = csv.DictReader(csv_file)
      csv_list = list(csv_obj)

      template = 'index.html'
      object_list = csv_list
      return render_template(template, object_list=object_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True, use_reloader = True)
