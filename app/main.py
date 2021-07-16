import os
import string
import random
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify, render_template
from app.worker import celery
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/app/temp'

dev_mode = True
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'

def allowed_file(filename):
    return '.' in filename and filename.endswith(".nii.gz")

def random_id(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            sample_id = random_id(20)
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], sample_id))
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], sample_id, "input"))
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], sample_id, "input", "orig"))
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], sample_id, "output"))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], sample_id, "input", "orig", "TOF.nii.gz"))
            task = celery.send_task('tasks.process', args=[sample_id], kwargs={})
            task_id = task.id
            return redirect(url_for('process_file', task_id=task_id, sample_id=sample_id))
    return render_template('home.html')


@app.route('/uploads/<filename>/', defaults={'sample_id': None, 'subdir': None})
@app.route('/uploads/<sample_id>/<subdir>/<filename>/')
def download_file(subdir, sample_id, filename):
    if sample_id is None:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    elif subdir == "input":
        return send_from_directory(os.path.join(app.config["UPLOAD_FOLDER"], sample_id, subdir, "orig"), filename)
    return send_from_directory(os.path.join(app.config["UPLOAD_FOLDER"], sample_id, subdir), filename)

@app.route('/processing/<sample_id>/<task_id>')
def process_file(sample_id, task_id):
    return render_template("process_file.html", task_id=task_id, sample_id=sample_id)

@app.route('/show/<sample_id>')
def show_file(sample_id):
    return render_template("show_file.html", sample_id=sample_id)

@app.route('/log/<sample_id>')
def get_log(sample_id):
    try:
        with open(os.path.join(app.config["UPLOAD_FOLDER"], sample_id, "log.txt"), "r") as f:
            log = f.read()
        return log
    except:
        return "0"

@app.route('/add/<int:param1>/<int:param2>')
def add(param1: int, param2: int) -> str:
    task = celery.send_task('tasks.add', args=[param1, param2], kwargs={})
    response = f"<a href='{url_for('check_task', task_id=task.id, external=True)}'>check status of {task.id} </a>"
    return response

@app.route('/check/<string:task_id>')
def check_task(task_id: str) -> str:
    res = celery.AsyncResult(task_id)
    return res.state

@app.route('/health_check')
def health_check() -> str:
        return jsonify("OK")
