from flask import Flask, render_template, send_from_directory
import os
from datetime import datetime

app = Flask(__name__)

def get_latest_experiment_dir():
    today = datetime.now().strftime('%Y-%m-%d')
    log_dir = os.path.join('../logs', today)
    if not os.path.exists(log_dir):
        return None
    experiments = sorted(os.listdir(log_dir), reverse=True)
    if not experiments:
        return None
    return os.path.join(log_dir, experiments[0])

@app.route('/')
def index():
    experiment_dir = get_latest_experiment_dir()
    if not experiment_dir:
        return "No experiment directory found"
    with open(os.path.join(experiment_dir, 'experiment.log'), 'r') as f:
        log_content = f.read()
    images = [f for f in os.listdir(experiment_dir) if f.endswith('.png')]
    return render_template('index.html', log_content=log_content, images=images)

@app.route('/images/<path:filename>')
def images(filename):
    experiment_dir = get_latest_experiment_dir()
    return send_from_directory(experiment_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)