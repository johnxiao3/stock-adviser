import os
from flask import Flask, render_template,request,jsonify
import subprocess
# app.py
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import time
import pytz
import json

app = Flask(__name__)
scheduler = BackgroundScheduler()

# Set timezone to EDT
edt = pytz.timezone("America/New_York")

# Variable to track the job status
job_status = {
    "last_run": None,
    "next_run": None,
    "time_until_next_run": None
}

def run_scheduled_task():
    global job_status
    # Run job_script.py as a detached process
    subprocess.Popen(['python3', 'screener7.py','deploymode'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.Popen(['python3', 'check_selling_status.py','deploymode'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Run the main task
    print("Running main task")
    time.sleep(2)  # Simulate task duration
    job_status["last_run"] = datetime.now(edt)

    # Update the next run time (next business day)
    next_run_time = get_next_business_day()
    job_status["next_run"] = next_run_time
    job_status["time_until_next_run"] = next_run_time - datetime.now(edt)

# Function to calculate the next business day with the desired time (16:40)
def get_next_business_day():
    now = datetime.now(edt)
    # Set the desired run time for today
    today_desired_time = now.replace(hour=16, minute=40, second=0, microsecond=0)

    # If the current time is before today's desired time and it's a weekday, use today
    if now < today_desired_time and now.weekday() < 5:  # 0-4 are weekdays
        return today_desired_time
    else:
        # Otherwise, calculate the next business day
        next_day = now + timedelta(days=1)
        while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_day += timedelta(days=1)
        return next_day.replace(hour=16, minute=40, second=0, microsecond=0)

next_run_time = get_next_business_day()
# Set up the job
scheduler.add_job(run_scheduled_task, 'interval', days=1, next_run_time=get_next_business_day())
scheduler.start()
# Initialize job status with the next run time
job_status["next_run"] = next_run_time
job_status["time_until_next_run"] = next_run_time - datetime.now(edt)


# Endpoint to get job status
@app.route('/status', methods=['GET'])
def get_status():
    if job_status["next_run"]:
        job_status["time_until_next_run"] = job_status["next_run"] - datetime.now(edt)
    
    # Convert timedelta to string for JSON serialization
    job_status["time_until_next_run"] = str(job_status["time_until_next_run"])

    last_run = job_status["last_run"].strftime("%Y-%m-%d %H:%M:%S %Z") if job_status["last_run"] else "N/A"
    next_run = job_status["next_run"].strftime("%Y-%m-%d %H:%M:%S %Z") if job_status["next_run"] else "N/A"
    time_until_next_run = str(job_status["time_until_next_run"])

    
    '''
    return jsonify({
        "last_run": job_status["last_run"].strftime("%Y-%m-%d %H:%M:%S %Z") if job_status["last_run"] else None,
        "next_run": job_status["next_run"].strftime("%Y-%m-%d %H:%M:%S %Z") if job_status["next_run"] else None,
        "time_until_next_run": job_status["time_until_next_run"]
    })
    '''
    return render_template('status.html', last_run=last_run, next_run=next_run, time_until_next_run=time_until_next_run)

@app.route('/log')
def log_page():
    log_file_path = os.path.join(app.static_folder, 'log.txt')  # Path to the log file in /static
    try:
        with open(log_file_path, 'r') as file:
            log_content = file.read()
    except FileNotFoundError:
        log_content = "Log file not found."
    
    return render_template('log_page.html', log_content=log_content)


# Define the path to your images directory
IMAGES_FOLDER = os.path.join('static', 'images')

def get_image_paths(date):
    # Define the path to the images directory
    image_folder = os.path.join('static', 'images', date)  # e.g., 'static/images/20241025'
    
    # Check if the directory exists
    if not os.path.exists(image_folder):
        return []  # Return an empty list if the directory does not exist
    
    # List all .png files in the directory
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    
    # Construct the full paths for the images
    image_paths = [os.path.join('/static','images', date, img) for img in image_files]  # e.g., 'images/20241025/image1.png'
    return image_paths

@app.route('/')
def home():
    subfolders = sorted([f.name for f in os.scandir(IMAGES_FOLDER) if f.is_dir()],reverse=True)
    return render_template('home.html', subfolders=subfolders)

@app.route('/images/<date>')
def image_gallery(date):
    # Get the list of images in the specified subfolder
    image_folder = os.path.join(IMAGES_FOLDER, date)
    if os.path.exists(image_folder):
        images = sorted([os.path.join(date, img) for img in os.listdir(image_folder) if img.endswith('.png')])
        return render_template('gallery.html', images=images[:10], total_images=len(images),date=date)  # Load first 10 images
    else:
        return "Folder not found", 404
    
@app.route('/images1/<date>')
def image_gallery_slide(date):
    images = get_image_paths(date)
    return render_template('gallery_slide.html', date=date, total_images=len(images))

@app.route('/api/images')
def api_images():
    date = request.args.get('date')
    index = int(request.args.get('index', 0))
    limit = int(request.args.get('limit', 10))
    
    images = get_image_paths(date)
    total_images = len(images)
    
    # Return a subset of images based on the index and limit
    return jsonify({
        'images': images[index:index + limit],
        'total_images': total_images
    })

@app.route('/api/save-starred', methods=['POST'])
def save_starred():
    data = request.json
    date = data['date']
    starred_images = data['starred']
    
    # Save to a file in the same directory as the images
    filepath = os.path.join('static', 'images', date, 'starred.json')
    with open(filepath, 'w') as f:
        json.dump(starred_images, f)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", threaded=True)
