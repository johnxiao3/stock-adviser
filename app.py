import os
from flask import Flask, render_template,request,jsonify
import subprocess
# app.py
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import time
import pytz
import json
from post_filtered import update_png
from check_selling_status import check_holding_stocks

# Function to log the information to a file
def log_to_file(log_message, log_file='./static/log.txt'):
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Get the current date and time in EDT
    edt = pytz.timezone('US/Eastern')
    current_time = datetime.now(edt).strftime("%Y-%m-%d %H:%M:%S %Z")

    # Create the final message with date and log
    log_entry = f"{current_time} - {log_message}\n"

    # Open the log file and append the log message
    with open(log_file, 'a') as f:
        f.write(log_entry)

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
    print("run scheduled")
    time.sleep(3)
    log_to_file("task run once")
    check_holding_stocks(1)
    time.sleep(2)  # Simulate task duration
    subprocess.Popen(['python3', 'screener7.py','1'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #subprocess.Popen(['python3', 'check_selling_status.py','deploymode'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Run the main task
    job_status["last_run"] = datetime.now(edt)
    # Update the next run time (next business day)
    next_run_time = get_next_business_day()
    job_status["next_run"] = next_run_time
    job_status["time_until_next_run"] = next_run_time - datetime.now(edt)

# Function to calculate the next business day with the desired time (16:40)
def get_next_business_day():
    runtime_hour,runtime_minute = 16,5
    now = datetime.now(edt)
    # Set the desired run time for today
    today_desired_time = now.replace(hour=runtime_hour, minute=runtime_minute, second=0, microsecond=0)
    print('now',now,today_desired_time)
    # If the current time is before today's desired time and it's a weekday, use today
    if now < today_desired_time-timedelta(minutes=1) and now.weekday() < 5:  # 0-4 are weekdays
        return today_desired_time
    else:
        # Otherwise, calculate the next business day
        next_day = now + timedelta(days=1)
        while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_day += timedelta(days=1)
        return next_day.replace(hour=runtime_hour, minute=runtime_minute, second=0, microsecond=0)

next_run_time = get_next_business_day()
# Set up the job
scheduler.add_job(run_scheduled_task, 'interval', days=1, 
                  next_run_time=get_next_business_day())
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

@app.route('/onestock')
def one_stock():
    return render_template('one_stock.html')


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
    ratings = data.get('ratings', {})  # Get ratings if present, empty dict if not
    notes = data.get('notes', {})      # Get notes if present, empty dict if not
    
    # Save to a JSON file in the same directory as the images
    filepath = os.path.join('static', 'images', date, 'starred.json')
    save_data = {
        'starred': starred_images,
        'ratings': ratings,
        'notes': notes
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)  # Added indent for better readability
    
    return jsonify({'success': True})

def load_starred_data(date):
    try:
        # File format now includes notes:
        # {
        #   "starred": ["image1.png", "image2.png"],
        #   "ratings": {"image1.png": 4, "image2.png": 5},
        #   "notes": {"image1.png": "This is a note", "image2.png": "Another note"}
        # }
        filepath = os.path.join('static', 'images', date, 'starred.json')
        print(f"Loading starred data for date {date} from {filepath}")
        
        with open(filepath, 'r') as file:
            data = json.load(file)
            print(f"Loaded data: {data}")
            
            # Ensure all expected fields exist, even if file is from old version
            if 'starred' not in data:
                data['starred'] = []
            if 'ratings' not in data:
                data['ratings'] = {}
            if 'notes' not in data:
                data['notes'] = {}
                
            return data
            
    except FileNotFoundError:
        print(f"No starred data found for date {date}")
        return {
            "starred": [],
            "ratings": {},
            "notes": {}
        }
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for date {date}: {e}")
        return {
            "starred": [],
            "ratings": {},
            "notes": {}
        }

@app.route('/api/get-starred', methods=['GET'])
def get_starred():
    # Retrieve date from query parameters
    date = request.args.get('date')
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400
        
    # Load starred data for the specified date
    starred_data = load_starred_data(date)
    
    # Return JSON response with starred images, ratings, and notes
    response = {
        "starred": starred_data["starred"],
        "ratings": starred_data["ratings"],
        "notes": starred_data["notes"]
    }
    
    return jsonify(response)


@app.route('/api/update-image', methods=['POST'])
def update_image_endpoint():
    try:
        data = request.get_json()
        date = data.get('date')
        filename = data.get('filename')
        
        if not date or not filename:
            return jsonify({'error': 'Missing date or filename'}), 400
            
        # Call your update function
        result = update_png(date, filename,0)
        result = True
        print(date,filename)
        
        if result:
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Update failed'}), 500
            
    except Exception as e:
        print(f"Error updating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-image1', methods=['POST'])
def update_image_endpoint1():
    try:
        data = request.get_json()
        date = data.get('date')
        filename = data.get('filename')
        
        if not date or not filename:
            return jsonify({'error': 'Missing date or filename'}), 400
            
        # Call your update function
        print(date,filename)
        result = update_png(date, filename,1)
        result = True
        
        if result:
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Update failed'}), 500
            
    except Exception as e:
        print(f"Error updating image: {str(e)}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    #if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    #scheduler.start()
    app.run(debug=False,host="0.0.0.0", threaded=True)
