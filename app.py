import os
from flask import Flask, render_template,request,jsonify

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", threaded=True)
