from flask import Flask, request, render_template
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Read the uploaded image
    img_stream = io.BytesIO()
    file.save(img_stream)
    img_stream.seek(0)
    img = Image.open(img_stream)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Perform edge detection
    edges = cv2.Canny(img, 100, 200)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to the bounding box
        cropped_img = img[y:y+h, x:x+w]

        # Draw a bounding box around the document
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cropped_img = img  # Use the original image if no contours are found
    
    
    # Example: Adjust the color (convert to grayscale)
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # Display the processed image
    processed_img = cv2.imencode('.jpg', gray_img)[1].tobytes()

    cv2.imwrite('processed_image.jpg', gray_img)
    # Save the processed image as a JPG file
    cv2.imwrite('bbox_image.jpg', img)
    
    # return processed_img

    return "Image processed and saved as processed_image.jpg"

if __name__ == '__main__':
    app.run(debug=True)
