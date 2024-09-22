
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import mediapipe as mp

app = Flask(__name__)

# Example size chart for demonstration purposes
size_chart = {
    'S': {'waist': 30, 'chest': 34},
    'M': {'waist': 34, 'chest': 38},
    'L': {'waist': 38, 'chest': 41},
    'XL': {'waist': 42, 'chest': 48},
    'XXL': {'waist': 46, 'chest': 51},
    'XXXL': {'waist': 50, 'chest': 53},
}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def find_best_size(measured_waist, measured_chest):
    """Find the best matching size based on waist and chest measurements."""
    best_match = None
    for size, measurements in size_chart.items():
        if measured_waist <= measurements['waist'] and measured_chest <= measurements['chest']:
            best_match = size
            break
    if not best_match:
        best_match = 'XL'  # Default to largest size if no match found
    return best_match

def process_image(image_path):
    """Process the uploaded image to detect reference circles and calculate dimensions."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or unable to read.")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,  # Minimum distance between circles
        param1=50,
        param2=30,
        minRadius=10,  # Adjust based on the printed circle size
        maxRadius=50   # Adjust based on the printed circle size
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"Detected Circles: {circles}")

        if len(circles) >= 2:
            # Sort circles by y-coordinate (assuming reference standard is horizontal)
            circles = sorted(circles, key=lambda x: x[0])  # Sort by x-coordinate
            ref_circle1, ref_circle2 = circles[:2]

            # Calculate the pixel distance between the two reference circles
            pixel_distance = np.sqrt((ref_circle2[0] - ref_circle1[0]) ** 2 + (ref_circle2[1] - ref_circle1[1]) ** 2)
            real_world_distance_cm = 10  # 10 cm

            pixels_per_cm = pixel_distance / real_world_distance_cm
            print(f"Pixel Distance: {pixel_distance}")
            print(f"Pixels per cm: {pixels_per_cm}")

            # Detect body landmarks using MediaPipe
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                print("No pose landmarks detected.")
                return None, None

            landmarks = results.pose_landmarks.landmark

            # Define key points for waist and chest
            # Using MediaPipe's Pose landmarks indices
            # For waist, we can use the midpoint between left and right hips
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Convert normalized landmarks to pixel coordinates
            image_height, image_width, _ = image.shape
            waist_x = (left_hip.x + right_hip.x) / 2 * image_width
            waist_y = (left_hip.y + right_hip.y) / 2 * image_height
            chest_x = (left_shoulder.x + right_shoulder.x) / 2 * image_width
            chest_y = (left_shoulder.y + right_shoulder.y) / 2 * image_height

            # Calculate waist and chest widths (Example: assuming horizontal measurement)
            # We'll take a fixed horizontal distance from the waist and chest points
            # Adjust the pixel_length based on your requirements

            # For waist, measure horizontal distance (e.g., 50 pixels left and right)
            waist_pixel_length = 100  # Example value; ideally, this should be dynamic or based on detected points
            real_world_waist = waist_pixel_length / pixels_per_cm

            # For chest, measure horizontal distance
            chest_pixel_length = 120  # Example value; ideally, this should be dynamic or based on detected points
            real_world_chest = chest_pixel_length / pixels_per_cm

            print(f"Real-world Waist: {real_world_waist} cm")
            print(f"Real-world Chest: {real_world_chest} cm")

            return real_world_waist, real_world_chest
    else:
        print("No circles detected.")
    
    return None, None

@app.route('/')
def index():
    return render_template('capture.html')

@app.route('/reference')
def reference():
    return render_template('reference.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f"Saved uploaded file to {file_path}")

        # Process the image
        waist, chest = process_image(file_path)
        if waist is not None and chest is not None:
            best_size = find_best_size(waist, chest)
            return jsonify({
                'waist': waist,
                'chest': chest,
                'recommended_size': best_size
            })
        else:
            return jsonify({'error': 'Could not detect reference standard or body landmarks in the image'}), 400

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
