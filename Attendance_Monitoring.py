import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime, timedelta
import pandas as pd


DATA_YAML_PATH = r"C:\Users\cmgau\Desktop\Cp\BIT Mesra\FootwearSafetyProject\data.yaml"
PERSON_DETECTOR_MODEL = 'yolov8n.pt'

TRAINED_SHOE_MODEL_PATH = r"C:\Users\cmgau\Desktop\Cp\BIT Mesra\FootwearSafetyProject\runs\detect\train\weights\best.pt"

CONF_THRESHOLD_PERSON = 0.5
CONF_THRESHOLD_SHOE = 0.3    
IOU_THRESHOLD = 0.5           

FOOT_REGION_HEIGHT_RATIO = 0.35


KNOWN_FACES_DIR = r"C:\Users\cmgau\Desktop\Cp\BIT Mesra\FootwearSafetyProject\known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

STUDENT_ROSTER_CSV = r"C:\Users\cmgau\Desktop\Cp\BIT Mesra\FootwearSafetyProject\student_roster.csv"

ATTENDANCE_PHOTOS_DIR = r"C:\Users\cmgau\Desktop\Cp\BIT Mesra\FootwearSafetyProject\attendance_photos"
os.makedirs(ATTENDANCE_PHOTOS_DIR, exist_ok=True)

NON_COMPLIANT_OUTPUT_DIR = r"C:\Users\cmgau\Desktop\Cp\BIT Mesra\FootwearSafetyProject\runs\detect\non_compliant"
os.makedirs(NON_COMPLIANT_OUTPUT_DIR, exist_ok=True)

ATTENDANCE_COOLDOWN_MINUTES = 60 
SHOES_ALERT_COOLDOWN_MINUTES = 10


student_cooldown_tracker = {}
# Format: { "student_name": { 'last_seen_timestamp', 'last_attendance_log_timestamp', 'last_shoe_alert_timestamp' } }

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create() # Using LBPH for simplicity and speed

# --- Helper Functions ---

def save_non_compliant_person(person_crop, detection_status_text):
    """
    Saves the cropped image of a non-compliant person to the NON_COMPLIANT_OUTPUT_DIR
    with a unique timestamped filename. This is specifically for safety violations.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"non_compliant_{timestamp}.jpg"
    filepath = os.path.join(NON_COMPLIANT_OUTPUT_DIR, filename)

    temp_crop = person_crop.copy()
    cv2.putText(temp_crop, detection_status_text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    if temp_crop.shape[0] > 0 and temp_crop.shape[1] > 0:
        cv2.imwrite(filepath, temp_crop)
    else:
        print(f"Warning: Attempted to save an empty non-compliant crop for {filename}.")


def save_attendance_photo(person_crop, student_name):
    """
    Saves the full cropped person image for attendance purposes to ATTENDANCE_PHOTOS_DIR.
    Returns the full path to the saved image.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{student_name}_attendance_{timestamp}.jpg"
    filepath = os.path.join(ATTENDANCE_PHOTOS_DIR, filename)
    
    # Ensure the crop is valid before saving
    if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
        cv2.imwrite(filepath, person_crop)
        return filepath
    else:
        print(f"Warning: Attempted to save empty person crop for {student_name} for attendance.")
        return "" # Return empty string if crop is invalid


def update_student_roster(student_name, attendance_status, shoes_status, df_roster, person_crop_for_save=None):
    """
    Updates the pandas DataFrame with the latest status for a student.
    Handles cooldowns for attendance and shoe status.
    Returns the updated DataFrame.
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    student_row_index = df_roster[df_roster['Name'] == student_name].index

    if not student_row_index.empty:
        idx = student_row_index[0]

        # Initialize cooldown tracker for student if not present
        if student_name not in student_cooldown_tracker:
            student_cooldown_tracker[student_name] = {
                'last_seen_timestamp': datetime.min,
                'last_attendance_log_timestamp': datetime.min,
                'last_shoe_alert_timestamp': datetime.min
            }

        # --- Handle Attendance Update and Photo Saving ---
        last_attendance_time = student_cooldown_tracker[student_name].get('last_attendance_log_timestamp')
        if (now - last_attendance_time) > timedelta(minutes=ATTENDANCE_COOLDOWN_MINUTES):
            df_roster.loc[idx, 'Attendance_Status'] = attendance_status
            df_roster.loc[idx, 'Timestamp'] = timestamp_str # Update timestamp for general activity
            
            # Save the full person crop for attendance if provided and valid
            if person_crop_for_save is not None and person_crop_for_save.size > 0:
                photo_path = save_attendance_photo(person_crop_for_save, student_name)
                df_roster.loc[idx, 'Photo_Path'] = photo_path 
                print(f"Excel Update: {student_name} attendance set to '{attendance_status}', photo saved to {photo_path}.")
            else:
                print(f"Excel Update: {student_name} attendance set to '{attendance_status}'. No valid person crop for photo save.")

            student_cooldown_tracker[student_name]['last_attendance_log_timestamp'] = now
        else:
            print(f"Suppressing attendance update for {student_name} (cooldown active).")

        # --- Handle Shoes Status Update and Alerts (if missing) ---
        ### Fix Start ###
        # Always update the shoes status in the DataFrame based on current detection
        df_roster.loc[idx, 'Shoes_Status'] = shoes_status
        df_roster.loc[idx, 'Timestamp'] = timestamp_str # Update timestamp to show recent activity

        if shoes_status == "MISSING":
            last_shoe_alert_time = student_cooldown_tracker[student_name].get('last_shoe_alert_timestamp')
            if (now - last_shoe_alert_time) > timedelta(minutes=SHOES_ALERT_COOLDOWN_MINUTES):

                student_cooldown_tracker[student_name]['last_shoe_alert_timestamp'] = now
                print(f"\nALERT: {student_name} detected with '{shoes_status}' shoes! Updating Excel.")
            else:
                print(f"Suppressing shoe alert update for {student_name} (cooldown active for alerts).")
        else: 
            pass
    return df_roster


def train_face_recognizer():
    """
    Trains the LBPH face recognizer using images in the KNOWN_FACES_DIR.
    Returns the trained recognizer and a mapping from label IDs to student names.
    """
    print("\n--- Training Face Recognizer ---")
    faces = []
    labels = []
    student_id_to_name = {} # Map integer ID (label) to student name
    current_id = 0
    label_map = {} # Map student name to integer ID

    if not os.path.exists(KNOWN_FACES_DIR) or not os.listdir(KNOWN_FACES_DIR):
        print(f"Warning: '{KNOWN_FACES_DIR}' is empty or does not exist. Face recognition will be disabled.")
        return None, None

    for student_name_dir in os.listdir(KNOWN_FACES_DIR):
        student_path = os.path.join(KNOWN_FACES_DIR, student_name_dir)
        if os.path.isdir(student_path): # Ensure it's a directory (student's folder)
            if student_name_dir not in label_map:
                label_map[student_name_dir] = current_id
                student_id_to_name[current_id] = student_name_dir
                current_id += 1
            
            label = label_map[student_name_dir]

            for image_name in os.listdir(student_path):
                image_path = os.path.join(student_path, image_name)
                # Ensure it's a recognized image file type
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    print(f"Skipping non-image file: {image_path}")
                    continue
                
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read image {image_path}, skipping.")
                        continue
                    
                    # Detect faces in the training image
                    detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(detected_faces) == 0:
                        print(f"Warning: No face found in training image {image_path}. Skipping.")
                        continue

                    # For training, ideally there's only one prominent face per image.
                    # We'll take the largest one if multiple are found.
                    (x, y, w, h) = max(detected_faces, key=lambda rect: rect[2] * rect[3])
                    faces.append(img[y:y+h, x:x+w])
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    if not faces:
        print("No valid faces found for training. Ensure 'known_faces' directory is populated with valid images.")
        return None, None

    try:
        face_recognizer.train(faces, np.array(labels))
        print(f"Face recognizer trained with {len(faces)} faces for {len(student_id_to_name)} unique students.")
        return face_recognizer, student_id_to_name
    except cv2.error as e:
        print(f"OpenCV error during face recognizer training: {e}")
        print("This often happens if you have too few training images or some are invalid.")
        return None, None
    except Exception as e:
        print(f"General error training face recognizer: {e}")
        return None, None

# --- Main Processing Logic ---

def process_single_frame(frame, person_model, shoe_model, person_class_id, shoe_class_name, 
                         face_recognizer_instance, student_id_map, face_cascade_detector, df_roster):
    """
    Helper function to process a single frame for person, face, and shoe detection,
    drawing bounding boxes, confidence scores, and safety status.
    Handles student identification, attendance, and non-compliance alerts,
    and updates the roster DataFrame.
    Returns the annotated frame and the updated DataFrame.
    """
    annotated_frame = frame.copy()
    h, w, _ = frame.shape
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    person_results = person_model.predict(
        source=frame, conf=CONF_THRESHOLD_PERSON, iou=IOU_THRESHOLD,
        classes=[person_class_id] if person_class_id is not None else None, verbose=False
    )

    # Process each detected person in the frame
    for person_r in person_results:
        person_boxes = person_r.boxes
        for p_box in person_boxes:
            px1, py1, px2, py2 = map(int, p_box.xyxy[0])
            # Ensure coordinates are within frame bounds
            px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)

            person_crop_full = frame[py1:py2, px1:px2].copy() # Get the full person crop (color)
            person_gray_crop = gray_frame[py1:py2, px1:px2].copy() # Get the full person crop (grayscale)

            student_name = "Unknown Student"
            
            # Detect faces within the detected person bounding box
            faces_in_person_crop = face_cascade_detector.detectMultiScale(person_gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces_in_person_crop) > 0:
                # Assuming the largest face is the primary one if multiple are found
                (fx, fy, fw, fh) = max(faces_in_person_crop, key=lambda rect: rect[2] * rect[3])
                face_crop_for_recognition = person_gray_crop[fy:fy+fh, fx:fx+fw]
                
                if face_recognizer_instance and student_id_map:
                    try:
                        label, confidence = face_recognizer_instance.predict(face_crop_for_recognition)
                        # A lower confidence score indicates a better match for LBPH
                        if confidence < 100: # This threshold might need adjustment (e.g., < 80 for stricter)
                            student_name = student_id_map.get(label, f"Unknown ID {label}")
                            cv2.putText(annotated_frame, f"{student_name} ({confidence:.0f})", (px1, py1 - 45),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Yellow for known
                            cv2.rectangle(annotated_frame, (px1 + fx, py1 + fy), (px1 + fx + fw, py1 + fy + fh), (255, 255, 0), 2)
                            
                            # --- Update attendance and save the FULL PERSON photo here ---
                            # Pass the full person crop for attendance photo
                            df_roster = update_student_roster(student_name, "Present", "", df_roster, person_crop_full)

                        else:
                            # Face recognized but confidence is too low to be certain
                            cv2.putText(annotated_frame, "Unknown Face", (px1, py1 - 45),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange for unknown
                            cv2.rectangle(annotated_frame, (px1 + fx, py1 + fy), (px1 + fx + fw, py1 + fy + fh), (0, 165, 255), 2)
                    except cv2.error as e:
                        # Error during prediction (e.g., face crop too small/invalid)
                        cv2.putText(annotated_frame, "Face Rec Error", (px1, py1 - 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red for error
                else:
                    # Recognizer not trained/available
                    cv2.putText(annotated_frame, "No Recognizer", (px1, py1 - 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # No face detected within the person bounding box
                cv2.putText(annotated_frame, "No Face", (px1, py1 - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Draw bounding box for the person
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (255, 100, 0), 2) # Blue-ish for person
            cv2.putText(annotated_frame, "Person", (px1, py1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            # --- Footwear Detection ---
            # Define the "foot region" at the bottom of the person's bounding box
            foot_region_y_start = int(py1 + (py2 - py1) * (1 - FOOT_REGION_HEIGHT_RATIO))
            foot_region_y_end = py2
            foot_region_x_start = px1
            foot_region_x_end = px2

            # Ensure foot region coordinates are within frame bounds
            foot_region_y_start = max(0, foot_region_y_start)
            foot_region_x_start = max(0, foot_region_x_start)
            foot_region_y_end = min(h, foot_region_y_end)
            foot_region_x_end = min(w, foot_region_x_end)

            foot_crop = frame[foot_region_y_start:foot_region_y_end, foot_region_x_start:foot_region_x_end]

            shoe_detected_in_region = False
            shoe_detections_info = []

            if foot_crop.shape[0] > 0 and foot_crop.shape[1] > 0: # Ensure foot crop is not empty
                shoe_results_on_crop = shoe_model.predict(
                    source=foot_crop, conf=CONF_THRESHOLD_SHOE, iou=IOU_THRESHOLD, verbose=False
                )

                for shoe_r_crop in shoe_results_on_crop:
                    shoe_boxes_crop = shoe_r_crop.boxes
                    if len(shoe_boxes_crop) > 0:
                        shoe_detected_in_region = True
                        for s_box_crop in shoe_boxes_crop:
                            sx1_crop, sy1_crop, sx2_crop, sy2_crop = map(int, s_box_crop.xyxy[0])
                            s_conf = float(s_box_crop.conf[0])

                            # Translate shoe crop coordinates back to original frame coordinates
                            sx1_orig = sx1_crop + foot_region_x_start
                            sy1_orig = sy1_crop + foot_region_y_start
                            sx2_orig = sx2_crop + foot_region_x_start
                            sy2_orig = sy2_crop + foot_region_y_start

                            shoe_label = f"{shoe_class_name} ({s_conf:.1f}%)"
                            cv2.rectangle(annotated_frame, (sx1_orig, sy1_orig), (sx2_orig, sy2_orig), (0, 255, 0), 2) # Green for detected shoes
                            cv2.putText(annotated_frame, shoe_label, (sx1_orig, sy1_orig - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            shoe_detections_info.append(f"{shoe_class_name} ({s_conf:.1f}%)")

            # Determine overall footwear status and display
            status_text = ""
            status_color = (0, 0, 255) # Default Red for "NOT DETECTED" (implies safety issue)
            
            current_shoe_status = "MISSING" if not shoe_detected_in_region else "PRESENT"

            if shoe_detected_in_region:
                status_text = f"Footwear: APPROVED - {', '.join(shoe_detections_info)}"
                status_color = (0, 255, 0) # Green for approved
            else:
                status_text = "Footwear: NOT DETECTED (Safety Issue!)"
                # Trigger alert for non-compliant footwear, respecting cooldown, and save image
                if student_name != "Unknown Student": # Only alert for identified students
                    now = datetime.now()
                    # Initialize student_cooldown_tracker entry if it doesn't exist for this student
                    if student_name not in student_cooldown_tracker:
                        student_cooldown_tracker[student_name] = {'last_shoe_alert_timestamp': datetime.min}
                    
                    last_alert_time = student_cooldown_tracker[student_name].get('last_shoe_alert_timestamp')
                    if (now - last_alert_time) > timedelta(minutes=SHOES_ALERT_COOLDOWN_MINUTES):
                        print(f"\nALERT: {student_name} detected with MISSING shoes! Saving image to '{NON_COMPLIANT_OUTPUT_DIR}'.")
                        # This saves the full person crop into the designated non-compliant folder
                        save_non_compliant_person(person_crop_full, status_text)
                        student_cooldown_tracker[student_name]['last_shoe_alert_timestamp'] = now
                    else:
                        print(f"Suppressing image save/alert for {student_name} (shoe cooldown active).")

            # Display footwear status text and draw rectangle around foot region
            cv2.putText(annotated_frame, status_text, (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.rectangle(annotated_frame, (foot_region_x_start, foot_region_y_start),
                          (foot_region_x_end, foot_region_y_end), status_color, 2)
            
            # Update shoes status in roster. Attendance photo logic is separate.
            # Pass the current_shoe_status directly to update_student_roster.
            if student_name != "Unknown Student":
                df_roster = update_student_roster(student_name, "Present", current_shoe_status, df_roster, None) # Pass None for person_crop here as attendance photo is handled above specifically when attendance is logged.
        
    return annotated_frame, df_roster


def run_footwear_safety_detection_pipeline(source_path='0'):
    """
    Runs the footwear safety detection pipeline for live camera footage.
    It detects persons, checks footwear, identifies students, and handles
    attendance and non-compliance alerts, updating a CSV roster.
    """
    print(f"\n--- Starting Footwear Safety Detection Pipeline ---")
    print(f"Using Person Detector: {PERSON_DETECTOR_MODEL}")
    print(f"Using Custom Shoe Detector: {TRAINED_SHOE_MODEL_PATH}")
    print(f"Known Faces Directory for Training: {KNOWN_FACES_DIR}")
    print(f"Student Roster CSV: {STUDENT_ROSTER_CSV}")
    print(f"Attendance Photos Directory: {ATTENDANCE_PHOTOS_DIR}")
    print(f"Non-Compliant Photos Directory: {NON_COMPLIANT_OUTPUT_DIR}")


    # Load Person Detection Model
    try:
        person_model = YOLO(PERSON_DETECTOR_MODEL)
        print(f"Person detector loaded. Classes: {person_model.names}")
        person_class_id = None
        # Find the ID for the 'person' class in YOLOv8's default COCO dataset
        for idx, name in person_model.names.items():
            if name == 'person':
                person_class_id = idx
                break
        if person_class_id is None:
            print(f"Warning: 'person' class not found in {PERSON_DETECTOR_MODEL}. Ensure it's a COCO-trained model.")
            print("Person detection might not be strictly filtered for 'person' objects by class ID.")
    except Exception as e:
        print(f"Error loading person detection model '{PERSON_DETECTOR_MODEL}': {e}")
        return

    # Load Custom Shoe Detection Model
    if not os.path.exists(TRAINED_SHOE_MODEL_PATH):
        print(f"Error: Trained shoe detector weights not found at '{TRAINED_SHOE_MODEL_PATH}'.")
        print("Please ensure you have trained your custom model (e.g., by running a 'train_shoes_model.py' script),")
        print(f"and that the 'TRAINED_SHOE_MODEL_PATH' variable is correctly updated (e.g., 'runs/detect/train/weights/best.pt').")
        return
    try:
        shoe_model = YOLO(TRAINED_SHOE_MODEL_PATH)
        print(f"Shoe detector loaded. Classes: {shoe_model.names}")
        # Assuming your custom model has 'shoes' as class 0
        shoe_class_name = shoe_model.names.get(0, 'shoes')
        if not shoe_model.names:
             print("Warning: Shoe model classes are empty. Ensure your custom model is trained correctly.")
             print("Defaulting shoe_class_name to 'shoes'.")

    except Exception as e:
        print(f"Error loading custom shoe detector model '{TRAINED_SHOE_MODEL_PATH}': {e}")
        return

    # Train face recognizer (or load if pre-trained)
    global face_recognizer, student_id_to_name # Access global variables
    face_recognizer, student_id_to_name = train_face_recognizer()
    if face_recognizer is None:
        print("Face recognizer not trained/loaded. Student identification will be limited or unavailable.")

    # Load Student Roster from CSV
    if not os.path.exists(STUDENT_ROSTER_CSV):
        print(f"Error: Student roster CSV not found at '{STUDENT_ROSTER_CSV}'.")
        print("Please create 'student_roster.csv' manually in your project directory with the following headers:")
        print("Roll_No,Name,DOB,Branch,Attendance_Status,Shoes_Status,Timestamp,Photo_Path")
        print("Example row: 1,Gautham,31/02/2004,ER,,,\"\",\"\"")
        return
    
    try:
        df_roster = pd.read_csv(STUDENT_ROSTER_CSV)
        # Ensure all required columns exist, adding them if they are missing (e.g., for a new CSV)
        for col in ['Attendance_Status', 'Shoes_Status', 'Timestamp', 'Photo_Path']:
            if col not in df_roster.columns:
                df_roster[col] = '' # Initialize with empty string
        print(f"Student roster loaded successfully from {STUDENT_ROSTER_CSV}")
    except Exception as e:
        print(f"Error loading student roster CSV '{STUDENT_ROSTER_CSV}': {e}")
        print("Please ensure the CSV is correctly formatted and accessible.")
        return

    # Initialize Video Capture (e.g., webcam)
    # '0' typically refers to the default webcam. Change if you have multiple cameras or a video file.
    cap = cv2.VideoCapture(int(source_path) if isinstance(source_path, str) and source_path.isdigit() else source_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source/webcam at {source_path}. Please check connection or source path.")
        return

    print("\nLive detection started. Press 'q' key in the video window to quit.")

    # --- Main Detection Loop ---
    while True:
        ret, frame = cap.read() # Read a frame from the camera
        if not ret:
            print("Failed to grab frame from video source. Exiting.")
            break

        # Process the current frame and get the updated annotated frame and roster DataFrame
        annotated_frame, df_roster = process_single_frame(
            frame, person_model, shoe_model, person_class_id, shoe_class_name,
            face_recognizer, student_id_to_name, face_cascade, df_roster
        )

        cv2.imshow('Student Lab Safety Monitor (Live)', annotated_frame) # Display the annotated frame

        if cv2.waitKey(1) & 0xFF == ord('q'): # Wait for 'q' key press to quit
            break

    # --- Cleanup and Save on Exit ---
    cap.release() # Release the camera
    cv2.destroyAllWindows() # Close all OpenCV windows
    
    # Save the updated roster to CSV upon graceful exit
    try:
        df_roster.to_csv(STUDENT_ROSTER_CSV, index=False) # index=False prevents writing DataFrame index as a column
        print(f"\nFinal student roster saved to {STUDENT_ROSTER_CSV}")
    except Exception as e:
        print(f"Error saving final student roster to CSV: {e}")

    print("Student lab safety monitoring session ended.")

# This block ensures that run_footwear_safety_detection_pipeline() is called only when the script is executed directly.
if __name__ == "__main__":
    # --- IMPORTANT Setup Checklist BEFORE Running ---
    # 1. Update TRAINED_SHOE_MODEL_PATH: Ensure it points to your actual 'best.pt' file
    #    from your YOLOv8 shoe training (e.g., 'runs/detect/train/weights/best.pt').
    # 2. Prepare 'known_faces' directory: Create this folder in your project root.
    #    Inside 'known_faces', create a separate sub-folder for EACH known student
    #    (e.g., 'known_faces/Gautham/', 'known_faces/Sanidhya/').
    # 3. Add student images: Place MULTIPLE (5-10+) clear, varied face images
    #    of each student in their respective sub-folders (e.g., 'known_faces/Gautham/gautham1.jpg').
    #    The quality and quantity of these images directly impact face recognition accuracy.
    # 4. Create 'student_roster.csv' manually: In your project root, create a file
    #    named 'student_roster.csv' with the EXACT header row below, and your
    #    initial student data. Leave Attendance_Status, Shoes_Status, and Timestamp
    #    blank, and Photo_Path blank with ""
    #    Example content for 'student_roster.csv':
    #    Roll_No,Name,DOB,Branch,Attendance_Status,Shoes_Status,Timestamp,Photo_Path
    #    1,Gautham,31/02/2004,ER,,,"",""
    #    2,Sanidhya,31/06/2005,CS,,,"",""
    #
    # 5. Ensure all necessary Python libraries are installed:
    #    pip install opencv-python opencv-contrib-python ultralytics pandas

    # Call the main pipeline function, using '0' for the default webcam
    run_footwear_safety_detection_pipeline(source_path='0')