import cv2
import time
from pygame import mixer

# Initialize alarm
mixer.init()
mixer.music.load(r"C:\Users\AM\Desktop\ESP\.idea\alarm_sound.mp3") # Load the alarm sound file from a specified path

# Load pre-trained Haar cascade classifiers for face, eyes, and yawn detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
yawn_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml') # Using 'smile' cascade as a proxy for yawn detection

# Configuration
ALARM_THRESHOLD = 0.5 # seconds to trigger alarm
start_time = None
alarm_triggered = False

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():    #Check Camera Connection
    exit("Camera error!")

# Resize the image for better performance on the Raspberry Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640px
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480px
cap.set(cv2.CAP_PROP_FPS, 15)  # Set the frame rate to 15 FPS to reduce processing load


# Functions for alarm
def play_alarm(): #Plays the alarm sound when drowsiness is detected.
    if not mixer.music.get_busy():
        mixer.music.play()

def stop_alarm():  #Stops the alarm sound when drowsiness is no longer detected.
    if mixer.music.get_busy():
        mixer.music.stop()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detects faces in the grayscale image.
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Flags to check detection status
    #eyes_detected and yawning_detected are initialized to False and will be set to True if eyes or yawning are detected.
    eyes_detected = False
    yawning_detected = False

    for (x, y, w, h) in faces: #it isolates the face area from the grayscale image (roi_gray), so further processing ( detecting eyes or yawning) can focus only on that region.
        roi_gray = gray[y:y + h, x:x + w]

        # Eye detection
        # this part detects eyes in the face region and marks eyes_detected as True if any eyes are found.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 8, minSize=(15, 15)) # Detect eyes in face region
        if len(eyes) > 0:
            eyes_detected = True

        # Yawning detection
        #If so, it sets yawning_detected to True, indicating a yawn.
        yawns = yawn_cascade.detectMultiScale(roi_gray, 1.7, 22) # Detect yawns (using smile cascade)
        if len(yawns) > 0:
            for (sx, sy, sw, sh) in yawns:
                aspect_ratio = sw / sh # Calculate aspect ratio of detected region
                if aspect_ratio > 2:  # Check if mouth appears wide open
                    yawning_detected = True

    # Debugging info and printing statement outputs of the current status of the eyes detection and yawning detection
    print(f"Eyes: {eyes_detected}, Yawning: {yawning_detected}")

    # Handle drowsiness detection
    if not eyes_detected or yawning_detected: # Trigger alarm if eyes closed or yawning detected
        if start_time is None:
            start_time = time.time() # Start timer when drowsiness detected
        elif time.time() - start_time >= ALARM_THRESHOLD: # Check if threshold exceeded
            if not alarm_triggered:
                play_alarm()
                alarm_triggered = True
    else:
        start_time = None # Reset timer if no drowsiness detected
        if alarm_triggered:
            stop_alarm()
            alarm_triggered = False

    # Display the video feed and exiting the loop and stop the video.
    cv2.imshow('Monitoring', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit loop on pressing 'q'
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
stop_alarm()
