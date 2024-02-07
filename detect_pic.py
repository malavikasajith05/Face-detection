# Import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def detect_and_predict_mask(image_path, faceNet, maskNet):
    # Load and preprocess the image
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (400, 400))
    
    # Detect faces in the frame and determine if they are wearing a mask
    (locs, preds) = detect_faces_and_predict_mask(frame, faceNet, maskNet)
    
    # Loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # Unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
    
        # Determine the class label and color
        label = "Face-Detected with mask" if mask > withoutMask else "Face-Detected with No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    
        # Display the label and bounding box rectangle on the image
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # Save or display the output image
    output_path = "output.jpg"
    cv2.imwrite(output_path, frame)
    cv2.imshow("Output", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # Initialize lists for faces, their corresponding locations, and predictions
    faces = []
    locs = []
    preds = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
    
        # Filter out weak detections by ensuring confidence is above a threshold
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    
            # Extract the face ROI, convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
    
            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    # Make batch predictions on all faces
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    return (locs, preds)

# Load our serialized face detector model and mask detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

# Path to the input image
image_path = "12.jpg"

# Detect faces and predict mask usage in the input image
detect_and_predict_mask(image_path, faceNet, maskNet)
