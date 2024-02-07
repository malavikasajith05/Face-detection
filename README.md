<h1> Face detection with mask or no mask </h1><br>

This uses a CNN and code trains a deep learning model to detect whether a person is wearing a face mask or not in images. 

1. Data Preparation: It loads and prepares images of people with and without masks.

2. Data Splitting: Splits the dataset into training and testing sets.

3. Data Augmentation: Applies random image transformations to increase dataset diversity.

4. Base Model (Transfer Learning): Uses a pre-trained MobileNetV2 model as the base, removing its top layers.

5. Model Architecture: Adds new layers on top of the base for classification.

6. Freezing Base Model Layers: Prevents base model layers from being updated during training.

7. Model Compilation: Compiles the model with settings for optimization and loss.

8. Training: Trains the model on the training data with data augmentation.

9. Evaluation: Evaluates the model's performance on the test set and prints a classification report.

10. Saving the Model: Saves the trained model to a file.

11. Plotting Training History: Creates a plot showing training and validation loss and accuracy over epochs.
VISUALIZING OR TESTING IF IMAGES ARE DETECTED
1)	Video Stream Setup: 
The code initializes a video stream from the default camera (webcam).
•	Real-time Detection Loop:
It enters a loop to continuously capture video frames.
For each frame, it detects faces using a pre-trained face detection model.
It predicts whether each detected face is wearing a mask or not using a pre-trained face mask detection model.
•	Display Results: The code draws bounding boxes around detected faces and labels them as "Face-Detected with mask" or "Face-Detected with No Mask" based on the mask prediction.
•	Keyboard Input: The program listens for the "q" key press. If "q" is pressed, it breaks out of the loop and terminates the application.
•	Cleanup: After exiting the loop, the code closes windows and stops the video stream.
In summary, continuously analyzes video frames to detect faces and determine if people are wearing masks or not, displaying the results in real-time.
2)	Ditecting on a random picture:
•	Imports: The code imports necessary libraries, including OpenCV (cv2), TensorFlow/Keras, and NumPy, for image processing and model inference.
•	Function Definitions:
detect_and_predict_mask(image_path, faceNet, maskNet): This function takes the path to an input image, a face detection model (faceNet), and a face mask detection model (maskNet). It performs the following steps: Loads and preprocesses the input image.Detects faces in the image using the face detection model.Predicts whether each detected face is wearing a mask using the face mask detection model.Labels each detected face as "Face-Detected with mask" or "Face-Detected with No Mask" and draws bounding boxes around the faces.
•	Saves the output image with bounding boxes and labels and displays it.
•	detect_faces_and_predict_mask(frame, faceNet, maskNet): This helper function performs face detection and mask prediction on a single frame (image). It takes the frame, face detection model, and mask detection model as input and returns the locations of detected faces and their corresponding mask predictions.
