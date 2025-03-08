import cv2

# Load models for face detection, age prediction, and gender prediction
faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

# Age and gender lists
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Function to detect faces and draw boxes around them, predicting gender and age
def faceBox(faceNet, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Draw box for high confidence faces
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)

            # Extract the face region
            face = frame[y1:y2, x1:x2]

            # Gender prediction
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [104, 117, 123], swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = GENDER_LIST[genderPreds[0].argmax()]

            # Age prediction
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = AGE_LIST[agePreds[0].argmax()]

            # Draw rectangle and display age/gender on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)  # Gold color
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

    return frame

# Start webcam video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame = faceBox(faceNet, frame)  # Apply face detection, age, and gender prediction
    cv2.imshow("Age-Gender Detection", frame)

    if cv2.waitKey(10) == ord('q'):  # Exit on 'q'
        break

video.release()
cv2.destroyAllWindows()
