import cv2

# Load the cascade - Algorithm to detection face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('kareem.mp4')

# To use a video file as input
while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.FONT_HERSHEY_SIMPLEX
    )
    # For condition for Draw the rectangle & Count & Display Number of Faces Detected
    face_count = 1
    for (x, y, w, h) in faces:
        # Draw the rectangle around each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (3, 77, 255), 2)

        # Count each number of faces
        cv2.putText(frame, 'Face #' + str(face_count), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
        face_count += 1

        # Display Number of Faces Detected -frame.shape[0] is height of photo
        cv2.putText(frame, "Number of Faces Detected: " + str(faces.shape[0]),
                    (120, frame.shape[0] - 18), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 255, 0), 1)
    # Display the output
    cv2.imshow('Faces Found In Video', frame)
    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
