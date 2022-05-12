import cv2



def detect_face(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(
        img, 1.05, 3, flags=None, minSize=(100, 100))

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        
        faces = img[y:y + h, x:x + w]

        faces = cv2.resize(
            faces, (224, 224))
    
    return faces























