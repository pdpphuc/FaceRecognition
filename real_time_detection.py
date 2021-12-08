import cv2
import dlib
from imutils import face_utils

class RealTimeDetection:

    def __init__(self):
        # Load the detector
        detector = dlib.get_frontal_face_detector()

        # Load the predictor
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # read the image
        cap = cv2.VideoCapture(0)

        while True:
            # load the input image and convert it to grayscale
            _, frame = cap.read()
            # Convert image into grayscale
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

            # Use detector to find landmarks
            faces = detector(gray)

            for face in faces:
                x1 = face.left()  # left point
                y1 = face.top()  # top point
                x2 = face.right()  # right point
                y2 = face.bottom()  # bottom point
                # Draw a rectangle
                cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(
                    x2, y2), color=(0, 255, 0), thickness=4)

                face_features = predictor(image=gray, box=face)

                # Loop through all 68 points
                for n in range(0, 68):
                    x = face_features.part(n).x
                    y = face_features.part(n).y

                    # Draw a circle
                    cv2.circle(img=frame, center=(x, y), radius=2,
                                color=(255, 0, 0), thickness=1)

                # # determine the facial landmarks for the face region, then
                # # convert the facial landmark (x, y)-coordinates to a NumPy
                # # array
                # shape = predictor(gray, face)
                # shape = face_utils.shape_to_np(shape)
            
                # # loop over the (x, y)-coordinates for the facial landmarks
                # # and draw them on the image
                # for (x, y) in shape:
                #     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # show the image
            cv2.imshow(winname="Face Recognition App", mat=frame)

            # Exit when escape is pressed
            if cv2.waitKey(delay=1) == 27:
                break

        # When everything done, release the video capture and video write objects
        cap.release()

        # Close all windows
        cv2.destroyAllWindows()