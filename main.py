import cv2
import os
import time

# Frame dimensions
frameWidth = 1000  # Frame Width
frameHeight = 480  # Frame Height

# Load the Haar Cascade for number plate detection
plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0

# Get the Downloads folder path
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

while True:
    success, img = cap.read()

    if not success:
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            # Draw rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]

            # Save the detected number plate image
            file_path = os.path.join(downloads_folder, f"NumberPlate_{count}.jpg")
            cv2.imwrite(file_path, imgRoi)
            count += 1

            # Display a message indicating the scan was saved
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)

    # Display the result frame
    cv2.imshow("Result", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
