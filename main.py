import cv2 as cv

original_image = cv.imread('xmas2018.jpg')
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier(
    '//Applications/anaconda3/envs/face-detection-env/bin/'
    'haarcascade_frontalface_alt.xml'
)
detected_faces = face_cascade.detectMultiScale(grayscale_image)

for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )

cv.imshow('Image', original_image)
cv.waitKey(0)
cv.destroyAllWindows()
