import face_recognition
import cv2



def no(path):
    image = face_recognition.load_image_file(path)
    face_landmarks_list = face_recognition.face_landmarks(image)
    print(face_landmarks_list)
    image = cv2.imread(path)

    color = (0, 0, 255)

    for i in face_landmarks_list:
        a = i.values()
        print(a)
        for j in a:
            for k in j:
                cv2.circle(image, (k[0], k[1]), 3, color, -1)


    cv2.imshow('Image with Point', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

no('05_face_recognize/je.png')

# no('05_face_recognize/s.png')

# no('05_face_recognize/no1.jpg')
# no('05_face_recognize/no2.jpeg')