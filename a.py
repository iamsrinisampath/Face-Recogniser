# import smtplib
import os.path
import cv2 as cv
# import imghdr
# from email.mime.text import MIMEText
# from email.mime.image import MIMEImage
# from email.mime.multipart import MIMEMultipart

# EMAIL_ADDRESS = "opencvmini@gmail.com"
# EMAIL_PASSWORD = "mukemaSS"
# ImgFileName = 'Intruder.jpg'

# path = 'C:/Users/Srinivasan/Desktop/Mini-Project/a.py'
# dirname = os.path.dirname(__file__)
# print(dirname)    

# with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
#     smtp.ehlo()
#     smtp.starttls()
#     smtp.ehlo()
    
#     img_data = open(ImgFileName, 'rb').read()
    
#     subject = 'Intruder Detected'
#     body = 'This person logged into your account'

#     msg = MIMEMultipart()
#     msg['Subject'] = subject
#     msg['From'] = EMAIL_ADDRESS
#     msg['To'] = EMAIL_ADDRESS

#     text = MIMEText('This person has tried to login into your account')
#     msg.attach(text)
#     image = MIMEImage(img_data, name = os.path.basename(ImgFileName))
#     msg.attach(image)
    
#     smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
#     smtp.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())
#     smtp.quit()
# import cv2 as cv
# import os
# import numpy as np

# capture = cv.VideoCapture(0)

# while True:
#     isTrue, frame = capture.read()
    
#     haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#     DIR = r'C:\Users\Srinivasan\Desktop\Mini-Project\Images'

#     people = []
#     for x in os.listdir(DIR):
#         people.append(x)

#     np_load_old = np.load

#     np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#     features = np.load('features.npy')
#     labels = np.load('labels.npy')

#     np.load = np_load_old
    
#     face_recognizer = cv.face.LBPHFaceRecognizer_create()
#     face_recognizer.read('face_trained.yml')

    
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
#     # Detect the person
#     faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 7)
#     for (x, y, a, b) in faces_rect:
#         faces_roi = gray[y:y+b, x:x+a]

#         label, confidence = face_recognizer.predict(faces_roi)
#         print(f'label = {people[label]} with a confidence of {confidence}')
    
    
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# capture.release()
# capture.destroyAllWindows()    

# cv.waitKey(0)


# cam = cv.VideoCapture(os.path.abspath('video.avi')) 
DIR = os.path.abspath('videos')
path = os.path.join(DIR, 'video.avi')
cam = cv.VideoCapture(path) 
DIR1 = os.path.abspath('Mini-Images')
name = 'Nithish'

path = os.path.join(DIR1, name)

os.mkdir(path)
currentframe = 0
print(path)
path = os.path.join(path, name)
while(currentframe <= 100): 
    ret,frame = cam.read()
    if ret: 
        cv.imwrite(path+str(currentframe)+'.jpg', frame)
        currentframe += 1
    else: 
        break

cam.release() 
cv.destroyAllWindows() 



# cam = cv.VideoCapture("Mini-Project/videos/video.avi") 
# DIR = os.path.abspath('Mini-Images')
# path = os.path.join(DIR, name)
# # mode = 0o666
# os.mkdir(path)
# new_path = os.path.abspath(name)
# # try: 
# #     # creating a folder named data 
# #     if not os.path.exists(name): 
# #         os.makedirs(name) 

# # # if not created then raise error 
# # except OSError: 
# #     print ('Error: Creating directory of data') 

# # frame 
# currentframe = 0

# while(currentframe <= 100): 
    
#     # reading from frame 
#     ret,frame = cam.read()
    
#     if ret: 
#         # if video is still left continue creating images 
#         Name = path+'/' + str(currentframe) + '.jpg'
#         # print ('Creating...' + name) 

#         # writing the extracted images 
#         cv.imwrite(new_path+'/'+str(currentframe)+'.jpg', frame)

#         # increasing counter so that it will 
#         # show how many frames are created 
#         currentframe += 1
#     else: 
#         break

# # Release all space and windows once done 
# cam.release() 
# cv.destroyAllWindows()  
