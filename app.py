from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
from flask_sqlalchemy import SQLAlchemy
import os
import cv2 as cv
import numpy as np
import smtplib
import os
import time
import imghdr
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
# dev or prod can be translated to user requirements
ENV = 'dev'
if ENV == 'dev':
    app.config['SQLALCHEMY_DATABASE_URI'] = ''
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = ''

app.config['SQLALCHEMY_TRACK_MODTIFICATION'] = False
db = SQLAlchemy(app)
name = ""
emailID = ""
UserName = ""
video_camera = None
global_frame = None
@app.route('/')
class User(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(200), unique = True)
    email = db.Column(db.String(200), unique = True)
    password = db.Column(db.String(200), unique = True)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password
            
def redirect():
    return render_template('redirect.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global name
    if request.method == 'POST':
        username    = request.values.get('Name')
        email       = request.values.get('e-mail')
        password = request.values.get('password')
        name = username
        if db.session.query(User).filter(User.password == password).count() == 0:
            data = User(username, email, password)
            db.session.add(data)
            db.session.commit()
            return render_template('index.html', name=username)
        else:
            return render_template('register.html', message = "You are already registered to my website please fill all the fields with unique values")     
    else:    
        return render_template('register.html')


@app.route('/simple')
def simple():
    frameCapture()
    return render_template('simple.html')

@app.route('/blank')
def blank():
    create_train()
    return render_template('blank.html')

@app.route('/checker')
def checker():
    status = use_trained()
    if status:
        return render_template('checker.html')
    else:
        mailsender()
        return render_template('failed.html')
@app.route('/verifier')
def verifier():
    return render_template('verifier.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    global emailID
    global UserName
    if request.method == 'POST':
        UserName = request.values.get('username')
        Password = request.values.get('Password')
        user_object = User.query.filter_by(username = UserName).first()
        if user_object == None:
            message = "Username or password is incorrect"
            return render_template('login.html', message = message)
        elif Password != user_object.password:
            message = "Incorrect password"
            return render_template('login.html', message = message)    
        else:
            emailID = user_object.email
            return render_template('verifier.html', name = UserName)
    else:
        return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        # FrameCapture()
        return jsonify(result="stopped")

def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream())
                    # mimetype='multipart/x-mixed-replace; boundary=frame')

def frameCapture():
    global name
    DIR = os.path.abspath('videos')
    path = os.path.join(DIR, 'video.avi')
    cam = cv.VideoCapture(path) 
    DIR1 = os.path.abspath('Mini-Images')
    
    path = os.path.join(DIR1, name)

    os.mkdir(path)
    currentframe = 0
    path = os.path.join(path, name)
    while(currentframe <= 300): 
        ret,frame = cam.read()
        if ret: 
            cv.imwrite(path+str(currentframe)+'.jpg', frame)
            currentframe += 1
        else: 
            break

    cam.release() 
    cv.destroyAllWindows() 

    
    
def create_train():
    # DIR = r'Mini-Project\Mini-Images'
    DIR = os.path.abspath('Mini-Images')
    people = []
    for x in os.listdir(DIR):
        people.append(x)
    
    # print(people)
    haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    features = []
    labels = []
    # path = os.path.join(DIR, name)
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 4)

            for (x, y, a, b) in faces_rect:
                faces_roi = gray[y:y+b, x:x+a]
                features.append(faces_roi)
                labels.append(label)

    
    print('Training done-------------------------')

    # converting to numpy arrays
    feature = np.array(features)
    labels = np.array(labels)

    # print(dir (cv.face)) 
    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    print(f'Length of features = {len(features)}')
    print(f'Length of labels = {len(labels)}')                

    # Train the Recognizer on features list and the labels list
    face_recognizer.train(features, labels)

    face_recognizer.save('face_trained.yml')
    np.save('features.npy',features)
    np.save('labels.npy', labels)

        
def use_trained():
    global name
    DIR = os.path.abspath('videos')
    path = os.path.join(DIR, 'video.avi')
    capture = cv.VideoCapture(path) 
    d = []
    count = 0
    recognised_name = ""
    wrong = True
    while count <= 20:
        isTrue, frame = capture.read()

        haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

        # DIR = r'Mini-Project\Images'
        DIRECT = os.path.abspath('Mini-Images')
    
        people = []
        for x in os.listdir(DIRECT):
            people.append(x)

        np_load_old = np.load

        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        features = np.load('features.npy')
        labels = np.load('labels.npy')

        np.load = np_load_old

        face_recognizer = cv.face.LBPHFaceRecognizer_create()
        face_recognizer.read('face_trained.yml')

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect the person
        faces_rect = haar_cascade.detectMultiScale(gray, 1.2, 6)
        
        for (x, y, a, b) in faces_rect:
            faces_roi = gray[y:y+b, x:x+a]

            label, confidence = face_recognizer.predict(faces_roi)
            recognised_name = people[label]    
            print(f'label = {people[label]} with a confidence of {confidence}')
            d.append(confidence)
        count += 1
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    counter = 0
    for c in d:
        if c>=70:
            counter += 1
    if UserName != recognised_name:
        path = os.path.abspath('Mini-project')
        print('The user image doesnt match')
        print(path)
        i = 0
        while(capture.isOpened() and i<=1):
            ret, frame = capture.read()
            if ret == False:
                break
            cv.imwrite('Intruder.jpg',frame)
            i+=1
        return False    
    else:
        return True
    capture.release()
    
    # cv.waitKey(0)
    # cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    # cv.rectangle(img, (x, y), ((x+a), (y+b)), (255, 0, 0), 2)

def mailsender():
    EMAIL_ADDRESS = ""  # Custom email address  
    EMAIL_PASSWORD = "" # Custom password
    ImgFileName = 'Intruder.jpg'
    

    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()

        img_data = open(ImgFileName, 'rb').read()

        subject = 'Intruder Detected'
        body = 'This person logged into your account'

        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = emailID

        text = MIMEText('This person has tried to login into your account')
        msg.attach(text)
        image = MIMEImage(img_data, name = os.path.basename(ImgFileName))
        msg.attach(image)

        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.sendmail(EMAIL_ADDRESS, emailID, msg.as_string())
        smtp.quit()

if __name__ == '__main__':
    app.run(host='127.0.0.1', threaded=True)
