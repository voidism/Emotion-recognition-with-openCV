import cv2
import glob
import random
import numpy as np
from PIL import Image, ImageFilter 

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
faceDet5 = cv2.CascadeClassifier("haarcascade_profileface.xml")
faceDet6 = cv2.CascadeClassifier("haarcascade_smile.xml")
faceDet7 = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
faceDet8 = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
#faceDet9 = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")



emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions
imgfilename = []
def Sharppp():
    files = glob.glob("pre_faces\\*")
    for i in files:
        img = Image.open(i)
        imgfilted = img.filter(ImageFilter.SHARPEN)
        imgfilted = imgfilted.filter(ImageFilter.SHARPEN)

        imgfilted.save(i)  


def detect_faces():
    files = glob.glob("pre_faces\\*") #Get list of all images with emotion
    
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face5 = faceDet5.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face6 = faceDet6.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face7 = faceDet7.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face8 = faceDet8.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #face9 = faceDet9.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        

        
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face7) == 1:
            facefeatures = face7
        elif len(face6) == 1:
            facefeatures = face6
        elif len(face) == 1:
            facefeatures = face
        elif len(face2) == 1:
            facefeatures = face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        elif len(face5) == 1:
            facefeatures = face5
        elif len(face8) == 1:
            facefeatures = face8
        #elif len(face6) == 1:
            #facefeatures = face6
        #elif len(face7) == 1:
            #facefeatures = face7
        else:
            facefeatures = ""
        
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print "face found in file: %s" %f
            imgfilename.append("%s" %f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("faces\\%s.jpg" %(filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number


detect_faces() #Call functiona




    

emotions1 = ["anger", "happy"] #Emotion list
fishface = cv2.createFisherFaceRecognizer() #Initialize fisher face classifier

data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    faces = glob.glob("faces\\*")
    random.shuffle(files)
    training = files
    prediction = faces
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    
    #prediction_labels = []
    for emotion in emotions1:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions1.index(emotion))
    
    for item in prediction: #repeat above process for prediction set
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prediction_data.append(gray)
        #prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data#, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data = make_sets()
    
    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))#Machine Learning Processing

    print "predicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        print ("we detect picture " + imgfilename[cnt] + " as " + emotions1[pred])
        #print image
        cnt+=1
        
    return 
#Now run it
#metascore = []
run_recognizer()
#print "got", correct, "percent correct!"
#metascore.append(correct)

#print "\n\nend score:", np.mean(metascore), "percent correct!"
#print prediction_data
