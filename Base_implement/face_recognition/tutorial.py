from sklearn.decomposition import RandomizedPCA
import numpy as np
import glob
import cv2
import math
import os.path
import string

#function to get ID from filename
def ID_from_filename(filename):
    part = string.split(filename, '/')
    return part[1].replace("p_", "")
 
#function to convert image to right format
def prepare_image(filename):
    img_color = cv2.imread(filename)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    return img_gray.flat

IMG_RES = 92 * 112 # img resolution
NUM_EIGENFACES = 10 # images per train person
NUM_TRAINIMAGES = 55 # total images in training set

#loading training set from folder train_faces
folders = glob.glob('train_faces/*')
 
cap = cv2.VideoCapture(0)
count = 0
pos = 'test_faces/'
pic_format = '.jpg'

faceCascade = cv2.CascadeClassifier('haar_files/haarcascade_frontalface_default.xml')

# Take a snapshot
def Snap(img):
    global count
    NAME = pos + str(count) + pic_format
    cv2.imwrite(NAME, img)

#'''
# Create an array with flattened images X
# and an array with ID of the people on each image y
X = np.zeros([NUM_TRAINIMAGES, IMG_RES], dtype='int8')
names = []

# Populate training array with flattened imags from subfolders of train_faces and names
c = 0
for x, folder in enumerate(folders):
    train_faces = glob.glob(folder + '/*')
    for i, face in enumerate(train_faces):
        X[c,:] = prepare_image(face)
        names.append(ID_from_filename(face))
        c = c + 1

# perform principal component analysis on the images
pca = RandomizedPCA(n_components=NUM_EIGENFACES, whiten=True).fit(X)
X_pca = pca.transform(X)

#'''
while (True):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
                                        gray,
                                        scaleFactor=1.3,
                                        minNeighbors=5
)
    for(x,y,w,h) in faces:
        if len(faces) == 1:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            gray = gray[y:y+h, x:x+w]
            s = cv2.resize(gray, (92,112))
            Snap(s)
        elif len(faces) > 1 : 
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

        # load test faces (usually one), located in folder test_faces
        test_faces = glob.glob('test_faces/*')
        # Create an array with flattened images X
        Z = np.zeros([len(test_faces), IMG_RES], dtype='int8')
 
        # Populate test array with flattened imags from subfolders of train_faces 
        for i, face in enumerate(test_faces):
            Z[i,:] = prepare_image(face)
 
        # run through test images (usually one)
        for j, ref_pca in enumerate(pca.transform(Z)):
            distances = []
            # Calculate euclidian distance from test image to each of the known images and save distances
            for i, test_pca in enumerate(X_pca):
                dist = math.sqrt(sum([diff**2 for diff in (ref_pca - test_pca)]))
                distances.append((dist, names[i]))
 
            found_ID = min(distances)[1]
        if min(distances)[0] < 1.8:
            found_ID = min(distances)[1]

            # Display the resulting frame
            cv2.putText(frame, found_ID, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print "Identified (result: "+ str(found_ID) +" - dist - " + str(min(distances)[0])  + ")"
            if found_ID == 'jssmile':
                print("It's member of Luolab")
            elif found_ID == 'willy':
                print("It's member of Luolab")
    cv2.imshow('frame',frame)
    # Ket event
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("quit")
        break
cap.release()
cv2.destroyAllWindows() 