import numpy as np
from signal_handler import Handler
import numpy as np
import fft_filter
import cv2 as cv
import hr_calculator
import math
from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding
from LE import LE

X, _ = load_digits(return_X_y=True)

freqs_min = 0.8
#freqs_max = 1.8
freqs_max = 4

def my_round(value, N):
    value = np.asarray(value).copy()
    zero_mask = (value == 0)
    value[zero_mask] = 1.0
    sign_mask = (value < 0)
    value[sign_mask] *= -1
    exponent = np.ceil(np.log10(value))
    result = 10**exponent*np.round(value*10**(-exponent), N)
    result[sign_mask] *= -1
    result[zero_mask] = 0.0
    return result

def get_hr(ROI, fps):
    signal_handler = Handler(ROI)
    blue, green, red = signal_handler.get_channel_signal()
    #matrix = np.array([blue, green, red])
    matrix = np.vstack((blue,green,red))
    
    matrix = matrix.T

    #component = signal_handler.ICA(matrix, 3)
    component = LE(matrix, dim = 1, k = 3, graph = 'k-nearest', weights = 'heat kernel', sigma = 5, laplacian = 'symmetrized')

    #embedding = SpectralEmbedding(n_components=2)
    #Y = embedding.fit_transform(matrix)

    Y = component.transform()

    fft, freqs = fft_filter.fft_filter(Y[0], freqs_min, freqs_max, fps)
    heartrate_1 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    fft, freqs = fft_filter.fft_filter(Y[1], freqs_min, freqs_max, fps)
    heartrate_2 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    fft, freqs = fft_filter.fft_filter(Y[2], freqs_min, freqs_max, fps)
    heartrate_3 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    return (heartrate_1 + heartrate_2 + heartrate_3) / 3

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

#cap = cv.VideoCapture(0)
cap = cv.VideoCapture('pei_reposo.mp4')
fps = cap.get(cv.CAP_PROP_FPS)
ROI = []

while 1:
    ret, img = cap.read()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x,y,w,h) in faces:        
        roi_gray = img_gray[y+h // 10*2 : y+h // 10*7, x+w // 9*2 : x+w // 9*8]
        
        """  roi_color = img[y+h//3 : y+h//3*2, x + w//4 : x+w//4*3]
        cv.rectangle(img,(x + w//4, y +h//2),(x +w//4*3, y +h//3*2),(255,0,0),2)
         """
        """ ROI DE LA FRENTE """
        roi_color = img[y : y+h//3, x + w//4 : x+w//4*3] 
        cv.rectangle(img,(x+ w//4, y),(x+w//4*3, y + h//3),(255,0,0),2)  
    
        ROI.append(roi_color)
        if len(ROI) == 300:
            heartrate = get_hr(ROI, fps)
            print(heartrate)
            for i in range(30):
                ROI.pop(0)

        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    #cv2.putText(img, '{:.1f}bps'.format(heartrate), (50, 300), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv.imshow('img',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()