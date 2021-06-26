from signal_handler import Handler
import numpy as np
import fft_filter
import dlib
import cv2 as cv
import hr_calculator
from LE import LE
from sklearn.manifold import SpectralEmbedding
import time

freqs_min = 0.8
freqs_max = 4

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


def get_hr(ROI, fps):
    signal_handler = Handler(ROI)
    blue, green, red = signal_handler.get_channel_signal()
    matrix = np.array([blue, green, red])
    component = signal_handler.ICA(matrix, 3)


    # matrix = np.vstack((blue,green,red))
    # matrix = matrix.T
    # #component = signal_handler.ICA(matrix, 3)
    # #component = LE(matrix, dim = 1, k = 3, graph = 'k-nearest', weights = 'heat kernel', sigma = 5, laplacian = 'symmetrized')
    # embedding = SpectralEmbedding(n_components=1)
    # Y = embedding.fit_transform(matrix)
    # #Y = component.transform()

    fft, freqs = fft_filter.fft_filter(component[0], freqs_min, freqs_max, fps)
    heartrate_1 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    fft, freqs = fft_filter.fft_filter(component[1], freqs_min, freqs_max, fps)
    heartrate_2 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    fft, freqs = fft_filter.fft_filter(component[2], freqs_min, freqs_max, fps)
    heartrate_3 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)

    return (heartrate_1 + heartrate_2 + heartrate_3) / 3
    

if __name__ == '__main__':
    # video_path = 'videos/rohin_face.mov'
    ROI = []
    HR = []
    heartrate = 0
    camera_code = 0
    #capture = cv.VideoCapture(camera_code)
    capture = cv.VideoCapture('pei_reposo.mp4')
    fps = capture.get(cv.CAP_PROP_FPS)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    tinicial = time.time_ns()
    heartrate2=0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            continue
    
        grayf = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = detector(grayf, 0)
        #for face in dects:
        if len(face) >0:
            shape = predictor(grayf, face[0])
            shape = shape_to_np(shape)

            for (a, b) in shape:
                 cv.circle(frame, (a, b), 1, (0, 0, 255), -1) #draw facial landmarks

            left = face[0].left()
            right = face[0].right()
            top = face[0].top()
            bottom = face[0].bottom()
            h = bottom - top
            w = right - left

            #roi = frame[top + h // 10 * 2:top + h // 10 * 7, left + w // 9 * 2:left + w // 9 * 8]

            cv.rectangle(frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
                        (shape[12][0],shape[33][1]), (0,255,0), 0)
            cv.rectangle(frame, (shape[4][0], shape[29][1]), 
                    (shape[48][0],shape[33][1]), (0,255,0), 0)
            
            ROI1 = frame[shape[29][1]:shape[33][1], #right cheek
                    shape[54][0]:shape[12][0]]
                    
            ROI2 =  frame[shape[29][1]:shape[33][1], #left cheek
                    shape[4][0]:shape[48][0]]

            #cv.rectangle(frame, (left + w // 9 * 2, top + h // 10 * 3), (left + w // 9 * 8, top + h // 10 * 7), color=(0, 0, 255))
            #cv.rectangle(frame, (left, top), (left + w, top + h), color=(0, 0, 255))
            ROI.append(ROI1)

            ROI.append(ROI2)

            if len(ROI) == 50:
                heartrate = get_hr(ROI, fps)
                HR.append(heartrate)
                for i in range(50):
                    ROI.pop(0)

            cv.imshow('frame', frame)
            cv.putText(frame, '{:.1f}bps'.format(heartrate2), (50, 300), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            tfinal = time.time_ns()
            
            if(tfinal-tinicial > 5000000000):
                tiempo_transcurrido = tfinal - tinicial
                tinicial = time.time_ns()

                if(max(HR-np.mean(HR))<30):
                    cv.putText(frame, '{:.1f}bps'.format(heartrate), (50, 300), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    cv.imshow('frame', frame)
                    heartrate2=heartrate
                    for c in len(HR):
                        HR.pop(0)
            

        else:
            cv.putText(frame, "No face detected",
                       (200,200), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
