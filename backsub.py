import cv2
import numpy as np 

cap	= cv2.VideoCapture(1)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
	ret,frame = cap.read()
	fgmask = fgbg.apply(frame)
	kernel = np.ones((3,3),np.uint8)
	kernel2 = np.ones((2,2),np.uint8)
	erosion = cv2.erode(fgmask,kernel,iterations = 2)
	#erosion = cv2.dilate(erosion,kernel2,iterations = 1)
	cv2.imshow('original',frame)
	cv2.imshow('fg',erosion)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()