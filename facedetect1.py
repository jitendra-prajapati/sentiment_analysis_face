from random import randint
import cv2
import sys
import os

CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_faces(image):
#	image=cv2.imread(image_path)
#	print(image)
	image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.10,minNeighbors=5,minSize=(25,25),flags=0)
	os.chdir("./Extracted")
	for x,y,w,h in faces:
		sub_img=image[y-10:y+h+10,x-10:x+w+10]
	#	print(sub_img.shape)
	#	print(len(sub_img))
		#os.chdir("./Extracted")
		shape = sub_img.shape
		if shape[0]!= 0 and shape[1] != 0 and shape[2] != 0:
			sub_img = cv2.resize(sub_img,(48,48))
			cv2.imwrite(str(randint(0,10000))+".jpg",sub_img)
#			os.chdir("../")
			cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)
	os.chdir("../")
	#cv2.imshow("Faces Found",image)
	# if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
	# 	cv2.destroyAllWindows()


if not "Extracted" in os.listdir("."):
	os.mkdir("Extracted")

