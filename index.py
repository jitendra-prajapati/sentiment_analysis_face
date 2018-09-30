import cv2
from facedetect1 import detect_faces
from predict import predict
import os
import sys
import dlib
import time
import tensorflow as tf
from model import build_model
from parameters import DATASET, TRAINING, NETWORK, VIDEO_PREDICTOR
from tflearn import DNN

def load_model():
	model = None
	with tf.Graph().as_default():
		print( "loading pretrained model...")
		network = build_model()
		model = DNN(network)
		if os.path.isfile(TRAINING.save_model_path):
			model.load(TRAINING.save_model_path)
		else:
			print( "Error: file '{}' not found".format(TRAINING.save_model_path))
	return model
	
def read_resize(image_path):
	img = cv2.imread(image_path)
	cv2.resize(img,(640,360))
	return img

file_list = []
def load_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename),0)
		if img is not None:
			file_list.append(filename)
			images.append(img)
	return images

def predict_class(image_path):
	img = read_resize(image_path)
	detect_faces(img)

	images = load_images_from_folder('./Extracted/')
	model = load_model()
	for k,i in enumerate(images):
        #predict(i)i
		print("")
		print("filename: ",file_list[k])
		print("shape_of_img ",i.shape)
		start_time = time.time()
		shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
		emotion, confidence = predict(i, model, shape_predictor)
		total_time = time.time() - start_time
		print( "Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence*100))
		print( "time: {0:.1f} sec".format(total_time))

predict_class("sample2.png")
