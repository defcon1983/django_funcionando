from django.shortcuts import render

from django.http import JsonResponse
from rest_framework.views import APIView
from django.apps import AppConfig

import keras
import tensorflow
import numpy as np
import cv2
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from django.conf import settings
import os


path = os.path.join(settings.MODELS)
path_to_model = path + "Emotion_little_vgg.h5"
path_to_cascade = path + "haarcascade_frontalface_defaults.xml"
face_detector = cv2.CascadeClassifier(str(path_to_cascade))
classifier = load_model(path_to_model)
emotion_labels = ("Angry", "Happy", "Neutral", "Sad", "Surprise")
print(emotion_labels)
print(path)


class emotion_detect(APIView):
    def post(self, request, img_name02):
        if request.method == 'POST':
            path = os.path.join(settings.IMG_PATH)
            img_name = path + img_name02
            image = cv2.imread(img_name)
            emotions = []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(image, 1.3, 5)

            for (x, y, w, h) in faces:

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi = image[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi]) != 0:

                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    predict = classifier.predict(roi)[0]
                    predict = predict.argmax()
                    emotion = emotion_labels[predict]
                    emotions.append(emotion)

                else:
                    continue

            response = {'emocion-detectada': emotions}

            return JsonResponse(response)
