from flask import Flask, request
from flask_cors import CORS
import keras
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

model_categories = ["Galaxy Note 10", "Galaxy Note 20", "Galaxy S10", "Galaxy S20", "Galaxy S21", "Galaxy Z Flip3", "Galaxy Z Fold3", "iPhone 7", "iPhone 11", "iPhone 12", "iPhone 13", "iPhone SE", "iPhone X", "iPhone XR"]
             
caregory_categories = ["smartphone", "smartwatch", "earphone"]
model_classes = len(model_categories)
category_classes = len(caregory_categories)
def m_getName(label):
    num = 0
    for i in range (model_classes):
        if label[i] > label[num]:
            num = i
    
    return model_categories[num]

def m_getPercent(label):
    num = 0
    for i in range (model_classes):
        if label[i] > label[num]:
            num = i
    
    return label[num]

def c_getName(label):
    num = 0
    for i in range (category_classes):
        if label[i] > label[num]:
            num = i
    
    return caregory_categories[num]

def c_getPercent(label):
    num = 0
    for i in range (category_classes):
        if label[i] > label[num]:
            num = i
    
    return label[num]

def ajaxImage(imageSize):
    data = []
    content = request.files['files']
    img = np.array(Image.open(content).convert("RGB"))
    image = cv2.resize(img, imageSize)/255.0
    data.append(np.asarray(image))
    data = np.array(data)
    return data
    
m_model = keras.models.load_model('SmartPhone_classification_67.h5')
c_model = keras.models.load_model('Category_classification_67.h5')

@app.route('/predict/smartphone', methods=['POST'])
def model_class():
    image = ajaxImage((128,128))
    ### code 작성 ###
    prediction = m_model.predict(image)
    return {'answer': m_getName(prediction[0]), 'percent': int(m_getPercent(prediction[0])*100)}
    
@app.route('/predict', methods=['POST'])
def category_class():
    image = ajaxImage((128,128))
    ### code 작성 ###
    
    
    prediction = c_model.predict(image)
    return {'answer': c_getName(prediction[0]), 'percent': int(c_getPercent(prediction[0])*100)}
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)
