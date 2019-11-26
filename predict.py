import glob
import cv2
import tensorflow as tf
import numpy as np
from train import train_generator
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
inf_model = tf.keras.models.load_model("Model/best.hdf5")

#Preprocess cho ảnh đầu vào.

# def preprocess_image(img):
#         if (img.shape[0] != 150 or img.shape[1] != 150):
#             img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
#         img = (img/127.5)
#         img = img - 1
#         img = np.expand_dims(img, axis=0)
#         return img

classes = train_generator.class_indices
classes = list(classes.keys())
img_width, img_height = 150, 150
# files = "Predict/anh.jpg"
# img = cv2.imread(files)
# pred = inf_model.predict(preprocess_image(img))
# result = classes[np.argmax(pred)]
# print("Flow is: ", result)

mypath = "Predict/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:
    img = image.load_img(mypath + file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    predict = inf_model.predict_classes(images, batch_size=10)
    result = classes[predict[0]]
    print("Flow is: ", result)

