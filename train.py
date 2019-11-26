import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,SeparableConv2D,BatchNormalization, Activation, Dense
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# tải dataset vê
zip_file = tf.keras.utils.get_file(origin="http://download.tensorflow.org/example_images/flower_photos.tgz",
                                   fname="flower_photos.tgz", extract=True)
# get đường dẫn của file đã tải về
base_dir, _ = os.path.splitext(zip_file)
# cấu hình đường dẫn của thư mục data
print(base_dir)
# Khai báo số class
num_class = 5
image_size = 150  # tất cả ảnh được resize về 160x160
# số sample được đưa vào xử lý
batch_size = 32
IMG_SHAPE = (image_size, image_size, 3)


# Load base model từ Mobinet đã được pre-train
# Khi load pretrained model, tham số include_top = False để không bao gồm các lớp Fully connected ở cuối, weights = 'imagenet' để load pretrained model train trên tập imagenet
base_model = keras.applications.MobileNet(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
base_model.trainable = False

#thêm các layer ở lớp cao nhất
model = tf.keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(num_class, activation='softmax')
])
# model.summary()
"""
# Cho phép train lại base model 
base_model.trainable = True

# Hiển thi số lớp trong base m
print("Number of layers in the base model: ", len(base_model.layers))

# Set điều kiện lấy layer  
fine_tune_at = 100

# Cho phóp train từ layer thứ 100 trở 
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
"""
# Sử dụng ImageDataGenerator từ keras. Chia tỉ lệ train-val là 75-25
# Sinh thêm nhiều dạng dữ liệu nữa sử dụng ImageDataGenerator của tensorflow
train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                 validation_split=0.25,
                                 rescale=1. / 255
                                 )
# nhận diện địa chỉ (folder chứa ảnh), tải toàn bộ ảnh trong folder đó, thu nhỏ kích thước còn 150x150,
# chuyển kết quả thành 2D Tensor,chuyển thang đo của mỗi kênh từ 0:255 thành 0:1; dán label (classes - Neu khai bao); tạo batch với kích thước 64
train_generator=train_datagen.flow_from_directory(base_dir,
                                                 target_size=(image_size,image_size),
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 subset='training')


validation_generator = train_datagen.flow_from_directory(
                                                base_dir, # same directory as training data
                                                target_size=(image_size,image_size),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                subset='validation') # set as validation data

#Training

epochs = 10
learning_rate = 0.0005
decay_rate = learning_rate / epochs
opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

# Set callback để lưu model và tensorboard.
filepath="Model/best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only = False, save_best_only=True, mode='min')
logdir="Log/mobilenet"
tfboard = TensorBoard(log_dir=logdir)
callbacks_list = [checkpoint, tfboard]

# train_generator.n: Lấy tổng data train
# validation_generator.samples: Lấy tổng data validate
step_size_train = train_generator.n // batch_size
step_size_val = validation_generator.samples // batch_size
# history = model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    validation_data = validation_generator,
#                    validation_steps =step_size_val,
#                    callbacks = callbacks_list,
#                    epochs=epochs)
# acc = history.history['acc']
# val_acc = history.history['val_acc']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,max(plt.ylim())])
# plt.title('Training and Validation Loss')
# plt.show()
