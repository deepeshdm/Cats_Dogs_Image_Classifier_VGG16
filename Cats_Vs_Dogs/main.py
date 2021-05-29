
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import Callback
from keras.models import load_model
from Dataset_Preparation import get_data
from keras.layers import Dense,Flatten

print("All Dependencies Installed")

#-----------------------------------------------------------

# Gets us 8000 images (224,224,3) & their labels as 4D arrays.
# 4000 of Cats & other 4000 of Dogs.
x,y = get_data(cnn_input_shape=(224,224))
x = np.array(x)
y = np.array(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print("Data Pre-preprocessing Completed !")

#-----------------------------------------------------------

base_model = VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3),pooling="max")

# It'll prevent learning of Conv Layers & we'll use pre-trained weights.
for layer in base_model.layers:
    layer.trainable=False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])

# Custom Keras callback to stop training when certain accuracy is achieved.
class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True

print("Model Creation Complete")

#-----------------------------------------------------------

model.fit(x_train,y_train,epochs=100,batch_size=10,
          callbacks=[MyThresholdCallback(0.97)],validation_data=(x_test,y_test))

# SAVING THE MODEL
model.save("Cats_Vs_Dogs_VGG16")

print("Model Training Completed !")

#-----------------------------------------------------------

# Classifying a single image.
model = load_model("Cats_Vs_Dogs_VGG16")

img = cv2.imread("cat.4062")
resized_img = cv2.resize(img,(224,224))
img_array = np.asarray(resized_img)

# the predict() accepts a 4D array.
img = img_array.reshape((-1,224,224,3))

predicted = model.predict(img)
print(predicted)

i = np.argmax(predicted)
if i == 0:
  print("It's a Dog !")
else:
  print("It's a Cat !")



