from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D
import keras
from keras.models import Model
from keras.layers import *

trian_dir = 'E:/projects/py/1'

test_dir = 'E:/projects/py/1t'

width = 224

hight = 224

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    trian_dir,
    target_size=(width, hight),
    batch_size=2,
    class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(width, hight),
    batch_size=2,
    class_mode='categorical')

backbone = keras.applications.ResNet50(weights=None)
backbone.load_weights("./resnet50_weights_tf_dim_ordering_tf_kernels.h5")
x = backbone.get_layer("avg_pool").output
prediction = Dense(2,activation="softmax")(x)
model = Model(input=backbone.input,output=prediction)


model.compile(loss='squared_hinge',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=5,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=5)
