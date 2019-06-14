from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D


trian_dir = 'D:/competition/train_data'

test_dir = 'D:/competition/test_data'

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
    batch_size=32,
    class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(width, hight),
    batch_size=32,
    class_mode='binary')

model = Sequential()

model.add(Conv2D(64, (3, 3),
        activation='relu',
        padding='same',
        name='block1_conv1',
        dim_ordering='tf',
        input_shape=(width,hight,3)))


model.add(GlobalAveragePooling2D(data_format=None))

model.add(Dense(1, input_shape=(width, hight), activation='relu'),)

model.compile(loss='squared_hinge',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=11,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=800)