import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Input
import matplotlib.pyplot as plt
import glob
import numpy as np
import string

train_data_path = './dataset/train/'
val_data_path = './dataset/validation/'
x_train = []
x_val = []
label_name_len = 5
img_height = 50
img_width = 200
batch_size = 128
epochs = 100

sym = sorted(string.ascii_lowercase + string.digits)
symbols = dict((i, char) for i, char in enumerate(sym))
reverse_symbols = dict((char, i) for i, char in symbols.items())

symbols_len = len(symbols)

file_count_train = len(glob.glob(train_data_path + '*.png'))
file_count_val = len(glob.glob(val_data_path + '*.png'))

y_train_dim = (label_name_len, file_count_train, symbols_len)
y_val_dim = (label_name_len, file_count_val, symbols_len)

y_train = np.zeros(y_train_dim)
y_val = np.zeros(y_val_dim)
np.shape(y_train)

for num, file in enumerate(glob.glob(train_data_path + '*.png')):
    filename = os.path.basename(file)
    filename = filename.split('.')[0] #get the label data
    x_train.append(img_to_array(load_img(file, color_mode='grayscale'))/255.)

    for i, char in enumerate(filename):
        y_train[i][num][reverse_symbols[char]] = 1

for num, file in enumerate(glob.glob(val_data_path + '*.png')):
    filename = os.path.basename(file)
    filename = filename.split('.')[0]
    x_val.append(img_to_array(load_img(file, color_mode='grayscale'))/255.)

    for i, char in enumerate(filename):
        y_val[i][num][reverse_symbols[char]] = 1

# %% visualis
'''
plt.figure(figsize=(10,2))

for _ in range(10):
    plt.subplot(2,5, _ + 1)
    plt.axis('off')
    plt.imshow(np.reshape(x_val[_], (50, 200)), cmap='gray')
'''
# building the model
input = Input(shape=(img_height,img_width,1))



x = Conv2D(16, (3,3), activation='relu', padding='same')(input)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

#x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
#x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2,2))(x)

# x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
# x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2,2))(x)

flat = Flatten()(x)
outputs = []
for _ in range(5):
    fc1 = Dense(64, activation='relu')(flat)
    drop = Dropout(0.2)(fc1)
    fc2 = Dense(64, activation='relu')(drop)
    drop = Dropout(0.2)(fc2)
    fc3 = Dense(64, activation='relu')(drop)
    drop = Dropout(0.2)(fc3)
    out = Dense(symbols_len, activation='softmax')(drop)
    outputs.append(out)

model = Model(input, outputs)

# reshaping the input array for training
x_train_new = np.reshape(x_train, (-1,img_height,img_width,1))
x_val_new = np.reshape(x_val, (-1,img_height,img_width,1))

checkpoint = ModelCheckpoint('captchaCracker_model.h5', monitor='val_loss', save_best_only=True,verbose=0)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_new,[y_train[0],y_train[1],y_train[2],y_train[3],y_train[4]],
          epochs=epochs,
          batch_size = batch_size,
          validation_split= 0.1,
          callbacks=[checkpoint]
          )

# to evaluate the model just uncomment the below line
# model.evaluate(x_val_new,[y_val[0],y_val[1],y_val[2],y_val[3],y_val[4]])
model.summary()
