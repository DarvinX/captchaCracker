# %%
import numpy as np
import string

symbols = string.ascii_lowercase + string.digits

symbols = sorted(symbols)
symbols_len = len(symbols)

print(symbols)
print(symbols_len)
# %%

from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import glob
import os

dataset_path = "./dataset/train/"

output_names = []
x_ = []

for file in glob.glob(dataset_path + "*.png"):
    filename = os.path.basename(file)
    filename = filename.split('.')[0]
    output_names.append(filename)
    x_.append(img_to_array(load_img(file, color_mode='grayscale'))/255.)

np.shape(x_)
x_ = np.reshape(x_, (-1,50,200,1))
np.shape(x_)
# %%
dim = (5, len(output_names), symbols_len)
y_ = np.zeros(dim)

for i, name in enumerate(output_names):
    for j in range(5):
        y_[j][i][symbols.index(name[j])] = 1


# %%

# %% load the dataset
plt.figure(figsize=(10,2))

for _ in range(10):
    plt.subplot(2,5, _ + 1)
    plt.axis('off')
    plt.title(output_names[_])
    plt.imshow(np.reshape(x_[_], (50, 200)), cmap='gray')

# %%
from keras.models import Model
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Input

input = Input(shape=(50,200,1))

x = Conv2D(16, (3,3), activation='relu', padding='same')(input)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

flat = Flatten()(x)
outputs = []
for _ in range(5):
    fc1 = Dense(128, activation='relu')(flat)
    drop = Dropout(0.2)(fc1)
    fc2 = Dense(64, activation='relu')(drop)
    drop = Dropout(0.2)(fc2)
    out = Dense(36, activation='softmax')(drop)
    outputs.append(out)
model = Model(input, outputs)
model.summary()
# %%
from keras.utils import plot_model
plot_model(model, to_file='model.png')
# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_,[y_[0],y_[1],y_[2],y_[3],y_[4]],
          epochs=60,
          batch_size = 64,
          validation_split=0.2,
          )
# %%
prediction =
