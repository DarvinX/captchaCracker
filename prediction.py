from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import sys
import string

filename = './dataset/validation/2g783.png'
output_name = ''
sym = sorted(string.ascii_lowercase + string.digits)
symbols = dict((i, char) for i, char in enumerate(sym))

if len(sys.argv) <= 1:
    print("Give some filename or path as argument")
    exit()

model = load_model('captchaCracker_model.h5')
input = img_to_array(load_img(sys.argv[1], color_mode='grayscale'))
input = np.reshape(input, (-1,50,200,1))
np.shape(input)
prediction = model.predict(input)
prediction = np.reshape(prediction, (5,36))
prediction = np.argmax(prediction, axis=1)

for i in prediction:
    output_name += symbols[i]

print(output_name)
