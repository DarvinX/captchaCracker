import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import sys
import string

sym = sorted(string.ascii_lowercase + string.digits)
symbols = dict((i, char) for i, char in enumerate(sym))

#API
def predict(imgs):
    model = load_model('captchaCracker_model.h5')
    output = []
    for img in imgs:
        output_name = ''

        input = img_to_array(load_img(img, color_mode='grayscale'))
        input = np.reshape(input, (-1,50,200,1))
        np.shape(input)
        prediction = model.predict(input)
        prediction = np.reshape(prediction, (5,36))
        prediction = np.argmax(prediction, axis=1)

        for i in prediction:
            output_name += symbols[i]

        output.append(output_name)
        #print(output)
    return output
        
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Give some filename or path as argument")
        exit()
    if len(sys.argv) > 2:
        print("multiple input files")

    predict(sys.argv[1:])
