#taking input from ./dataset/validation

from os import listdir, system
from os.path import isfile, join

import prediction

validation_dir = "./dataset/validation/"
pic_extension = 'png'

success = 0
failure = 0

#get all the files
validation_captcha = [f for f in listdir(validation_dir) if isfile(join(validation_dir, f))]
validation_captcha_fullpath = [validation_dir+f for f in validation_captcha]
total_captcha = len(validation_captcha)

#print(validation_captcha)

out = prediction.predict(validation_captcha_fullpath)

for i, captcha in enumerate(validation_captcha):
    name = captcha.split('.')[0]
    print(name+" => "+out[i])
    if name == out[i]:
        success += 1
    else:
        failure += 1

print('success: ',success)
print('failure: ',failure)
print('success rate: ',(success/total_captcha)*100)