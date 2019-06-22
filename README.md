# captchaCracker
This is a keras model(with tensorflow) to decode the captchas.
The dataset is collected from https://www.kaggle.com/fournierp/captcha-version-2-images. For better usability I've splited the dataset into two folders, train and validation. There is no seperate test set, though validation set can be uses as test set.

# prediction
I've uploaded the trained moded. you can use it by calling the prediction.py file, it takes the image name as an argument.
> python predicton.py ./dataset/validation/7nnnx.png

# Train
To train it on your own dataset just put the pictures in the traing directory, use the filename to label the pictures.
