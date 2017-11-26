# Kiddo
Playing with Google Quick draw dataset with Keras.
Dataset can be found [here](https://quickdraw.withgoogle.com/#)
Kiddo is a convolution neural network model designed to identify hand drwan images. The speciality of these images is that these are drawn with a very small intervall of time.
The images in a single class are not drawn by all different person in small time, so they are least possible to have any sort of similarities.
10 images from the 10 datasets we used are shown below:
![alt text](https://github.com/Subarno/Kiddo/blob/master/img/data.png "Dataset Screenshot")

The experment consisted of various types of subject of images that are present in Google's Quickdraw dataset.
The convolution network used in experiment is a bit modification of LeNet. Runtime of LeNet is also very fast and the prediction can be made with minimun computations.
The accuracy results shown were as follow:
![alt text](https://github.com/Subarno/Kiddo/blob/master/img/lenet_acc.png "Accuracy")

The experiment can be carried out further increasing the number of classes of images. Enriching the classes will give out better results.
