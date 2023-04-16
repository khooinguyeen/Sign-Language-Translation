# LNT model for recognizing Vietnamese sign language

![Python](https://img.shields.io/badge/Language-Python-blue?logo=python)
![Tensorflow](https://img.shields.io/badge/Framework-Tensorflow-important?logo=tensorflow)
![Numpy](https://img.shields.io/badge/Package-Numpy-%23150458?logo=numpy)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-%23white?logo=opencv)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This folder contains the model code for the app "Look & Tell"

* [Look & Tell (Beta) - communication support solution for deaf people with artificial intelligence](https://github.com/khooinguyeen/LookandTell-OfficialApp)

<img src="https://github.com/khooinguyeen/Sign-Language-Translator/blob/main/Sign%20Language%20Translator/Demo/lookandtell.png" width="400">

# 1. About this code
* A deep-learning neural network for hand action recognition, made by Vietnamese students, to translate Vietnamese sign language

# 2. Morality and 
* Kiet, my born deaf neighbor, has not been able to hear his mother's voice. For 10 years from now, the 5 grader kid has been having difficulty expressing the simplest wishes of an elementary school child - hearing the life.

* As can be seen from external statistics about hearing difficulties, there are currently 1.5 billion speech and hearing impaired people in the world (approximately 2.5 million people in Vietnam). However, it entails a huge economic burden for rising countries such as South East Asia countries including Vietnam. According to a 2015 WHO report, the costs of speech and hearing impairment range from $750 to $790 billion. 

* In Vietnam, only a small handful of 20 people have created qualified sign language interpreters. There was no other reasonable communication support product than the one made by Ho Chi Minh City University of Science and Technology. However, the equipment costs nearly 1 million VND during the conduction of a 250 deaf and hard of hearing people survey. Results showed that nearly all of the surveyed people want to support this type of product. 

* **With the desire to communicate with Kiet (my neighbor) as well as all the hearing deprived people in the world, my friend and I settled to create this project.**

# 3. How to use this code
### (1) clone LNT
```
git clone https://github.com/khooinguyeen/Sign-Language-Translator.git
```

### (2) install appropriate dependencies
```
!pip install tensorflow==2.5.0 tensorflow-gpu==2.5.0 opencv-python mediapipe sklearn matplotlib
```

### (3) go to the file [RunModel.ipynb](https://github.com/khooinguyeen/Sign-Language-Translation/blob/main/Sign%20Language%20Translator/RunModel.ipynb) to play with ipython notebooks and run the model
* Relative path:
```
Sign Language Translator\RunModel.ipynb
```

### (4) to collect more data by your own, go to the file [CollectData.ipynb](https://github.com/khooinguyeen/Sign-Language-Translator/blob/main/Sign%20Language%20Translator/CollectData.ipynb)
* Relative path:
```
Sign Language Translator\CollectData.ipynb
```

# 3. Demo
* **Me testing the model :>**
![](https://user-images.githubusercontent.com/91497379/232302443-5b7f3eaa-874e-4c1d-b8fc-25540bc2368d.gif)


* **My cute little neighbour Kiet testing the model :>**
![](https://user-images.githubusercontent.com/91497379/232302456-e1b7a9f2-434b-4849-85b7-20144b6c9797.gif)


* **My friend testing the model**
![](https://user-images.githubusercontent.com/91497379/232302501-008143b3-b291-410c-8053-c9883724eda4.gif)


# 4. Contact
* If you appreciate the humanity of the project, have any problem, want to talk, or even just want to contribute to our project. Please contact us via email: **khoinguyenmai17102005@gmail.com**
