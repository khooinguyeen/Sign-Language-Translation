# LNT model for recognizing Vietnamese sign language

![Python](https://img.shields.io/badge/Language-Python-blue?logo=python)
![Tensorflow](https://img.shields.io/badge/Framework-Tensorflow-important?logo=tensorflow)
![Numpy](https://img.shields.io/badge/Package-Numpy-%23150458?logo=numpy)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-%23white?logo=opencv)

This folder contains the model code for the app "Look & Tell"

* [Look & Tell (Beta) - communication support solution for deaf people with artificial intelligence](https://github.com/khooinguyeen/LookandTell-OfficialApp)

<img src="https://github.com/khooinguyeen/Sign-Language-Translator/blob/main/Sign%20Language%20Translator/Demo/lookandtell.png" width="450">

# 1. About this code
* A deep-learning neural network for hand action recognition, made by Vietnamese students, to translate Vietnamese sign language

# 2. Motivation
* The story begins when Kiet becomes my neighbor, he is a born deaf, ie since he was born, he has not been able to hear his mother's voice. And every day, for 10 years - until now, that child is in 5th grade, I have witnessed every day that the child struggles, has difficulty in expressing the simplest wishes of an elementary school child.
* Learn about the deaf and hard of hearing, there are currently 1.5 billion speech and hearing impaired people in the world, and 2.5 million people in Vietnam - a huge community! However, it entails a huge economic burden for countries. According to a 2015 WHO report, the cost of speech and hearing impairment ranges from $750 to $790 billion. By 2019 it will exceed $981 billion (according to Global Burden Disease), mainly due to community and social costs - an too great economic burden for countries. * In Vietnam, the number of qualified sign language interpreters is only about 20 people, and there are no reasonable communication support products (only 1 product is produced by a student of Ho Chi Minh City University of Science and Technology but requires an additional one). equipment costs nearly 1 million VND), while conducting a survey of 250 people including deaf and hard of hearing people, it was found that nearly 100% of people want to create a bridge to integrate and really need a product that supports communication. 
* **With the desire to talk with Kiet (my neighbor) as well as all the deaf and hard of hearing people in the world, my friend and I decided to do this project.**

# 3. How to use this code
### (1) clone LNT
```
git clone https://github.com/khooinguyeen/Sign-Language-Translator.git
```

### (2) install appropriate dependencies
```
!pip install tensorflow==2.5.0 tensorflow-gpu==2.5.0 opencv-python mediapipe sklearn matplotlib
```

### (3) go to the file [ActionDetection.ipynb](https://github.com/khooinguyeen/Sign-Language-Translator/blob/main/Sign%20Language%20Translator/ActionDetection.ipynb) to play with ipython notebooks and run the model
* Relative path:
```
Sign Language Translator\ActionDetection.ipynb
```

### (4) to collect more data by your own, go to the file [CollectData.ipynb](https://github.com/khooinguyeen/Sign-Language-Translator/blob/main/Sign%20Language%20Translator/CollectData.ipynb)
* Relative path:
```
Sign Language Translator\CollectData.ipynb
```

# 3. Demo
* **Me testing the model :>**
![](Demo/../Sign%20Language%20Translator/Demo/bạn%20hiểu%20ngôn%20ngữ%20ký%20hiệu%20không.gif)



* **My cute little neighbour Kiet testing the model :>**
![](Sign%20Language%20Translator/Demo/tôi%20thích%20ăn%20mì.gif)

# 4. Contact
* If you appreciate the humanity of the project or have a problem or even just want to talk to us, please contact us via email: **khoinguyenmai17102005@gmail.com**
