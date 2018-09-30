# sentiment_analysis_face

<p>
To train the model, we used Fer2013 datset that contains 30,000 images of facial expressions grouped in seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.
</p>
<br>
<p>
The faces are first detected using opencv, then we extract the face landmarks using dlib. We also extracted the HOG features and we input the raw image data with the face landmarks+hog into a convolutional neural network.
</p>
<br>
For our experiments, we used  CNN models:
<br><br>

![model_architecture](CNN_models.png)

<br>

## Download and prepare the data
<p>
Download Fer2013(Kaggle Fer2013 challenge) dataset and the Face Landmarks model(Dlib Shape Predictor model)
</p>

<br>

## Train the model
<p>
The training parameters were define at parameters.py
</p>
<code>python train.py --train=yes</code>
<br>
![training](training.jpeg)
<br>

## Evaluate the model
<p>
The evaluation parameters were define at parameters.py<br>
Set "save_model_path" parameter to the path of your pretrained file.
</p>
<code>python train.py --evaluate=yes</code>
<br>
![training](evaluation.jpeg)
<br>

## Prediction the model
<p>
The prediction parameters were define at parameters.py<br>
Set "save_model_path" parameter to the path of your pretrained file.
</p>
<code>python index.py --image=sample1.png</code>
<br>
<p>Extract face from input image </p>
![input](sample2.png)
<br>
![extracted](extract_sample.PNG)
<br>
<p>Then prediction on each image </p>
![predition](preditions.jpeg)
<br>





