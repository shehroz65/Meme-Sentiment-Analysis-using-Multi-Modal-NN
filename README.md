This repository contains code for training a combined sentiment analysis model that processes both images and text from memes to classify their overall sentiment. The model uses a neural network to process the image data and a separate neural network to process the text data. These two networks are then combined into a single model that outputs the sentiment classification.

The dataset is taken from https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset
Make sure to copy paste the images from there into this repos image folder.

To run this file:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Or you can also run it in Docker.
