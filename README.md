# Meme-Sentiment-Analysis-using-Multi-Modal-NN
This is a multi modal NN trained to detect the sentiment of memes, and deployed through Flask with a front-end for inputting images and getting their sentiment inference.

Data Preprocessing and Loading:

Images: You're resizing the images to 256x256 and then normalizing them using given mean and standard deviation values.
Text: The textual content of the memes (captions) is tokenized and transformed into numerical form. Textual data is padded to have uniform length across all samples.
Image Processing Sub-network (ModelA or NN):

Flattening Layer: Converts the 3D tensor (from the image) into a 1D tensor.
Fully Connected Layer 1 (inn): Has 196608 input neurons (representing a flattened 256x256 image with 3 color channels) and 256 output neurons. It uses the sigmoid activation function.
Hidden Layer 1 (hidden1): 256 input neurons and 128 output neurons with a sigmoid activation.
Hidden Layer 2 (hidden2): 128 input neurons and 64 output neurons with a sigmoid activation.
Output Layer (outt): 64 input neurons and 3 output neurons. This layer provides predictions for image sentiment. (Note: No activation is mentioned for this layer, so it's a linear layer).
Text Processing Sub-network (ModelB or NN_text):

Flattening Layer: Converts the input tensor into a 1D tensor.
**Fully Connected Layer 1 (inn)**: Has 187 input neurons (coming from the padded text) and 256 output neurons. Uses the sigmoid activation function.
**Hidden Layer 1 (hidden1)**: 256 input neurons and 128 output neurons with a sigmoid activation.
**Hidden Layer 2 (hidden2)**: 128 input neurons and 64 output neurons with a sigmoid activation.
**Output Layer (outt)**: 64 input neurons and 3 output neurons. This layer provides predictions for text sentiment.
Combined Model (Combined_model): It takes the outputs of both ModelA and ModelB (image and text outputs respectively).
The outputs are concatenated into a single tensor.
This tensor then passes through a fully connected layer (classifier), which has 6 input neurons (3 from image sub-network and 3 from text sub-network) and 3 output neurons.
The final output is the combined sentiment prediction for the meme.




