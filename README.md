# Facial Expression Recognition

This project implements a Convolutional Neural Network (CNN) for facial expression recognition. The goal is to classify facial expressions into one of seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The model is trained on the FER2013 dataset, which contains grayscale images of faces labeled with their corresponding emotions.

## Dataset

The FER2013 dataset consists of 48x48-pixel grayscale images of faces. The dataset is split into training and test sets. The training set is used to train the CNN model, while the test set is used to evaluate the model's performance. The model achieves an accuracy of approximately 70% on the test set.

## Model Architecture

The CNN model consists of multiple layers, including convolutional layers, max-pooling layers, and fully connected layers. The architecture of the model is as follows:

1. Input layer: 48x48-pixel grayscale images
2. Convolutional layer with 32 filters and ReLU activation
3. Max-pooling layer with a pool size of (2, 2)
4. Convolutional layer with 64 filters and ReLU activation
5. Max-pooling layer with a pool size of (2, 2)
6. Convolutional layer with 128 filters and ReLU activation
7. Max-pooling layer with a pool size of (2, 2)
8. Convolutional layer with 256 filters and ReLU activation
9. Max-pooling layer with a pool size of (2, 2)
10. Flatten layer to convert 3D feature maps to a 1D vector
11. Fully connected layer with 128 units and ReLU activation
12. Dropout layer with a dropout rate of 0.3 to reduce overfitting
13. Fully connected layer with 7 units (one for each emotion category) and softmax activation

## Training

The model is trained using the Adam optimizer and sparse categorical cross-entropy loss. The training is performed over 10 epochs with a batch size of 32. The training process is monitored using accuracy as the evaluation metric. After training, the model achieves an accuracy of approximately 70% on the test set.

## Evaluation

The trained model is evaluated on the test set to measure its performance. The confusion matrix is computed to visualize the model's performance for each emotion category. The model shows good performance in classifying Happy and Neutral expressions but may have some challenges with distinguishing between similar emotions like Sad and Neutral.

## Conclusion

Facial expression recognition is a challenging task, and the achieved accuracy of 70% indicates that the model has learned to recognize patterns in facial expressions to some extent. However, there is still room for improvement, and further fine-tuning of the model or exploring more advanced architectures could potentially lead to better performance.

 
