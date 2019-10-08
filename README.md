# Face-Kinship-Recognition

Code for the Face Kinship Recognition competition from Kaggle

This project aims to deliver a solution for the face kinship recognition problem, proposed at https://www.kaggle.com/c/recognizing-faces-in-the-wild.

It makes use of the following libraries:
- Numpy (1.16.3)
- Pandas (0.24.2)
- Tensorflow-GPU (2.0.0-beta1)
- Opencv (4.1.0.25)

Training parameters are defined in the file **config.py**

Before running any code, the function **initialize_devices**, in the module **run**, must be run in order for the GPU's to be used.

The code first processes the input, reading the data from disk, through the module **input_manager**. The module generates both *train* and *validation* pipelines of data, composed of triplets, with each triplet containing an anchor image, a positive image (person is related to anchor) and a negative image (person is not related to anchor).

All the structures are contained in the module **models.py**, which loads pre_trained models and builds custom layers. The final choice for a pre-trained model was the **FaceNet Model**. The model creates a vector of embeddings for each image (anchor, positive and negative); then, L2 distances are taken between anchor and positive, and anchor and negative images by the next layer. The distances are, then, transformed into probabilities using 1 - tanh(&alpha; * distance + &beta;), where &alpha; and &beta; are trainable parameters. The positive and negative probability tensors, as well as the positive and negative distance tensors are all concatenated together to form the output tensor (the final output used by the loss function is composed of only the probability tensors, but the distance tensors were added because one can wish to monitor the distances during training, and tensorflow only allows metrics to be computed on the final output tensor).

The loss function, as well as the metrics utilized during training, are implemented in the **losses.py** module. The loss for a single entry is calculated as L = -(log(P<sub>p</sub> + &epsilon;) + log(1 - P<sub>n</sub> + &epsilon;))/2, where P<sub>p</sub> and P<sub>n</sub> are, respectively, the probability of the anchor and positive examples,  and the probability of the anchor and negative examples. &epsilon; is added to both terms to improve numerical stability. The batch loss is, as expected, computed as the average of each individual loss in the batch. Besides the loss, the module also displays some metrics: the embedding distances, the probabilities, and the area under curve (*AUC*) of the *ROC* curve taken from the probabilities.

For displaying metrics and and losses during training, the module **training_log_callbacks.py** was implemented; it basically contains a custom callback class, which receives a python dictionary containing all the information needed for plotting the desired variable. This way, multiple callback objects can be created, each being initialized with metadata from specific variables, and multiple plots can be generated during training using a single implementation. The module also manages the logs generated during training, doing tasks such as creating a folder for the training session and saving the configuration module and logs file into the folder.

Finally, the **run.py** module is responsible for initializing, training and predicting. It contains a initializer function (**initialize_devices**) that must be run before everything else is done, so the GPU's can be set. 

The model trained is able to achieve a score of 0.770.


