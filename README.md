# HandwrittenDigitsRecognizer
Handwritten digits recognizer using a neural network

# How to use
* Download the MNIST training images from [here](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
* Download the MNIST training labels from [here](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
* Compile the neural network file: `javac NeuralNetwork.java`
* Compile the JavaFX file: `javac --add-modules $FX_PATH --add-modules javafx.controls HandRecognizer.java` were `$FX_PATH` contains the path of your lib folder in the javafx installation directory.
* (optional) Train the model: `java NeuralNetwork.java`
* Run the JavaFX application: `java --add-modules $FX_PATH --add-modules javafx.controls HandRecognizer`

# Instructions
* Draw a number on the left (left mouse button) and the network will tell you what number it is and the confidence.
* Use the `DELETE` key to clear the canvas
* Use the digits from `0` to `9` to train the network while using it (for example if it recognizes some digits wrongly)

All the training data is saved in the file `training.txt`