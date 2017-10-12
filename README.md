# NeuralNetwork
My first knock on an Artificial Neural Network with configurable layer and node counts. I followed the step by step instructions found in ![here](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/).


# Importing

You can import the script using:

    import neuralnetwork

# Instanciation:

You can instanciate the network just like any object:

    network = neuralnetwork.Network([2, 2, 2], [0.35, 0.6], 0.5)

This will instanciate a network similar to the image below:

![Neural Network](https://matthewmazur.files.wordpress.com/2018/03/neural_network-9.png)


# Training

In order to train the network, you must compile the existing Input/Output pairs into a dictionary, like the following example:

    andGateTrainingSet = {(0, 0): (0,), (0, 1): (0,), (1, 0): (0,), (1, 1): (1,)}
    
# Post-Training usage:

After training the network, you can use it by using the forwardPass(input) function with an input tuple:

    network.forwardPass((0, 1))
    
# Retrieving Output

Output can be retrieved using the getOutput() function:

    x = network.getOutput()
    
Note, however, that the output is returned as a list.
