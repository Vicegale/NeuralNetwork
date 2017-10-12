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
    network.training(andGateTrainingSet)
    
Currently, the network executes the training loop until the average error of the training set is less than 1e-06.

# Debugging

If for some reason the network isn't training, you can toggle the periodic error display. It will show you the average error of the training set every 10000 iterations:
    
    network.debug(True)
    
# Post-Training usage:

After training the network, you can use it by using the forwardPass(input) function with an input tuple:

    network.forwardPass((0, 1))
    
# Retrieving Output

Output can be retrieved using the getOutput() function:

    x = network.getOutput()
    
Note, however, that the output is returned as a list.

# Saving and Loading Networks

To avoid training a network everytime you use it, you may need to save it. For that, you can use pickle.

Saving:

    pickle.dump(network, open("filename.p", "wb"))
Loading:

    network = pickle.load(open("filename.p", "rb"))
    
Note that ![pickle is not a safe way to store objects and can be tampered with](https://docs.python.org/3/library/pickle.html).


# Full Example (test.py)

Mixing all the elements spoken above, we can set up the following script:

    import neuralnetwork
    import pickle

    #Creating, Training and Saving
    network = neuralnetwork.Network([2, 3, 1], [0.2, 0.1], 0.5)
    andGateTrainingSet = {(0, 0): (0,), (0, 1): (0,), (1, 0): (0,), (1, 1): (1,)}
    network.debug(True) #periodically shows the average error of the training set
    network.training(andGateTrainingSet)

    pickle.dump(network, open("filename.p", "wb"))

    #Loading and Using
    loadedNetwork = pickle.load(open("filename.p", "rb"))
    loadedNetwork.forwardPass((0, 1))
    result = loadedNetwork.getOutput()
    print(result) #outputs [0.001156001503972413] which is close to the intended output (0)
