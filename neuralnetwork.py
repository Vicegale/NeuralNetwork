import math
import random

class Node:
    def __init__(self):
        self.net = 0
        self.out = 0
        
class Connection:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.weight = random.random()
        self.newWeight = None
        
class Layer:
    def __init__(self, nodeCount):
        self.nodes = []
        self.bias = None
        for i in range(nodeCount):
            self.nodes.append(Node())

    def setBias(self, newBias):
        self.bias = newBias

class Network:
    def __init__(self, layerDims, biases, learningRate):
        if len(layerDims) > 2 and len(biases) == len(layerDims) - 1:
            self.inputCount = layerDims[0]
            self.outputCount = layerDims[-1]
            #create layers
            self.layers = []
            for x in layerDims:
                self.layers.append(Layer(x))
            #hook biases
            for i in range(len(biases)):
                self.layers[i+1].bias = (biases[i])
            #connect layers
            self.connections = []
            for i in range(len(self.layers) - 1):
                for begin in self.layers[i].nodes:
                    for end in self.layers[i+1].nodes:
                        self.connections.append(Connection(begin, end))
            #set learning rate
            self.learningRate = learningRate
        else:
            print("Network badly built. Add hidden layers and biases to the hidden layers")
    
    def forwardPass(self, input):
        if len(input) == self.inputCount:
            for i in range(len(self.layers[0].nodes)):
                self.layers[0].nodes[i].out = input[i]
            for i in range(len(self.layers) - 1):
                 for node in self.layers[i+1].nodes:
                    layerConnections = [x for x in  self.connections if x.end is node]
                    node.net = sum(connection.start.out * connection.weight for connection in layerConnections) + self.layers[i+1].bias
                    node.out = 1/(1+math.exp(-node.net))
        else:
            print("Input list size mismatch")
            
    def calculateError(self, target):
        if len(target) == self.outputCount:
            return sum([(0.5)*(target[i] - self.layers[-1].nodes[i].out)**2 for i in range(len(target))])
        else:
            print("Target list size mismatch")
    
    def backpropagateError(self, target):
        #TODO
        #output layer propagation
        connections = [connection for connection in self.connections if connection.end in self.layers[-1].nodes]
        i = 0
        for conn in connections:
            relativeError = (conn.end.out - target[i]) * conn.end.out *(1-conn.end.out) * conn.start.out
            conn.newWeight = conn.weight - (relativeError * self.learningRate)
        for i in range(len(self.layers) - 2):
            #single layer propagation
            pass
        #hidden layers propagation
        
    def getOutput(self):
        return [node.out for node in self.layers[-1].nodes]
        
    def getInput(self):
        return [node.out for node in self.layers[0].nodes]   
        
if __name__ == "__main__":
    input = [0.1, 0.3]
    target = [0.3, 0.1]
    
    x = Network([2, 3, 2], [0.3, 0.2], 0.5)
    x.forwardPass(input)
    x.backpropagateError(target)
    print("Input: {0}".format(x.getInput()))
    print("Output: {0}".format(x.getOutput()))
    print("Target: {0}".format(target))
    print("Error: {0}".format(x.calculateError(target)))
    for i in range(len(x.layers)):
        print("---------\nLayer {0}".format(i))
        for node in x.layers[i].nodes:
            print("Node ID: {0}".format(hex(id(node))))
        print("Bias: {0}".format(x.layers[i].bias))
    print("---------\nNetwork has {0} connections:".format(len(x.connections)))
    for i in x.connections:
        print("Connection from {0} to {1} with weight {2}".format(hex(id(i.start)), hex(id(i.end)), i.weight))