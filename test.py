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