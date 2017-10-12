import neuralnetwork
import pickle


trainingSet = {(0, 0): (0,), (0, 1): (1,), (1, 0): (1,), (1, 1): (0,)}
x = neuralnetwork.Network([2, 3, 1], [0.3, 0.2], 0.5)    
print("Network created. Training...")
x.training(trainingSet)
print("Training Complete. Running....")
'''pickle.dump(x, open("XOR.p", "wb"))
print("Saved!")
x = pickle.load(open("AND.p", "rb"))
'''

x.forwardPass((1, 1))
parsedOutput = round(x.getOutput()[0])
print(parsedOutput)'''