import NeuralNetwork as neural
import json
import random
import numpy as np

#load training data
f = open("json data/training_data.json", "r")
training_data = json.loads(f.read())


#define NN
nn = neural.NeuralNetwork(2,2,1)
#set learning rate
nn.learning_rate = 0.001
#start predictions
print("iteration: "+str(0))
print("[0,0] = "+ str(nn.predict([0,0])))
print("[1,0] = "+ str(nn.predict([1,0])))
print("[0,1] = "+ str(nn.predict([0,1])))
print("[1,1] = "+ str(nn.predict([1,1])))

#train for 50000 times
for i in range(100):
    for j in range(500):
        item = random.choice(training_data)
        nn.train(item.get("inputs"), item.get("targets"))

    #show progress:
    print("iteration: "+str(i+1))
    print("[0,0] = "+ str(nn.predict([0,0])))
    print("[1,0] = "+ str(nn.predict([1,0])))
    print("[0,1] = "+ str(nn.predict([0,1])))
    print("[1,1] = "+ str(nn.predict([1,1])))
    

