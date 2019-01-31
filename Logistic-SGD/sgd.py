from math import exp
import random

def logistic(x):
    return 1./(1.+exp(-x))

def dot(x, y):
    s = 0
    for i in range(len(x)):
        s += x[i]*y[i]
    return s

def predict(model, point):
    s = dot(model, point['features'])
    return logistic(s)
    

def accuracy(data, predictions):
    correct = 0
    for i in range(len(data)):
        correct += (data[i]['label'] == (predictions[i]>=0.5))
    return float(correct)/len(data)

# TODO: Update model using learning rate and L2 regularization
def update(model, point, rate, lam):
    prob = predict(model,point)
    for i in range(len(model)):
        model[i] += rate*(point['features'][i]*(point['label']-prob) - lam*model[i])
        
def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

# TODO: Train model using training data
def train(data, epochs, rate, lam):
    model = initialize_model(len(data[0]['features']))
    for i in range(epochs):
        for j in range(len(data)):
            update(model, data[random.randint(0,len(data)-1)], rate, lam)
        
    return model
        
def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(r['marital'] == 'Married-civ-spouse')
        #TODO: Add more feature extraction rules here!
        point['features'] = features
        data.append(point)
    return data

# TODO: Tune your parameters for final submission
def submission(data):
    return train(data, 1, .01, 0)
    
