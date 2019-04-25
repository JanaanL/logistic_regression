
# coding: utf-8

# In[166]:


import numpy as np
def load_data(path, add_bias=True):
    """
    Loads and processes the bank note data set
    
    Inputs:
    -path:  string representing the path of the file
    -add_bias:  boolean representing whether an extra column of ones is added to the data, representing
                a bias value
    
    Returns:
    -X:  a numpy array of shape [no_samples, no_attributes (+1 if add_bias is True)]
    -y:  a numpy array of shape [no_samples] that represents the labels {-1, 1} for the dataset X
    """
        
    data = []
    with open(path, 'r') as f:
        for line in f:
            example = line.strip().split(',')
            if len(example) > 0:
                example = [float(i) for i in example]
                data.append(example)
    X = np.array(data, dtype=np.float64)
    y = X[:,-1]
    y = y.astype(int)
    y[y == 0] = -1
    X = X[:,:-1]
    
    if add_bias == True:
        bias = np.ones((X.shape[0],1),dtype=np.float64) 
        X = np.hstack((X,bias))

    return X, y


# In[167]:


def logistic_regression(X, y, epochs, model='MAP', variance=1.0, gamma=2.0, d=2.0, batch_size=16):
    """
    Uses the logistic regression MAP estimator to predict a linear classifier.
    
    Input:
    - X:  a numpy array of shape [no_samples, no_attributes] representing the dataset
    - y:  a numpy array of shape [no_samples] that has the labels {-1, 1} for the dataset
    - epochs:  an int that represents the number of epochs to perform.
    - model:  A string from {'MAP', 'ML'}, representing the type of logistic regression to perform.  
    - variance: a float hyperparamter that represents  different variances in the prior distribution 
    - gamma:  A float hyperparameter that represents the learning rate to be used in calculating the gradient.
    - d:  a float hyperparamter used in the learning rate schedule
    
    Returns:
    -w:  a numpy array of shape [no_attributes ] representing the set of weights learned in the algorithm.
    """    

    N = X.shape[0]
    w = np.zeros((X.shape[1],1)) # initialize the weights as zero
    subgradient = []
    t = 0
    
    for epoch in range(epochs):
        
        #Randomly shuffle data
        index = np.random.permutation(N)
        X = X[index]
        y = y[index]
        start = 0
        end = batch_size
        
        #Iterate through randomly shuffled training examples
        while(end < N):
            x_batch = X[start:end]
            y_batch = y[start:end].reshape(-1,1)
            t += 1 
            xy = y_batch * x_batch
            sigmoid = (-1 / (1 + (np.exp(np.dot(xy, w))))).reshape(-1,1)
            gradient = np.dot(xy.T, sigmoid)
            if model == "MAP":
                gradient += w / variance
            learning_rate = gamma / (1 + gamma / d * t)
            #gradient /= batch_size
            w = w - learning_rate * gradient
            subgradient.append(np.sum(gradient * learning_rate))
            start += batch_size
            end += batch_size
     
    #Plot the convergence data
    from matplotlib import pyplot as plt
    plt.title('Convergence of Logistic Regression')
    plt.xlabel('Iteration')
    plt.ylabel('Update')
    plt.plot(subgradient)
    plt.show()
    
    return w
    


# In[168]:


def predict(X, y, w):
    """
    Computes the average prediction error of a dataset X and given weight vector w.
    
    Inputs:
    -X:  a numpy array of shape [no_samples, no_attributes + 1] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels for the dataset
    -w:  a numpy array of shape [no_attributes + bias] representing the set of weights to predict the labels.
    
    Returns:
    -error:  A float representing the average prediction error of for the dataset X.
    """
    
    incorrect = 0
    for i in range(X.shape[0]):
        if np.sign(np.dot(X[i].reshape((-1,1)).T, w)) != y[i]:
            incorrect += 1
           
    print("The total incorrect is " + str(incorrect) + " out of " + str(X.shape[0]))
    return float(incorrect) / X.shape[0] * 100


# In[170]:


#Test the Logistic Regressions
X, y = load_data('/Users/janaanlake/Documents/CS_5350/HW3/bank-note/train.csv', add_bias=True)
X_test, y_test = load_data("/Users/janaanlake/Documents/CS_5350/HW3/bank-note/test.csv", add_bias=True)

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
print("Results for MAP logistic regression:")
print("\n")
for v in variances:
    print("Results for the hyperparameters v = " + str(v) + " :")
    w = logistic_regression(X, y, 10, model="MAP", variance=v)
    #print("The weights are: ")
    #print(w)
    print("The average training error is " + "{0:.1f}".format(predict(X, y, w)) + "%")
    print("The average test error for is " + "{0:.1f}".format(predict(X_test, y_test, w)) + "%")
    print('\n')
    
print("Results for ML logistic regression:")
w = logistic_regression(X, y, 10, model="ML")
print("The average training error is " + "{0:.1f}".format(predict(X, y, w)) + "%")
print("The average test error for is " + "{0:.1f}".format(predict(X_test, y_test, w)) + "%")

