"""
Implementation of algorithm to train N-Layer Neural Network.

Author: Tan Pengshi Alvin
"""
import numpy as np
import pandas as pd

class Layer:
    def __init__(self, units=None, input_units=None, activation=None, l2_regularization=0):
            self.input_units = input_units
            self.output_units = units
            self.regularization = l2_regularization
            assert activation in [None,'tanh','relu','sigmoid','softmax'], 'Activation not recognized'
            self.activation = activation

class Model:
    def __init__(self, layers, random_state=21):
        
        np.random.seed(random_state) 

        self.parameters = {}
        self.velocity = {}
        self.activations = {}
        self.regularization ={}
        self.cache = {}
        self.X = None
        self.Y = None
        
        self.L = len(layers)
        
        for l in range(1,self.L+1):
            
            layer = layers[l-1]
            
            if layer.input_units:
                input_units = layer.input_units
                output_units = layer.output_units
            else:
                input_units  = previous_output_units
                output_units = layer.output_units
               
            # Apply He Initialization
            self.parameters['W' + str(l)] = np.random.randn(output_units,input_units) * np.sqrt(2./input_units)  
            self.parameters['b' + str(l)] = np.zeros((layer.output_units,1)) 
            self.velocity['W' + str(l)] = np.zeros((output_units,input_units))
            self.velocity['b' + str(l)] = np.zeros((layer.output_units,1))
            self.activations['a' + str(l)] = layer.activation
            self.regularization['r' + str(l)] = layer.regularization
            
            previous_output_units = layer.output_units

    def forward(self, X):

        locals()['A0'] = X
        for l in range(1,self.L+1):
            locals()['W'+str(l)] = self.parameters['W' + str(l)]
            locals()['b'+str(l)] = self.parameters['b' + str(l)]
            locals()['a'+str(l)] = self.activations['a' + str(l)]

            locals()['Z'+str(l)] = np.dot(eval('W'+str(l)),eval('A'+str(l-1))) + eval('b'+str(l))
            if eval('a' + str(l))=='tanh':
                locals()['A'+str(l)] = np.tanh(eval('Z'+str(l)))
            elif eval('a' + str(l))=='relu':
                locals()['A'+str(l)] = np.maximum(eval('Z'+str(l)),0)
            elif eval('a' + str(l))=='sigmoid':
                locals()['A'+str(l)] = 1/(1+np.exp(-eval('Z'+str(l))))
            elif eval('a' + str(l))=='softmax':
                locals()['A'+str(l)] = np.exp(eval('Z'+str(l)))/np.sum(np.exp(eval('Z'+str(l))),axis=0, keepdims = True)

            self.cache['Z'+str(l)] = eval('Z'+str(l))
            self.cache['A'+str(l)] = eval('A'+str(l))

        return eval('A'+str(self.L))


    def loss(self, AL, Y):

        m = Y.shape[1] # number of examples
        regularization_term = 0
        for l in range(1,self.L+1):
            regularization_term += self.regularization['r' + str(l)]*np.linalg.norm(self.parameters['W' + str(l)],ord='fro')
        regularization_term /= (2*m)

        if self.activations['a'+str(self.L)]=='sigmoid': 
            cost = np.sum(((- np.log(AL))*Y + (-np.log(1-AL))*(1-Y)))/m  + regularization_term # compute cost
        elif self.activations['a'+str(self.L)]=='softmax':
            loss_each_example = -np.sum(Y * np.log(AL),axis=1)
            all_losses = np.sum(loss_each_example)
            cost = all_losses/m + regularization_term  # compute cost

        return cost

    def backward(self, learning_rate=0.05, beta=0.9):

        m = self.X.shape[1] # number of examples

        locals()['A0'] = self.X
        for l in range(1,self.L+1):
        # First, retrieve all weights and biases from the dictionary "parameters".
            locals()['W'+str(l)] = self.parameters['W' + str(l)]
            locals()['b'+str(l)] = self.parameters['b' + str(l)]
        # Retrieve activations and regularization parameters
            locals()['a'+str(l)] = self.activations['a' + str(l)]
            locals()['r'+str(l)] = self.regularization['r' + str(l)]
        # Retrieve also all activations (A and Z) from dictionary "cache".
            locals()['A'+str(l)] = self.cache['A' + str(l)]
            locals()['Z'+str(l)] = self.cache['Z' + str(l)]

        # Backward propagation: calculate dW1, db1, ..., dWL, dbL.
        for l in reversed(range(1,self.L+1)):

            if eval('a' + str(l))=='sigmoid' or eval('a' + str(l))=='softmax':
                locals()['dZ'+str(l)] = eval('A'+str(l))-self.Y
            elif eval('a' + str(l))=='tanh':
                locals()['dZ'+str(l)] = np.multiply(np.dot(eval('W'+str(l+1)).T,eval('dZ'+str(l+1))),(1-np.square(eval('A'+str(l)))))
            elif eval('a' + str(l))=='relu':
                locals()['dZ'+str(l)] = np.multiply(np.dot(eval('W'+str(l+1)).T,eval('dZ'+str(l+1))),eval('Z'+str(l))>0)
            
            locals()['dW'+str(l)] = np.dot(eval('dZ'+str(l)),eval('A'+str(l-1)).T) / m + eval('r'+str(l))*eval('W'+str(l)) / m 
            locals()['db'+str(l)] = np.sum(eval('dZ'+str(l)),axis=1,keepdims=True) / m

        # Update rule for each parameter
        for l in range(1,self.L+1):

            self.velocity['W'+str(l)] = beta*self.velocity['W'+str(l)] + (1-beta)*eval('dW'+str(l))
            self.velocity['b'+str(l)] = beta*self.velocity['b'+str(l)] + (1-beta)*eval('db'+str(l))

            locals()['W'+str(l)] -= learning_rate * self.velocity['W'+str(l)]
            locals()['b'+str(l)] -= learning_rate * self.velocity['b'+str(l)]

            self.parameters['W' + str(l)] = eval('W'+str(l))
            self.parameters['b' + str(l)] = eval('b'+str(l))


    def get_batch(self, X, Y, batch_size):

        m = X.shape[1]       # number of training examples

        batch_start_index = np.arange(0,m,batch_size)
        num_batches = len(batch_start_index)
        mini_batches = []

        for batch in range(1, num_batches+1):

            if batch < num_batches:
                mini_batch_X = X[:,(batch-1)*batch_size:batch*batch_size]
                mini_batch_Y = Y[:,(batch-1)*batch_size:batch*batch_size]
            elif batch == num_batches:
                mini_batch_X = X[:,(batch-1)*batch_size:]
                mini_batch_Y = Y[:,(batch-1)*batch_size:]

            mini_batch = (mini_batch_X,mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches 


    def fit(self, 
            X, 
            Y, 
            validation_data=None, 
            batch_size=32, 
            epochs=100, 
            learning_rate=0.05, 
            beta=0.9,
            patience=None):

        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(Y, (pd.core.frame.DataFrame,pd.core.series.Series)):
            Y = Y.values

        if X.ndim > 1:
            X = X.T
        else:
            X = np.expand_dims(X, axis=1)

        if Y.ndim > 1:
            Y = Y.T
        else:
            Y = np.expand_dims(Y, axis=0)

        self.full_Y = Y
        self.full_X = X

        mini_batches = self.get_batch(X, Y, batch_size)

        if validation_data:
            X_val, Y_val = validation_data

            if isinstance(X_val, pd.core.frame.DataFrame):
                X_val = X_val.values
            if isinstance(Y_val, (pd.core.frame.DataFrame,pd.core.series.Series)):
                Y_val = Y_val.values

            if X_val.ndim > 1:
                X_val = X_val.T
            else:
                X_val = np.expand_dims(X_val, axis=1)

            if Y_val.ndim > 1:
                Y_val = Y_val.T
            else:
                Y_val = np.expand_dims(Y_val, axis=0)

        if patience:
            epoch_cache ={}        

        for i in range(0, epochs):   

            for mini_batch in mini_batches:

                mini_batch_X, mini_batch_Y = mini_batch

                self.X = mini_batch_X
                self.Y = mini_batch_Y
                
                AL = self.forward(mini_batch_X)           
                cost = self.loss(AL, mini_batch_Y)  
                
                self.backward(learning_rate,beta)

            if validation_data:
                AL_val = self.forward(X_val)
                cost_val = self.loss(AL_val, Y_val)
            
            # Print the cost every epoch
            if validation_data:
                print(f"Epoch {i+1} - Train_Loss:{cost}  Val_Loss:{cost_val}")
            else:
                print(f"Epoch {i+1} - Train_Loss:{cost}")

            if patience:
                if not epoch_cache or cost_val <= epoch_cache['best_loss']:
                    epoch_cache['best_loss'] = cost_val
                    epoch_cache['best_parameters'] = self.parameters
                    counter=0
                elif cost_val > epoch_cache['best_loss']:
                    counter += 1

                if counter == patience:
                    self.parameters = epoch_cache['best_parameters']
                    print(f"Early Stopping! Best Epoch {i+1-patience} - Val_Loss:{epoch_cache['best_loss']}")
                    break

        return self.parameters


    def predict(self, X):

        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        if X.ndim > 1:
            X = X.T
        else:
            X = np.expand_dims(X, axis=1)

        AL = self.forward(X)

        if AL.shape[0]==1:
            predictions = AL[0]>0.5
        else:
            predictions = (AL == np.max(AL,axis=0)).astype(bool).T
        
        return predictions
        


