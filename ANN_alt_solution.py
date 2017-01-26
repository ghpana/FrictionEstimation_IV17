# Alt. neural network


# Import packages
import numpy as np
import csv
from sklearn import preprocessing
from sklearn import cross_validation
import sys
import time
import matplotlib.pyplot as plt
import scipy.io
#import seaborn as sns
from sklearn.datasets import make_classification
#from pandas import DataFrame


# plt.figure(figsize=(12, 10))
# sns.corrplot(df, annot=False)

# Neural Network class
class NeuralNetwork:
    
    # Activation functions
    def active_tanh(x, Derivative=False):
        if Derivative:
            return (1.0 - np.tanh(x/2)**2)/2 # Derivate of tanh
        else:
            return np.tanh(x/2) # tanh(x/2)
    def active_softplus(x, Derivative=False): # WROING!!!
        if Derivative:
            return np.exp(x)/(np.exp(x)+1)
        else:
            return np.log(1+np.exp(x))
            
    def __init__(self):
        

        
        #X, y = make_classification(1000, n_features=20, n_informative=2, 
        #                   n_redundant=2, n_classes=2, random_state=0)

        
        #df = DataFrame(np.hstack((X, y[:, None])), 
        #       columns = list(range(20)) + list(["class"]))
        
        #_ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=1.5)
        #plt.figure(figsize=(12, 10))
        #_ = sns.corrplot(df, annot=False)


        # Learning rate
        self.learningrate = 0.01
        # Momentum
        self.momentum = 0.0
        # Reset weights, shape and functions
        self.shape = (47,47,47,2)
        self.afunctions = []
        self.weights = []

        # Layer info
        self.num_layers = len(self.shape)-1
        
        # Initialise the biases randomly with uniform distribution [-1,1]
        self.biases =  np.random.uniform(low=-1,high=1,size=self.num_layers)
        
        #Set the activation functions to tanh
        self.afunctions = [NeuralNetwork.active_tanh,NeuralNetwork.active_tanh,NeuralNetwork.active_tanh]
        #self.afunctions = [np.repeat(NeuralNetwork.active_tanh)]

        # Data from last Run
        self.Inputlayer = []
        self.Outputlayer = []
        self.prevWeightdelta = []
        
        # Create the weight arrays
        for (l1,l2) in zip(self.shape[:-1], self.shape[1:]):
            #Create random uiform distribution of weights between -0.1 and 0.1
            self.weights.append(np.random.uniform(low=-0.1, high=0.1, size = (l2, l1+1)))
            self.prevWeightdelta.append(np.zeros((l2, l1+1)))
    
    
    def FeedForward(self, input):
        
        num_samples = input.shape[0]

        # Clear input and output lists
        self.Inputlayer = []
        self.Outputlayer = []
        

        for i in range(self.num_layers):
            if i == 0:

                layerInput = self.weights[0].dot(np.vstack([input.T,\
                    np.repeat(self.biases[0],num_samples,axis=0)]))
            else:
                layerInput = self.weights[i].dot(np.vstack([self.Outputlayer[-1], \
                    np.repeat(self.biases[i],num_samples,axis=0)]))

            self.Inputlayer.append(layerInput)
            self.Outputlayer.append(self.afunctions[i](layerInput))


        # Return the output layer
        return self.Outputlayer[-1].T

    # Calculate and return the error
    def getError(self,target):
        odelta = np.absolute(target.T-np.sign(self.Outputlayer[-1]))
        error_value = 1/2/len(target)*np.sum(odelta)
        return error_value

    # Backpropagation algorithm
    def Backpropagation(self, input, target):
        
        # Number of samples
        num_samples = input.shape[0]

        deltas = []

        self.FeedForward(input)

        # Loop through the layers in reverse order and 
        #calculate the delta values
        for i in reversed(range(self.num_layers)):
            # If last layer
            if i == self.num_layers-1:

                output_delta = self.Outputlayer[i]-target.T
                deltas.append(output_delta * \
                    self.afunctions[i](\
                        self.Inputlayer[i], True))
            else:
                dd = self.weights[i + 1].T.dot(\
                    deltas[-1])
                deltas.append(dd[:-1, :] * \
                    self.afunctions[i](\
                        self.Inputlayer[i], True))
            
        # Compute weight deltas
        for i in range(self.num_layers):
            delta_index = self.num_layers - 1 - i
            
            if i == 0:
                layerOutput = np.vstack([input.T, np.repeat(self.biases[0],\
                    num_samples,axis=0)])
                
            else:
                layerOutput = np.vstack([self.Outputlayer[i-1], \
                    np.repeat(self.biases[i],num_samples,axis=0)])


            currentdeltaweight = np.sum(\
                                 layerOutput[None,:,:].transpose(2, 0 ,1) * \
                                 deltas[delta_index][None,:,:].transpose(2, 1, 0),\
                                 axis=0)

            weightDelta = self.learningrate*currentdeltaweight +\
                             self.prevWeightdelta[i]*self.momentum
            

            self.weights[i] -= weightDelta
            self.prevWeightdelta[i] = weightDelta
        return 0

#Start the program
if __name__ == "__main__":
    
    # Reset the input and results
    Input = []
    Result = []
    
    #Read the training file
    spamReader = csv.reader(open('train_data_2016.txt', newline=''), \
        delimiter=' ', quotechar='|')
    for row in spamReader:
        Input.append([float(row[0]),float(row[1])])
        Result.append([float(row[2])])
    
     #Convert to arrays
    Input = np.asarray(Input, dtype=np.float32)
    Result = np.asarray(Result,dtype=np.float32)

    # Normalizing, setting mean=0 and std = 1
    Input = preprocessing.scale(Input, with_mean=True,with_std=True)
    #Result = preprocessing.scale(Result, with_mean=True,with_std=True)


    # Read validation data
    ValidInput = []
    CorrectTarget = []
    spamReader = csv.reader(open('valid_data_2016.txt', newline=''), \
        delimiter=' ', quotechar='|')
    for row in spamReader:
        ValidInput.append([float(row[0]),float(row[1])])
        CorrectTarget.append([float(row[2])])

    #Convert to arrays
    ValidInput = np.asarray(ValidInput, dtype=np.float32)
    CorrectTarget = np.asarray(CorrectTarget, dtype=np.float32)

    # Normalizing, setting mean=0 and std = 1
    ValidInput = preprocessing.scale(ValidInput, with_mean=True,with_std=True)

    
    # Load newdataset from mat files
    mat_contents = scipy.io.loadmat('../cleareddataset.mat')
    cleareddataset = mat_contents['cleareddataset']
    #cleareddataset[:,4] = np.sign(cleareddataset[:,4])
    #cleareddataset[:,5] = np.sign(cleareddataset[:,5])
    #cleareddataset[:,6] = np.sign(cleareddataset[:,6])
    #cleareddataset[:,6] = np.sign(cleareddataset[:,6]-0.5)
    Input = cleareddataset[:,6:]
    Result = np.sign((cleareddataset[:,2]-0.5).reshape((-1,1)))
    print(cleareddataset[:,6:].shape)
    #print(Input)
    #print(Result)
    time.sleep(1)

    # Normalizing, setting mean=0 and std = 1
    Input = preprocessing.scale(Input, with_mean=True,with_std=True)
    Result = preprocessing.scale(Result, with_mean=True,with_std=True)
    #initilize the network
    nn = NeuralNetwork()

    #Number of iterations
    iterations = 2000
    #Batch size
    batch_size = 10
    last_errors = []

    #Save data to these files
    targetfileT = open('data_training_'+str(nn.shape)+'.txt', 'w')
    targetfileV = open('data_validation_'+str(nn.shape)+'.txt', 'w')

    #Number of experiments
    exp = 1

    # Print and save after this many iterations
    steps = 100

    errors_train = np.zeros(iterations/steps)
    errors_validation = np.zeros(iterations/steps)

    for itera in range(0,exp):

        # Reset the network
        nn = NeuralNetwork()
        
        # Counter
        error_item = 0

        # Run the back-propagation algorithm
        for i in range(iterations + 1):
            randomitem = np.random.randint(Input.shape[0])
            B=np.random.randint(Input.shape[0],size=batch_size)

            #New batch
            inp = Input#[B,:]
            res = Result#[B,:]

            #Mini-batch or sequence update
            nn.Backpropagation(inp, res)

            #Batch update
            #nn.Backpropagation(Input, Result)

            #Save and print error and iteration
            if i % steps == 0 and i > 0:
                nn.FeedForward(Input)
                err_t = nn.getError(Result)
                errors_train[error_item] += err_t
                
                '''nn.FeedForward(ValidInput)
                err_v = nn.getError(CorrectTarget)
                errors_validation[error_item] += err_v
                '''

                print("{iter} Iteration: {first} with error: {sec}".format(\
                    iter=itera+1,first=i, sec=err_t))

                error_item += 1

        # Get last error
        nn.FeedForward(Input)
        lerr = nn.getError(Result)
        last_errors.append(lerr)
        
        
        Output = nn.FeedForward(Input)
        countTrain = 0
        for i in range(Input.shape[0]):
            if np.sign(Output[i]) == Result[i]:
                countTrain+=1

        #print(countTrain)
        #print(len(Input))


        '''ValidOutput = nn.FeedForward(ValidInput)
        verr = nn.getError(CorrectTarget)

        countValid = 0
        for i in range(ValidInput.shape[0]):
            if np.sign(ValidOutput[i]) == CorrectTarget[i]:
                countValid+=1'''

        #print(countValid)
        #print(len(ValidInput))

        targetfileT.write(str(lerr)+'\n')
        '''targetfileV.write(str(verr)+'\n')'''

    #Get the averages
    errors_train /= exp
    errors_validation /= exp

    # Save the errors into mat-files
    scipy.io.savemat('data_training_e_'+str(nn.shape)+'.mat', \
        mdict={'train': errors_train})
    scipy.io.savemat('data_validation_e_'+str(nn.shape)+'.mat', \
        mdict={'validation': errors_validation})

