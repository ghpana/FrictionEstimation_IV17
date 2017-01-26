# A implementation of tensorflow  for road friction prediction
# Artificial neural network

# Import all packages
import scipy.io
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

# Parameters:
# Learning rate for stochastic gradient descent (default: 0.001)
learning_rate = 0.0001
# Number of epochs
training_epochs = 5000
# Batch size (default: 10)
batch_size = 10
# Display results after this amount of steps (default: 20)
display_step = 10
# Threshold for slippery (default: 4.5)
slippery = 0.35
# Minimum  quality of measurement (default: 4.5)
min_quality = 4
# Number of folds in KFold (default: 5)
kfolds = 3
# CSV file name
#filename = 'data/cleareddataset_slack0min_1min_0.03_4_833373149_or_833373150.csv'
filename = 'data/new/cleareddataset_748861108_748861109_or_748861110_or_748861111_slack60.csv'
# Other files
#cleareddataset_709255549_or_709255550_or_709255551_slack0
#cleareddataset_748861108_748861109_or_748861110_or_748861111_slack0


# Use pca?
use_pca = False

# Different setups for the feature sets
W_All = [1,2,3,4,5,6,8,9,10,11,12,13,19,20,21,22,23,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
WO_Prevfriction = [1,2,3,4,5,6,7,8,9,10,11,12,13,19,20,21,22,23,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
WO_Dewpoint = [1,2,3,4,8,9,10,11,12,13,19,20,21,22,23,29,30,31,32,33,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
#WO_Duration = [1,2,3,4,6,8,9,10,11,12,13,19,20,21,22,23,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
#WO_Distance = [1,2,3,4,5,8,9,10,11,12,13,19,20,21,22,23,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
WO_Wiperspeed = [1,2,3,4,8,9,10,11,12,13,19,20,21,22,23,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64];
WO_Temp = [1,2,3,4,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
WO_Rain = [1,2,3,4,8,9,10,11,12,13,19,20,21,22,23,34,35,36,37,38,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
WO_Snow = [1,2,3,4,8,9,10,11,12,13,19,20,21,22,23,39,40,41,42,43,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
W_4hours = [1,2,3,4,5,6,8,9,10,11,12,13,19,20,21,22,23,29,30,31,32,33,44,45,46,47,48,54,55,56,57,58,59,60,61,62,63,64];
W_3hours = [1,2,3,4,5,6,8,9,10,11,12,13,18,19,20,21,22,23,28,29,30,31,32,33,38,43,44,45,46,47,48,53,54,55,56,57,58,59,60,61,62,63,64];
W_2hours = [1,2,3,4,5,6,8,9,10,11,12,13,17,18,19,20,21,22,23,27,28,29,30,31,32,33,37,38,42,43,44,45,46,47,48,52,53,54,55,56,57,58,59,60,61,62,63,64];
W_1hours = [1,2,3,4,5,6,8,9,10,11,12,13,16,17,18,19,20,21,22,23,26,27,28,29,30,31,32,33,36,37,38,41,42,43,44,45,46,47,48,51,52,53,54,55,56,57,58,59,60,61,62,63,64];

which_features = [7,14,15,16,17,18,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43]
which_features3h = [7,14,15,16,17,24,25,26,27,34,35,36,37,39,40,41,42,49,50,51,52]
#which_features = which_features3h

temp_features = np.array(range(1,65))

# Select feature set
#which_features = np.delete(temp_features,[x-1 for x in WO_Prevfriction])

# Index start at 0
which_features[:] = [x-1 for x in which_features]
print(which_features)

# Apply PCA
if use_pca == True:
    num_features = 15 # Select the number of dimensions
else:
    num_features = len(which_features)


n_hidden_1 = num_features # 1st layer number of features
n_hidden_2 = num_features # 2nd layer number of features
n_hidden_3 = num_features # 3nd layer number of features
n_hidden_4 = num_features # 4nd layer number of features
n_hidden_5 = num_features # 4nd layer number of features

n_input = num_features # Number of inputs
n_classes = 2 # Number of classes.

# tf Graph input (for debugging)
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# For dropout technique (not used)
keep_prob = tf.placeholder(tf.float32)
logs_path = '/tmp/tensorflow_logs/example'


# Helper class for building training- and validation datasets.
class HelperDataSets(object):
    def __init__(self, loadeddataset,train_indices,test_indices):
        
        # Weight the friction measurements w.r.t time and distance

        # Max search region (gps metric.)
        d_t = 0.04

        # Get the distances from the three last friction measurements
        d_1 = loadeddataset[4]
        d_2 = loadeddataset[57]
        d_2.loc[d_2>d_t] = d_t
        d_3 = loadeddataset[60]
        d_3.loc[d_3>d_t] = d_t

        # Get total distance
        d_tt = (1-d_1/d_t)+(1-d_2/d_t)+(1-d_3/d_t)
        
        # Max 5 hours
        t_t = 5* 60

        # And the duration since the last three measurements
        t_1 = loadeddataset[5]
        t_2 = loadeddataset[58]
        t_2.loc[t_2>t_t] = t_t
        t_3 = loadeddataset[61]
        t_3.loc[t_3>t_t] = t_t

        # Calculate total time
        t_tt = (1-t_1/t_t)+(1-t_2/t_t)+(1-t_3/t_t)

        # Get the friction values from the three last measurements
        # Remove measuremments thats too far or old 
        f_1 = loadeddataset[6]
        f_2 = loadeddataset[59]
        f_2.loc[(d_2 >= d_t) | (t_2 >= t_t)] = 0
        f_3 = loadeddataset[62]
        f_3.loc[(d_3 >= d_t) | (t_3 >= t_t)] = 0

        # Calculate the weighted friction values w.r.t time and distance
        f_prev_dis = (f_1*(1-d_1/d_t)+f_2*(1-d_2/d_t)+f_3*(1-d_3/d_t))/d_tt
        f_prev_dur = (f_1*(1-t_1/t_t)+f_2*(1-t_2/t_t)+f_3*(1-t_3/t_t))/t_tt

        # Update the friction measurements by a new weighted friction value
        #loadeddataset[6] = (f_prev_dis+f_prev_dur)/2

        
        # Extract the eplanatory and responsive datasets
        x = loadeddataset.iloc[:,which_features]
        y = loadeddataset.iloc[:,2]
        print(x)
        # Limit the target data to one or zero
        y = np.sign(y-slippery)
        y = -y
        y[y==-1] = 0
        #y[1:300] = 0

        # Normalizing the training set
        x_norm = preprocessing.scale(x)
        x = x_norm

        if use_pca == True:
            # Apply PCA
            pca = PCA(n_components=num_features)
            pca.fit(x)
            x = pca.transform(x)
        

        # Without KFold, use this instead
        #x_train, x_val, y_train, y_val = train_test_split(x,
        #                                                  y,
        #                                                  test_size=0.3,
        #                                                  #random_state=39)

        x_train, x_val = x[train_indices], x[test_indices]
        y_train, y_val = y[train_indices], y[test_indices]
        print(train_indices)
        print(y_train)
        print(y)
        # Create a label array with ones and zeros
        y_train = y_train.astype(int)
        lb = preprocessing.LabelBinarizer()
        lb.fit([0, 1, 2])
        new_train = lb.transform(y_train)
        new_train = np.delete(new_train,-1,1)

        # Create a dataframe and add it to a DataSet module
        new_train = pd.DataFrame(new_train)
        self.train = DataSet(x_train, new_train)

        # Create a dataframe and add it to a DataSet module
        y_val = y_val.astype(int)
        new_y_validation = lb.transform(y_val)
        new_y_validation = np.delete(new_y_validation,-1,1)
        self.validation = DataSet(x_val, pd.DataFrame(new_y_validation))


# The DataSet class is a container for traning and validation data
# Useful when doing stochastic gradient descent
class DataSet(object):

    # Initialize the module
    def __init__(self, explanatory, labels=None, fake_data=False):

        # Store the data
        self._num_examples = explanatory.shape[0]
        explanatory = explanatory.astype(np.float32)
        self._explanatory = explanatory
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def explanatory(self):
        return self._explanatory

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._explanatory = self._explanatory[perm,:]
            self._labels = self._labels.iloc[perm,:]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

            # Unit test
            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        return self._explanatory[start:end], self._labels[start:end]



# Create multilayer perceptron model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #layer_1 = tf.nn.dropout(layer_1,keep_prob) # Dropout layer 1
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #layer_2 = tf.nn.dropout(layer_2,keep_prob) # Dropout layer 2
    
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    #layer_3 = tf.nn.dropout(layer_3,keep_prob) # Dropout layer 3

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    #layer_4 = tf.nn.dropout(layer_4,keep_prob) # Dropout layer 4

    # Hidden layer with RELU activation
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1,weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

#GradientDescentOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.initialize_all_variables()

# Create a summary to monitor cost tensor
tf.scalar_summary("loss", cost)

# Create a summary to monitor accuracy tensor
tf.scalar_summary("accuracy", accuracy)

# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()


#I generate a KFold in order to make cross validation
k_fold = sk.model_selection.KFold(n_splits=kfolds, shuffle=True, random_state=1)

# Read the CSV file, returns a dataframe
loadeddataset = pd.read_csv(filename,header=None)
#loadeddataset.drop(range(100)) #= loadeddataset2.ix[1:]
# Remove friction measurements with lower then a quality of min_quality
#loadeddataset = loadeddataset[loadeddataset[3] >= min_quality]

# Dummy storage variable
cm_total = [[0,0],[0,0]]
err_final = 0.0
sensitivity_final = 0.0
selectivity_final = 0.0

#I start the cross validation
for train_indices, test_indices in k_fold.split(loadeddataset.values):
    print(" ------------NEW FOLD-------------------")
    cdataset = HelperDataSets(loadeddataset,train_indices,test_indices)

    # Variables to store the maximum accuracy, sensitivity and selectivity in
    max_accuracy = 0.0
    max_sensitivity = 0.0
    max_selectivity  = 0.0

    # Start the session
    with tf.Session() as sess:
        # For logging output (not used)
        #tensorboard --logdir=/path/to/your/log/file/folder
        #tf.InteractiveSession()

        # Run the instance
        sess.run(init)
        # For saving output (not used)
        #summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())


        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            
            # Number of batches
            total_batch = int(cdataset.train.num_examples/batch_size)
            
            # Loop over all batches
            for i in range(total_batch):

                # Fetch next batch
                batch_x, batch_y = cdataset.train.next_batch(batch_size)

                # Run optimization op (backprop) and cost op (to get loss value)
                m, c, summary = sess.run([optimizer, cost, merged_summary_op], \
                    feed_dict={x: batch_x, y: batch_y,keep_prob: 0.1})

                # write to log output (not used)
                #summary_writer.add_summary(summary, epoch * total_batch + i)

                # Compute average loss
                avg_cost += c / total_batch
                
            # Display logs per epoch step
            if epoch % display_step == 0:

                #accuracy_ = sess.run(accuracy, \
                #    feed_dict={x: cdataset.validation.explanatory, y: cdataset.validation.labels,keep_prob: 1})

                # For getting the predicted values
                y_p = tf.argmax(pred, 1)

                # Get the accuract and prediction from the trained network
                val_accuracy, y_pred = sess.run([accuracy,y_p], feed_dict={x: cdataset.validation.explanatory, y: cdataset.validation.labels,keep_prob: 1})

                # Used to get the true values from the target data
                y_true = np.argmax(cdataset.validation.labels.values,1)
                
                # Calculate the confusion matrix
                cm = sk.metrics.confusion_matrix(y_true,y_pred)

                # Calculate the sensitivity and selectivity
                sensitivity = cm[1][1]/(cm[1][1]+cm[1][0])
                selectivity = cm[0][0]/(cm[0][0]+cm[0][1])
        
                # Find and store the best results from the model
                if max_accuracy < val_accuracy:
                    max_accuracy = val_accuracy
                    max_sensitivity = sensitivity
                    max_selectivity = selectivity

                # Output results in console
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost), "acc=",\
                    "{:.4f}".format(val_accuracy), "min error rate=",\
                    "{:.4f}".format(1-max_accuracy), "sen=",\
                    "{:.4f}".format(max_sensitivity), "sel=",\
                    "{:.4f}".format(max_selectivity))


        print("Optimization Finished!")
        
        # For getting the predicted values
        y_p = tf.argmax(pred, 1)
        
        # Get the accuract and prediction from the trained network
        val_accuracy, y_pred = sess.run([accuracy,y_p], feed_dict={x: cdataset.validation.explanatory, y: cdataset.validation.labels,keep_prob: 1})

        # For getting the actual values
        y_true = np.argmax(cdataset.validation.labels.values,1)

        #print("confusion_matrix")
        #cm = sk.metrics.confusion_matrix(y_true,y_pred)

        # Store the confusion matrix
        cm_total += cm
        print(cm)
        # Store the error rate
        err_final += (1-max_accuracy)

        sensitivity_final += max_sensitivity
        selectivity_final += max_selectivity

        
        # Calculate the sensitivity and selectivity
        #sensitivity = cm[1][1]/(cm[1][1]+cm[1][0])
        #selectivity = cm[0][0]/(cm[0][0]+cm[0][1])

        # Print the results
        print("Best error rate:")
        print(1-max_accuracy)
        print("Best sensitivity:")
        print(max_sensitivity)
        print("Best selectivity:")
        print(max_selectivity)

        fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)

print("-----FINAL RESULTS-------------")
#print("Final confusion matrix:")
#print(cm_total)
print("Final error rate:")
print(err_final/kfolds)

#sensitivity = cm_total[1][1]/(cm_total[1][1]+cm_total[1][0])
#selectivity = cm_total[0][0]/(cm_total[0][0]+cm_total[0][1])
print("Final sensitivity:")
print(sensitivity_final/kfolds)
print("Final selectivity:")
print(selectivity_final/kfolds)

