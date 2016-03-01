''' Implementing a Deep Neural Network for classifying hand-written digits
Trained and tested on the MNIST dataset.

@author : Utkarsh Simha (utkarshsimha@gmail.com) '''


import tensorflow as tf
import numpy as np
import cPickle as pickle
import math
import os
import gzip

def load_data():
    dataset = "mnist.pkl.gz"
    if ( not os.path.isfile(dataset) ):
        print "Download from {}".format("http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz")
        return None
    else:
        print "Loading data..."

        # Load the dataset
        with gzip.open(dataset, 'rb') as f:
            try:
                train, valid, test = pickle.load(f, encoding='latin1')
            except:
                train, valid, test = pickle.load(f)
                
        ''' Reformat '''
        train = ( train[0], reformat( train[1] ) )
        test = ( test[0], reformat( test[1] ) )
        valid = ( valid[0], reformat( valid[1] ) )
        
        print 'Training set', train[0].shape, train[1].shape
        print 'Validation set', valid[0].shape, valid[1].shape
        print 'Test set', test[0].shape, test[1].shape
        return train, test, valid
    
def reformat( vec ):
    ''' Convert vector into a one-hot vector '''
    return ( np.arange( max( vec ) + 1 ) == vec[ :,None ] ).astype( np.float32 )


class DeepNeuralNetwork:
    ''' Deep Neural Network Implementation '''
    def __init__( self, n_in, n_out, test, valid, hidden_layers, activation=tf.nn.sigmoid, batch_size=128, learning_rate=0.01 ):
        '''
            || DESCRIPTION ||
            Create graph for  Deep Neural Network

            || PARAMS ||
            * n_in :
                type - int
                brief - The number of neurons for the input layer
            * n_out :
                type - int
                brief - The number of neurons in the output layer
            * test :
                type - tuple
                brief - Tuple of numpy arrays representing the input and target values for testing
            * valid :
                type - tuple
                brief - Tuple of numpy arrays representing the input and target values for validation
            * hidden_layers :
                type - list
                brief - List of tuples - where each tuple represent the number of hidden neurons for
                the hidden layer and its corresponding dropout factor
            * activation :
                type - function
                brief - Activation function such as - tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu
            * batch_size :
                type - int
                brief - Mini batch size for training input and target
            * learning_rate :
                type - float
                brief - Learning rate for training
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():

            ''' Training dataset, given in mini-batches '''
            self.tf_train = ( tf.placeholder( tf.float32, shape=(batch_size, n_in ) ), tf.placeholder( tf.float32, shape=(batch_size, n_out ) ) )

            ''' Validation dataset '''
            tf_valid = ( tf.cast( tf.constant( valid[0] ), tf.float32 ), tf.cast( tf.constant( valid[1] ), tf.float32 ) )

            ''' Testing dataset '''
            tf_test = ( tf.cast( tf.constant( test[0] ), tf.float32 ), tf.cast( tf.constant( test[1] ), tf.float32 ) )

            ''' Model '''
            self.weights = [] #Weights list
            self.bias = [] #Bias list

            ''' L2 Regularization to avoid overfitting '''
            self.l2_reg = 0.

            '''Inputs'''
            train_input = self.tf_train[0]
            valid_input = tf_valid[0]
            test_input = tf_test[0]

            layer_in = n_in #number of incoming connections to the layer
            ''' Add hidden layers '''
            for layer_out, hdf in hidden_layers:
                train_input = self._add_layer( train_input, layer_in, layer_out, activation=activation, dropout=hdf, l2_reg=True )
                valid_input = self._add_layer( valid_input, layer_in, layer_out, activation=activation, weights=self.weights[-1], bias=self.bias[-1] )
                test_input = self._add_layer( test_input, layer_in, layer_out, activation=activation, weights=self.weights[-1], bias=self.bias[-1] )
                ''' Number of input connections to next layer is the number of output connections of the current layer '''
                layer_in = layer_out
                
            ''' Output layers '''
            train_logits = self._add_layer( train_input, layer_in, n_out )
            valid_logits = self._add_layer( valid_input, layer_in, n_out, weights=self.weights[-1], bias=self.bias[-1] )
            test_logits = self._add_layer( test_input, layer_in, n_out, weights=self.weights[-1], bias=self.bias[-1] )

            ''' Cross-Entropy Cost function '''
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, self.tf_train[1])) + 0.0001 * self.l2_reg

            ''' Adagrad '''
            self.optimizer = tf.train.AdagradOptimizer( learning_rate ).minimize( self.cost )

            ''' Prediction functions '''
            self.train_pred = tf.nn.softmax( train_logits )
            self.valid_pred = tf.nn.softmax( valid_logits )
            self.test_pred = tf.nn.softmax( test_logits )

    def _add_layer( self, input, n_in, n_out, activation=None, weights=None, bias=None, dropout=None, l2_reg=False ):
        if( weights is None ):
            ''' Xavier init '''
            init_range = math.sqrt(6.0 / (n_in + n_out))
            init_w = tf.random_uniform( [n_in,n_out], -init_range, init_range)
            weights = tf.cast( tf.Variable( init_w ), tf.float32 )
            self.weights.append( weights )

        if( bias is None ):
            bias = tf.cast( tf.Variable( tf.zeros( [ n_out ] ) ), tf.float32 )
            self.bias.append( bias )

        if( l2_reg ):
            ''' L2 regularization '''
            l2_reg = tf.nn.l2_loss( weights )
            self.l2_reg += l2_reg

        layer = tf.matmul( input, weights ) + bias
        if( activation is not None ):
            layer = activation( layer )

        if( dropout is not None ):
            ''' Dropout + scaling '''
            layer = tf.nn.dropout( layer, 1-dropout ) * 1/( 1- dropout )

        return layer

def accuracy( pred, labels ):
    '''
        || DESCRIPTION ||
        Compute accuracy

        || PARAMS ||
        * pred :
            type - numpy.ndarray
            brief - The prediction of the nueral network (assumed to be one hot vectors)
        * labels :
            type - numpy.ndarray
            brief - The actual target values
    '''
    return ( 100.0 * np.sum( np.argmax( pred, 1 ) == np.argmax( labels, 1 ) ) / pred.shape[0] )

if __name__ == '__main__':
    ''' Dataset '''
    train,valid,test = load_data()
    train_X = train[0]
    train_Y = train[1]

    ''' Params '''
    n_epochs = 5000 #Number of epochs
    batch_size = 128 #Batch size
    learning_rate = 0.01 #Learning rate
    hidden_layers = [ ( 1024, 0.5 ), ( 1024, 0.5 ) ] #Number of hidden neurons and corresponding dropout factor
    n_in = train[0].shape[1] #Number of input neurons
    n_out = train[1].shape[1] #Number of ouptut neurons - number of classes

    ''' Model '''
    dnn = DeepNeuralNetwork( n_in, n_out, test, valid, hidden_layers, tf.nn.relu, batch_size, learning_rate )

    with tf.Session( graph = dnn.graph ) as session:
        ''' Initialize TensorFlow variables '''
        tf.initialize_all_variables().run()
        for ep in range( n_epochs ):
            ''' Mini-batching '''
            offset = (ep * batch_size) % (train_Y.shape[0] - batch_size)
            batch_X = train_X[ offset:(offset + batch_size) ]
            batch_Y = train_Y[ offset:(offset + batch_size) ]

            ''' Input to placeholders '''
            feed_dict = { dnn.tf_train[0]:batch_X, dnn.tf_train[1]:batch_Y }

            ''' Train step '''
            _, cost, train_pred = session.run( [ dnn.optimizer, dnn.cost, dnn.train_pred ], feed_dict=feed_dict )

            if( ep % 100 == 0 ):
                print "Cost at {} - {}".format( ep, cost )
                print "Training accuracy : {}".format( accuracy( train_pred, batch_Y ) )
                print "Validation accuracy : {}".format( accuracy( dnn.valid_pred.eval(), valid[1] ) )

        ''' Testing '''
        print "Test accuracy : {}".format( accuracy( dnn.test_pred.eval(), test[1] ) )
