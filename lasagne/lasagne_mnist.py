import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.nonlinearities import *
import gzip
import cPickle


def build_dnn( n_in, n_out, hiddenLayers, inp, idf ):
    dnn = lasagne.layers.InputLayer( shape=( None, n_in ), input_var=inp )
    if( idf ):
        dnn = lasagne.layers.DropoutLayer( dnn, p=idf )

    #for h_in,h_out in zip( shapes[:-1], shapes[1:] ):
    for h_unit,hdf in hiddenLayers:
        dnn = lasagne.layers.DenseLayer( dnn, num_units=h_unit, nonlinearity=sigmoid, W=lasagne.init.GlorotUniform() )
        if( hdf ):
            dnn = lasagne.layers.DropoutLayer( dnn, p=hdf )

    dnn = lasagne.layers.DenseLayer( dnn, num_units=n_out, nonlinearity=softmax )

    return dnn

    
def loadDataset():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    print "Shapes of input :"
    print "Training:   ",train_set[0].shape, train_set[1].shape
    print "Test:       ",test_set[0].shape, test_set[1].shape
    return train_set, test_set

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def createOneHotVec(labels, max_index):
    ''' Given a vector, convert it to a vector where the values are represented with 1 in the argument of the value
    and 0 otherwise. Eg : 5 will be represented as [ 0 0 0 0 0 1 0 0 0 0 0 ] '''
    vec = np.zeros((labels.shape[0], max_index + 1))
    for i in xrange(labels.shape[0]):
        vec[i, labels[i]] = 1
    return vec

def main():
    train, test = loadDataset()

    n_out = 10
    n_epochs = 100
    n_hid = 50
    batch_size = 200
    inp = T.dmatrix('inp')
    targ = T.dmatrix('targ')
    hiddenLayers = [ ( n_hid, 0.2 ), ( n_hid, 0.1 ) ] 
    idf = 0.2
    dnn = build_dnn( train[0].shape[1], n_out, hiddenLayers, inp, idf  )
    out = lasagne.layers.get_output( dnn )
    loss = lasagne.objectives.categorical_crossentropy( out, targ )
    loss = loss.mean()
    #lasagne.regularization
    params = lasagne.layers.get_all_params( dnn, trainable=True )
    updates = lasagne.updates.adagrad( loss, params, learning_rate=0.01 )

    test_out = lasagne.layers.get_output( dnn, deterministic=True )
    test_loss = lasagne.objectives.categorical_crossentropy( test_out, targ )
    test_loss = test_loss.mean()
    test_acc = T.mean( T.eq( T.argmax( test_out, 1 ), T.argmax( targ, 1 ) ) )

    train_fn = theano.function( [ inp, targ ], loss, updates=updates )

    test_fn = theano.function( [ inp, targ ], [ test_loss, test_acc ], updates=updates )

    for ep in range( n_epochs ):
        cost = 0.
        n_batches = 0
        for batch_inp,batch_targ in iterate_minibatches( train[0], createOneHotVec( train[1], 9 ), batch_size, shuffle=True ):
            cost += train_fn( batch_inp, batch_targ )
            n_batches += 1
        print "Epoch {} : {}".format( ep, cost/n_batches )

    cost = 0.
    n_batches = 0
    test_acc = 0.
    for batch_inp, batch_targ in iterate_minibatches( test[0], createOneHotVec( test[1], 9 ), batch_size, shuffle=False ):
        loss, acc = test_fn( batch_inp, batch_targ )
        cost += loss
        test_acc += acc
        n_batches += 1

    print "Test accuracy : {}%".format( (test_acc/n_batches)*100 )

if __name__ == '__main__':
    main()
