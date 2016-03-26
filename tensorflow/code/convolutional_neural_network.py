import tensorflow as tf
import numpy as np
import math
import cPickle as pickle

def load_data():
    ''' Load dataset '''
    pickle_file = 'notMNIST.pickle'
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
        train_X = dataset['train_dataset']
        train_Y = dataset['train_labels']
        valid_X = dataset['valid_dataset']
        valid_Y = dataset['valid_labels']
        test_X = dataset['test_dataset']
        test_Y = dataset['test_labels']
        train = reformat( train_X, train_Y )
        valid = reformat( valid_X, valid_Y )
        test = reformat( test_X, test_Y )
        print 'Training set', train[0].shape, train[1].shape
        print 'Validation set', valid[0].shape, valid[1].shape
        print 'Test set', test[0].shape, test[1].shape
        return train, test, valid 

def reformat( X, Y ):
    #Reshape X to be N X 784
    image_size = 28
    X = X.reshape((-1, image_size * image_size)).astype(np.float32)

    #Reshape Y to be a one hot vector
    Y = (np.arange(max(Y)+1) == Y[:,None]).astype(np.float32)
    return (X,Y)

class ConvNeuralNet:
    def __init__( self, valid, test, n_in, n_out, hidden_layers, activation=tf.nn.relu, batch_size=128, learning_rate=0.01 ):
        self.activation = activation 

        self.tf_train = ( tf.placeholder( tf.float32, shape=[ None, 784 ] ), tf.placeholder( tf.float32, shape=[ None, 10 ] ) )
        self.tf_valid = ( tf.cast( tf.constant( valid[0] ), tf.float32 ), tf.cast( tf.constant( valid[1] ), tf.float32 ) )
        self.tf_test = ( tf.cast( tf.constant( test[0] ), tf.float32 ), tf.cast( tf.constant( test[1] ), tf.float32 ) )
    
        ''' First layer '''
        hid1 = self._add_layer( self.tf_train[0], [ 5, 5, 1, 32 ], [ 32 ], [ -1, 28, 28, 1 ] )
        hid2 = self._add_layer( hid1, [ 5, 5, 32, 64 ], [ 64 ] )
        dense = self._add_layer( hid2, [ 7 * 7 * 64, 1024 ], [ 1024 ], [ -1, 7 * 7 * 64 ], dense=True )

        w_out = self._init_weight( [ 1024, 10 ] )
        b_out = self._init_bias( [ 10 ] )
        self.dropout = tf.placeholder( tf.float32 )
        train_logits = tf.matmul( tf.nn.dropout( dense, self.dropout ), w_out ) + b_out
        self.out = tf.nn.softmax( train_logits )
        self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( train_logits, self.tf_train[1] ) )
        self.optimizer = tf.train.AdamOptimizer( learning_rate ).minimize( self.cost )
        correct_pred = tf.equal( tf.argmax( self.out, 1 ), tf.argmax( self.tf_train[1], 1 ) )
        self.accuracy = tf.reduce_mean( tf.cast( correct_pred, tf.float32 ) )
        self.saver = tf.train.Saver()

    def _add_layer( self, inp, w_shape, b_shape, inp_trans=None, dense=False ):
        W_conv = self._init_weight( w_shape )
        B_conv = self._init_bias( b_shape )

        layer_inp = inp
        if inp_trans:
            layer_inp = tf.reshape( inp, inp_trans )

        if not dense:
            h_conv = self.activation( self._conv2d( layer_inp, W_conv ) + B_conv )
            h_pool = self._max_pool_2x2( h_conv )
            layer_out = h_pool
        else:
            layer_out = tf.nn.relu( tf.matmul( layer_inp, W_conv ) + B_conv )

        return layer_out


    def _init_weight( self, shape ):
        init_w = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable( init_w )

    def _init_bias( self, shape ):
        return tf.Variable( tf.zeros( shape ) )

    def _conv2d( self, inp, weight, strides=[ 1, 1, 1, 1 ], padding='SAME' ):
        return tf.nn.conv2d( inp, weight, strides=strides, padding=padding )

    def _max_pool_2x2( self, inp, ksize=[ 1, 2, 2, 1 ], strides=[ 1, 2, 2, 1 ], padding='SAME' ):
        return tf.nn.max_pool( inp, ksize=ksize, strides=strides, padding=padding )


if __name__ == '__main__':
    train, test, valid = load_data()
    train_X = train[0][:20000]
    train_Y = train[1][:20000]
    valid = ( valid[0][:5000], valid[1][:5000] )

    n_epochs = 10
    batch_size = 128
    n_batches = train_X.shape[0] / batch_size

    cnn = ConvNeuralNet( valid, test, None, None, None )
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for ep in range( n_epochs ):
            avg_cost = 0.
            avg_acc = 0.
            for mb in range( n_batches ):
                offset = mb
                batch_X = train_X[ offset*batch_size:(offset + 1)*batch_size ]
                batch_Y = train_Y[ offset*batch_size:(offset + 1)*batch_size ]

                feed_dict = { cnn.tf_train[0]: batch_X, cnn.tf_train[1]: batch_Y, cnn.dropout: 0.5 }

                _, cost, acc = sess.run( [ cnn.optimizer, cnn.cost, cnn.accuracy ], feed_dict=feed_dict )
                avg_cost += cost
                avg_acc += acc
            if( True ):#ep % 10 == 0 ):
                print "Cost at {} - {}".format( ep, avg_cost/n_batches )
                print "Training accuracy : {}".format( avg_acc/n_batches )
                print "Validation accuracy : {}".format( cnn.accuracy.eval(
                    feed_dict={ cnn.tf_train[0]:valid[0], cnn.tf_train[1]:valid[1], cnn.dropout: 1.0 }
                ) )

        print "Testing accuracy : {}".format( cnn.accuracy.eval(
            feed_dict={ cnn.tf_train[0]:test[0], cnn.tf_train[1]:test[1], cnn.dropout: 1.0 }
        ) )
        cnn.saver.save(sess, 'model.ckpt')
