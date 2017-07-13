import tensorflow as tf
import numpy as np
import math
import cPickle as pickle
import imdb

def load_data( maxlen=3000 ):
    ''' Load dataset '''
    train, valid, test = imdb.load_data()
    tr_inp, _, tr_targ = imdb.prepare_data( train[0], train[1], maxlen=maxlen )
    te_inp, _, te_targ = imdb.prepare_data( test[0], test[1], maxlen=maxlen )
    v_inp, _, v_targ = imdb.prepare_data( valid[0], valid[1], maxlen=maxlen )
    train = shuffle( np.transpose( tr_inp ), reformat( np.asarray( tr_targ ), 2 ) )
    test = shuffle( np.transpose( te_inp ), reformat( np.asarray( te_targ ), 2 ) )
    valid = shuffle( np.transpose( v_inp ), reformat( np.asarray( v_targ ), 2 ) )
    print "Train shape : {}, {}".format( train[0].shape, train[1].shape )
    print "Test shape : {}, {}".format( test[0].shape, test[1].shape )
    print "Valid shape : {}, {}".format( valid[0].shape, valid[1].shape )
    imdb_dict = pickle.load( open('imdb.dict.pkl','rb') )
    return train, test, valid, imdb_dict

def reformat( vec, num_classes ):
    ''' Create one hot vector '''
    return ( np.arange( num_classes ) == vec[ :,None ] ).astype( np.float32 )

def shuffle( X, Y ):
    x = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    shuffle_inp = x[:, :X.size//len(X)].reshape(X.shape)
    shuffle_targ = x[:, X.size//len(X):].reshape(Y.shape)
    np.random.shuffle( x )
    np.random.shuffle( x )
    return shuffle_inp, shuffle_targ

class ConvNeuralNet:
    def __init__( self, test, valid, n_in, n_out, vocab_size, embedding_size, filter_sizes, num_filters, batch_size=128, learning_rate=1e-4 ):

        self.learning_rate = learning_rate
        self.n_in = n_in
        self.n_out = n_out
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.tf_train = ( tf.placeholder( tf.int32, shape=[ batch_size, n_in ], name="train_inp" ),\
                        tf.placeholder( tf.float32, shape=[ batch_size, n_out ], name="train_targ" ) )
        self.tf_valid = ( tf.cast( tf.constant( valid[0] ), tf.int32 ), tf.cast( tf.constant( valid[1] ), tf.float32 ) )
        self.tf_test = ( tf.cast( tf.constant( test[0] ), tf.int32 ), tf.cast( tf.constant( test[1] ), tf.float32 ) )
        self.dropout = tf.placeholder( tf.float32, name="dropout" )

        self.cp_weights = []
        self.cp_bias = []
        self.W = None
        self.B = None
        self.train = self.addPhase( "train", self.tf_train[0], self.tf_train[1] )
        self.test = self.addPhase( "test", self.tf_test[0], self.tf_test[1] )
        self.valid = self.addPhase( "valid", self.tf_valid[0], self.tf_valid[1] )

    def addPhase( self, phase, input, labels ):
        with tf.device('/cpu:0'), tf.name_scope( "embedding-{}".format( phase ) ):
            W = tf.Variable( tf.random_uniform( [ self.vocab_size, self.embedding_size ], -1.0, 1.0 ) )
            embedded_chars = tf.nn.embedding_lookup( W, input )
            embedded_chars_expanded = tf.expand_dims( embedded_chars, -1 )

        pooled_outputs = []
        for i, filter_size in enumerate( self.filter_sizes ):
            with tf.name_scope( "cp-layer-{}-{}".format( i, phase ) ):
                filter_shape = [ filter_size, self.embedding_size, 1, self.num_filters ]
                if( phase is "train" ):
                    W = tf.Variable( tf.truncated_normal( filter_shape, stddev=0.1, name="W" ) )
                    self.cp_weights.append( W )
                    B = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="B" )
                    self.cp_bias.append( B )
                else:
                    W = self.cp_weights[i]
                    B = self.cp_bias[i]
                conv = tf.nn.conv2d( embedded_chars_expanded,\
                                     W,\
                                     strides=[1,1,1,1],\
                                     padding="VALID",\
                                     name="conv")
                hid = tf.nn.relu( tf.nn.bias_add(conv, B), name="relu" )
                max_pool = tf.nn.max_pool( hid,\
                                           ksize=[ 1, self.n_in-filter_size+1, 1, 1 ],\
                                           strides=[1,1,1,1],\
                                           padding="VALID",\
                                           name="max_pool" )
                pooled_outputs.append( max_pool )

        tot_filters = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat( 3, pooled_outputs )
        h_pool_flat = tf.reshape( h_pool, [ -1, tot_filters ] )

        with tf.name_scope("dropout-{}".format( phase )):
            if( phase is "train" ):
                h_drop = tf.nn.dropout( h_pool_flat, self.dropout )
            else:
                h_drop = tf.nn.dropout( h_pool_flat, 1.0 )

        with tf.name_scope("output-{}".format( phase )):
            if( phase is "train" ):
                self.W = tf.Variable( tf.truncated_normal( [ tot_filters, self.n_out ], stddev=0.1, name="W" ) )
                self.B = tf.Variable(tf.constant(0.1, shape=[self.n_out]), name="B" )
            out = tf.nn.xw_plus_b( h_drop, self.W, self.B, name="out" )
            pred = tf.argmax( out, 1, name="pred" )

        with tf.name_scope("cost-{}".format( phase )):
            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( out, labels ) )

        with tf.name_scope("accuracy-{}".format( phase )):
            correct_pred = tf.equal( pred, tf.argmax( input, 1 ) )
            acc = tf.reduce_mean( tf.cast( correct_pred, tf.float32) )

        with tf.name_scope("updates-{}".format( phase )):
            optimizer = tf.train.AdamOptimizer( self.learning_rate ).minimize( cost )
        return { 'cost':cost, 'optimizer':optimizer, 'acc':acc }

if __name__ == '__main__':
    train, test, valid, imdb_dict = load_data()
    train_X = train[0]
    train_Y = train[1]

    #n_epochs = 1000
    n_epochs = 1
    batch_size = 128
    dropout = 0.5
    n_in = train_X.shape[1]
    n_out = train_Y.shape[1]
    vocab_size = len(imdb_dict.keys())
    embedding_size = 100
    filter_sizes = [ 2,3,4 ]
    num_filters = 2
    #n_batches = train_X.shape[1]/batch_size + 1
    n_batches = 1

    cnn = ConvNeuralNet( test, valid, n_in, n_out, vocab_size, embedding_size, filter_sizes, num_filters, batch_size=batch_size )
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        
        for ep in range( n_epochs ):

            avg_cost = 0.
            avg_acc = 0.
            for mb in range( n_batches ):
                batch_X = train_X[ mb*batch_size:(mb + 1)*batch_size ]
                batch_Y = train_Y[ mb*batch_size:(mb + 1)*batch_size ]

                feed_dict = { cnn.tf_train[0]: batch_X, cnn.tf_train[1]: batch_Y, cnn.dropout: 0.5 }

                _, cost, acc  = sess.run( [ cnn.train['optimizer'], cnn.train['cost'], cnn.train['acc'] ], feed_dict=feed_dict )
                avg_cost += cost
                avg_acc += acc 

            if( ep % 10 == 0 ):
                print "Cost at {} - {}".format( ep, avg_cost/n_batches )
                print "Training accuracy : {}".format( avg_acc/n_batches )
                print "Validation accuracy : {}".format( cnn.valid['acc'].eval() )

        '''print "Testing accuracy : {}".format( cnn.acc.eval(
            feed_dict={ cnn.tf_train[0]:test[0], cnn.tf_train[1]:test[1], cnn.dropout: 1.0 }
        ) )'''
