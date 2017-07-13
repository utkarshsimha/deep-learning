import numpy as np
import tensorflow as tf
import cPickle as pickle
import math
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn


def loadDataset():
    ''' Load dataset '''
    global vocab_size, char_to_idx, idx_to_char, data, data_size
    dataFile = open( "input.txt", "r" )
    inpText = dataFile.read()
    inpText = inpText.decode('utf-8', 'ignore')
    data = inpText
    data_size = len(data)
    vocab = list(set(data))
    vocab_size = len(vocab)
    print "Vocab size : {}".format( vocab_size )
    char_to_idx = { w:i for i,w in enumerate(vocab) }
    idx_to_char = { i:w for i,w in enumerate(vocab) }

def genData( ptr, batch_size, n_seq, data, ret_targ=True ):
    inp = np.zeros( ( batch_size, n_seq, vocab_size ) )
    targ = np.zeros( ( batch_size, vocab_size ) )
    for batch_idx in range( batch_size ):
        idx = batch_idx
        for step in range( n_seq ):
            inp[ batch_idx, step, char_to_idx[ data[ ptr+idx+step ] ] ] = 1
        if( ret_targ ):
            char_idx = char_to_idx[ data[ ptr+idx+n_seq ] ]
            targ[ batch_idx ] = [ 1 if i == char_idx else 0 for i in range(vocab_size) ]
    return inp, targ

def sample( generation_phrase, session, getPred, inp, n_seq ):
    assert( len( generation_phrase ) >= n_seq )
    sample_ix = []
    x,_ = genData( len( generation_phrase )-n_seq, 1, n_seq, generation_phrase, False )
    for i in range( 50 ):
        ix = session.run( [ getPred ], feed_dict={ inp:x } )[0][0]
        sample_ix.append( ix )
        x[:,0:n_seq-1,:] = x[:,1:,:]
        x[:,n_seq-1,:] = 0
        x[0,n_seq-1,sample_ix[-1]] = 1.

    with open("rnn_out3.txt","w") as f:
        sample_ix = [ idx_to_char[ i ].decode( 'utf-8', 'ignore' ) for i in sample_ix ]
        f.write( generation_phrase + "|" + "".join( sample_ix ) )
        f.write("\n-----------------------\n")

class RecurrentNeuralNetwork:
    def __init__( self, vocab_size, seq_len, hiddenLayer, activation=tf.nn.sigmoid, batch_size=128, learning_rate=0.01, num_layers=2 ):
        ''' Recurrent Neural Network implementation '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            n_in = n_out = vocab_size

            '''Training input, given in mini-batches '''
            self.tf_train = ( tf.placeholder( tf.float32, shape=(None, seq_len, n_in ) ), tf.placeholder( tf.float32, shape=(None, n_out ) ) )

            self.weights = []
            self.bias = []
            self.l2_reg = 0.
            train_input = self.tf_train[0]
            n_hid, hdf = hiddenLayer

            ''' Add LSTM layers '''
            hidden_layer = self._addRNNLayer( train_input, n_in, seq_len, n_hid, activation=activation,\
            dropout=hdf, l2_reg=True, num_layers=num_layers )

            ''' LSTM layer for sampling - no dropout '''
            sample_layer = self._addRNNLayer( train_input, n_in, seq_len, n_hid, activation=activation, num_layers=num_layers, \
            weights=self.weights[-1], bias=self.bias[-1] )

            '''Output layer '''
            train_logits = self._addOutLayer( hidden_layer, n_hid, seq_len, n_out, l2_reg=True )

            '''Output layer for sampling'''
            sample_logits = self._addOutLayer( sample_layer, n_hid, seq_len, n_out, weights=self.weights[-1], bias=self.bias[-1] )


            '''Cost function - Softmax Cross Entropy and L2 Regularization'''
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, self.tf_train[1])) + 0.0001 * self.l2_reg

            ''' Clip gradients '''
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm( tf.gradients( self.cost, tvars ), 100 )

            ''' Stochastic Gradient Descent '''
            self.optimizer = tf.train.GradientDescentOptimizer( learning_rate ).apply_gradients( zip( grads, tvars ) )

            ''' Predicting the next character - Sampling '''
            self.pred = tf.argmax( tf.nn.softmax( sample_logits ), 1 )

            self.saver = tf.train.Saver()


    def _addRNNLayer( self, input, n_in, n_seq, n_out, activation=None, weights=None, bias=None, dropout=None, l2_reg=False, num_layers=2 ):
        if( weights is None ):
            ''' Xavier init '''
            init_range = math.sqrt(6.0 / (n_in + n_out))
            init_w = tf.random_uniform( [n_in,n_out], -init_range, init_range)
            weights = tf.Variable( init_w )
            self.weights.append( weights )

        if( bias is None ):
            bias = tf.Variable( tf.zeros( [ n_out ] ) )
            self.bias.append( bias )

        ''' L2 regularization '''
        if( l2_reg ):
            l2_reg = tf.nn.l2_loss( weights )
            self.l2_reg += l2_reg 

        ''' Input reshaping - ( batch_size, seq_len, n_in ) -> ( batch_size * seq_len, n_in ) '''
        input = tf.transpose(input, [1, 0, 2])
        input = tf.reshape(input, [-1, n_in])


        ''' Weighted sum '''
        input = tf.matmul( input, weights ) + bias

        if( activation is not None ):
            input = activation( input )

        with tf.variable_scope("RNN"):
            ''' LSTM Cell '''
            lstmCell = rnn_cell.BasicLSTMCell( n_out )
            if( dropout is not None ):
                ''' Dropout '''
                lstmCell = rnn_cell.DropoutWrapper( lstmCell, output_keep_prob=dropout )
            else:
                tf.get_variable_scope().reuse_variables()

            ''' Multiple LSTM layers '''
            cell = rnn_cell.MultiRNNCell( [lstmCell] * num_layers )

            input = tf.split(0, n_seq, input)
            ''' Build RNN structure '''
            outputs, states = rnn.rnn( cell, input, dtype=tf.float32 )

        layer = outputs[-1]
        return layer

    def _addOutLayer( self, input, n_in, n_seq, n_out, weights=None, bias=None, l2_reg=False ):
        if( weights is None ):
            ''' Xavier Init '''
            init_range = math.sqrt(6.0 / (n_in + n_out))
            init_w = tf.random_uniform( [n_in,n_out], -init_range, init_range)
            weights = tf.Variable( init_w )
            self.weights.append( weights )

        if( bias is None ):
            bias = tf.Variable( tf.zeros( [ n_out ] ) )
            self.bias.append( bias )

        ''' L2 regularization '''
        if( l2_reg is not False ):
            l2_reg = tf.nn.l2_loss( weights )
            self.l2_reg += l2_reg 

        logits = tf.matmul( input, weights ) + bias
        return logits

if __name__ == '__main__':
    ''' Dataset '''
    loadDataset()

    ''' Params '''
    n_epochs = 1000
    batch_size = 128
    hiddenLayer = ( 512, 0.5 )
    n_seq = 8
    num_layers = 1
    n_batches = data_size/(batch_size*n_seq) 

    ''' Model '''
    dnn = RecurrentNeuralNetwork( vocab_size, n_seq, hiddenLayer, batch_size=batch_size, num_layers=num_layers )
    with tf.Session( graph = dnn.graph ) as session:
        ''' Variable Init '''
        tf.initialize_all_variables().run()

        ptr = 0
        sample_txt = ""
        for ep in range( n_epochs ):
            avg_cost = 0.

            ''' Mini-batching '''
            for mb in range( n_batches ):
                ''' Get next mini-batch '''
                inp,targ = genData( ptr, batch_size, n_seq, data )

                ptr += n_seq + batch_size + 1 #Advance pointer
                if( ptr + batch_size + n_seq >= vocab_size ):
                    ptr = 0 #Reset pointer

                ''' Input to placeholders '''
                feed_dict = { dnn.tf_train[0]:inp, dnn.tf_train[1]:targ }

                ''' Train step '''
                _, cost = session.run( [ dnn.optimizer, dnn.cost ], feed_dict=feed_dict )
                avg_cost += cost

            if( ep % 10 == 0 ):
                ''' Sampling '''
                rand_n = np.random.randint( 0, len(data)-n_seq-10 )
                gen_text = data[rand_n:rand_n+n_seq]
                sample( gen_text, session, dnn.pred, dnn.tf_train[0], n_seq )

        ''' Save model '''
        dnn.saver.save(session, 'model.ckpt', global_step=ep)
