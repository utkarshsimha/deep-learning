import lasagne
import theano
import theano.tensor as T
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
vocab_size = 0
data_size = 0
char_to_idx = {}
idx_to_char = {}
data = ""
class RNN:
    def __init__( self, n_in, n_hid, n_out, n_seq, batch_size, learning_rate=0.03, grad_clip=100):

        self.inpLayer = lasagne.layers.InputLayer( shape=( batch_size, n_seq, n_in ) )
        self.hidLayer1 = lasagne.layers.LSTMLayer( self.inpLayer, n_hid, grad_clipping=grad_clip, nonlinearity=lasagne.nonlinearities.tanh )
        self.hidLayer2 = lasagne.layers.LSTMLayer( self.hidLayer1, n_hid, grad_clipping=grad_clip, nonlinearity=lasagne.nonlinearities.tanh )
        self.hidSliced = lasagne.layers.SliceLayer( self.hidLayer2, -1, 1 )
        self.out = lasagne.layers.DenseLayer( self.hidSliced, n_out, W=lasagne.init.GlorotNormal(), nonlinearity=lasagne.nonlinearities.softmax )
        self.targ = T.ivector('targ')
        self.net_out = lasagne.layers.get_output( self.out )
        self.cost = lasagne.objectives.categorical_crossentropy( self.net_out, self.targ ).mean()
        self.params = lasagne.layers.get_all_params( self.out, trainable=True )
        self.updates = lasagne.updates.adagrad( self.cost, self.params, learning_rate )
        self.train = theano.function( [ self.inpLayer.input_var, self.targ ], self.cost, updates=self.updates, allow_input_downcast=True )
        self.pred = theano.function( [ self.inpLayer.input_var ], T.argmax( self.net_out, 1 ), allow_input_downcast=True )

def loadDataset():
    global vocab_size, char_to_idx, idx_to_char, data, data_size
    dataFile = open( "input.txt", "r" )
    inpText = dataFile.read()
    inpText = inpText.decode('utf-8', 'ignore')
    data = inpText
    data_size = len(data)
    vocab = list(set(data))
    vocab_size = len(vocab)
    print vocab_size
    print data_size
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

def sample( generation_phrase, getPred, n_seq ):
    assert(len(generation_phrase)>=n_seq)
    sample_ix = []
    x,_ = genData(len(generation_phrase)-n_seq, 1, n_seq, generation_phrase, False)
    for i in range(10):
        ix = getPred(x)[0]
        sample_ix.append(ix)
        x[:,0:n_seq-1,:] = x[:,1:,:]
        x[:,n_seq-1,:] = 0
        x[0,n_seq-1,sample_ix[-1]] = 1. 

    #random_snippet = generation_phrase + ''.join(idx_to_char[ix] for ix in sample_ix)    
    sample_ix = [ idx_to_char[ i ].decode( 'utf-8', 'ignore' ) for i in sample_ix ]
    print generation_phrase + " ".join( sample_ix )


if __name__ == '__main__':
    loadDataset()
    n_in = n_out = vocab_size
    n_hid = 100
    n_seq = 10
    batch_size = 128
    rnn = RNN( n_in, n_hid, n_out, n_seq, batch_size )
    n_epochs = 100
    ptr = 0
    for ep in range( n_epochs ):
        avg_cost = 0.
        n_batches = data_size/batch_size
        for mb_idx in range( n_batches ):
            inp,targ = genData( ptr, batch_size, n_seq, data )
            ptr += n_seq + batch_size + 1
            if( ptr + batch_size + n_seq >= vocab_size ):
                ptr = 0
            avg_cost += rnn.train( inp, targ )
        print "Epoch {} : Cost {}".format( ep, avg_cost/n_batches )
        gen_text = "Manchester United are the best team in the world"
        sample( gen_text, rnn.pred, n_seq )
