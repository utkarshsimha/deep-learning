import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gzip
import cPickle as pickle

def generateData( shape ):
    return np.random.uniform(-1., 1., size=shape)

def loadData():
    with gzip.open( 'mnist.pkl.gz' ) as f_mnist:
        train, test, valid = pickle.load( f_mnist )

    return train[0][:5000]

class GAN:
    def __init__( self, batch_size, n_feats, gen_n_feats, gen_n_hid, disc_n_hid ):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.n_feats = n_feats
            self.scopes = {}
            with tf.variable_scope("z"):
                self.z = tf.placeholder( tf.float32, shape=[ batch_size, gen_n_feats ], name="z" )
                tf.histogram_summary( "z", self.z )
            with tf.variable_scope("x"):
                self.x = tf.placeholder( tf.float32, shape=[ batch_size, n_feats ], name="x" )
                tf.histogram_summary( "x", self.x )
            with tf.variable_scope( "eta" ):
                self.eta = tf.placeholder( tf.float32, shape=[], name="eta" )
            with tf.variable_scope("G") as self.gen_scope:
                ''' Minimize - log( D_2( G( z ) ) ) '''
                self.scopes[ 'gen_scope' ] = self.gen_scope
                self.G = self.Generator( batch_size, n_feats, gen_n_feats, gen_n_hid, self.scopes ) 
            with tf.variable_scope("D") as self.disc_scope:
                ''' Minimize - log( D_1(x) ) - log( 1 - D_2( G( z ) ) )  '''
                self.scopes[ 'disc_scope' ] = self.disc_scope
                self.D_real = self.Discriminator( batch_size, n_feats, disc_n_hid, self.scopes )
                tf.get_variable_scope().reuse_variables()
                self.D_gen = self.Discriminator( batch_size, n_feats, disc_n_hid, self.scopes, gen_inp=self.G.gen_x )

            #self.disc_loss = - tf.reduce_mean( tf.log( self.D_real.out ) + tf.log( 1. - self.D_gen.out ) )
            self.disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( self.D_real.logits, tf.ones_like( self.D_gen.logits ) ) ) + \
                    tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( self.D_gen.logits, tf.zeros_like( self.D_gen.logits ) ) )

            self.gen_loss = - tf.reduce_mean( tf.log( self.D_gen.out ) )
            self.gen_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( self.D_gen.logits, tf.ones_like( self.D_gen.logits ) ) )
            tf.scalar_summary( "disc_loss", self.disc_loss )
            tf.scalar_summary( "gen_loss", self.gen_loss )

            self.optimizers = self.optimize()

            self.summaries = tf.merge_all_summaries()

    class Generator:
        def __init__( self, batch_size, n_feats, gen_n_feats, n_hid, scopes ):
            self.z = tf.get_variable( "z", [ batch_size, gen_n_feats ] )
            with tf.variable_scope( "theta/weights", initializer=tf.random_normal_initializer() ):
                self.W_1 = tf.get_variable( "W_1", [ gen_n_feats, n_hid ] )
                self.W_2 = tf.get_variable( "W_2", [ n_hid, n_hid ] )
                self.W_3 = tf.get_variable( "W_3", [ n_hid, n_feats ] )
            with tf.variable_scope( "theta/bias", initializer=tf.constant_initializer(0.0) ):
                self.B_1 = tf.get_variable( "B_1", [ n_hid ] )
                self.B_2 = tf.get_variable( "B_2", [ n_hid ] )
                self.B_3 = tf.get_variable( "B_3", [ n_feats ] )
            self.hid_1 = tf.nn.relu( tf.matmul( self.z, self.W_1 ) + self.B_1 )
            self.hid_2 = tf.nn.sigmoid( tf.matmul( self.hid_1, self.W_2 ) + self.B_2 )
            with tf.variable_scope( "gen_x" ):
                self.gen_x = tf.matmul( self.hid_2, self.W_3 ) + self.B_3
                tf.histogram_summary( "gen_x", self.gen_x )

    class Discriminator:
        def __init__( self, batch_size, n_feats, n_hid, scopes, gen_inp=None ):
            if( gen_inp is not None ):
                self.x = gen_inp
            else:
                self.x = tf.get_variable( "x", shape=[ batch_size, n_feats ] )

            with tf.variable_scope( "theta/weights", initializer=tf.random_normal_initializer() ):
                self.W_1 = tf.get_variable( "W_1", [ n_feats, n_hid ] )
                self.W_2 = tf.get_variable( "W_2", [ n_hid, n_hid ] )
                self.W_3 = tf.get_variable( "W_3", [ n_hid, 1 ] )
            with tf.variable_scope( "theta/bias", initializer=tf.constant_initializer(0.0) ):
                self.B_1 = tf.get_variable( "B_1", [ n_hid ] )
                self.B_2 = tf.get_variable( "B_2", [ n_hid ] )
                self.B_3 = tf.get_variable( "B_3", [ 1 ] )

            self.hid1 = tf.nn.relu( tf.matmul( self.x, self.W_1 ) + self.B_1 )
            self.hid2 = tf.nn.relu( tf.matmul( self.hid1, self.W_2 ) + self.B_2 )
            self.logits = tf.matmul( self.hid2, self.W_3, ) + self.B_3
            self.out = tf.nn.sigmoid( self.logits )

    def optimize( self ):
        vars = tf.trainable_variables()
        disc_theta = [v for v in vars if v.name.startswith('D/theta')]
        gen_theta = [v for v in vars if v.name.startswith('G/theta')]
        disc_optimizer = tf.train.AdamOptimizer().minimize( self.disc_loss, var_list=disc_theta )
        gen_optimizer = tf.train.AdamOptimizer().minimize( self.gen_loss, var_list=gen_theta )
        return disc_optimizer, gen_optimizer

if __name__ == '__main__':
    mu = -1
    sigma = 1
    eta = 0.01
    batch_size=128
    x = loadData()
    N = x.shape[0]
    z = generateData( ( N, 1 ) )
    gan = GAN( batch_size, x.shape[1], z.shape[1], 128, 128 )
    with tf.Session( graph = gan.graph ) as sess:
        summary_writer = tf.train.SummaryWriter("/tmp/tf_logs/gan", graph=gan.graph)
        tf.initialize_all_variables().run()
        loss = []
        for ep in range( 1, 1000000 ):
            offset = (ep * batch_size) % (N - batch_size)
            batch_x = x[ offset:(offset + batch_size) ]
            batch_z = z[ offset:(offset + batch_size) ]
            feed_dict = { gan.x: batch_x, gan.z: batch_z, gan.eta: eta }
            _, disc_loss, gen_loss, disc_out, disc_gen_out, gen_out = sess.run( [ gan.optimizers, gan.disc_loss, gan.gen_loss, gan.D_gen.out,
                gan.D_real.out, gan.G.gen_x ], feed_dict=feed_dict )
            #print np.mean( disc_out ), ";", np.mean( disc_gen_out )
            summaries = sess.run( gan.summaries, feed_dict=feed_dict )
            loss.append( ( disc_loss, gen_loss ) )
            summary_writer.add_summary( summaries, ep )
            if( ep % 1000 == 0 ):
                print np.mean( disc_out ), "; ", np.mean( disc_gen_out )
                #plt.imshow( np.mean( gen_out, 0 ).reshape( ( 28, 28 ) ) )
                #plt.show()
                
        #print disc_out
        disc_loss, gen_loss = zip( *loss )
        plt.plot( range( len( disc_loss ) ), disc_loss )
        plt.plot( range( len( gen_loss ) ), gen_loss )
        plt.show()
