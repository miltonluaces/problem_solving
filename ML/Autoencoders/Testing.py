import unittest
import numpy as np
import AE as Ae
import MNIST_data.inputData as id


# Test TsCat, transforming numerical time series in categorical
# -------------------------------------------------------------------
   
class AETest(unittest.TestCase):

    # Setup & TearDown
    # ---------------------------------------------------------------
    
    def setUp(self):
        pass


    def tearDown(self):
        pass

    if __name__ == "__main__":
        #import sys;sys.argv = ['', 'Test.testName']
        unittest.main()
        
 # Tests AE: Test BinVar and BinTS
 # ---------------------------------------------------------------
    
    def testAE(self):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        
        print ("Test AE")
        
        input = np.array([[2.0, 1.0, 1.0, 2.0], [-2.0, 1.0, -1.0, 2.0], [0.0, 1.0, 0.0, 2.0], [0.0, -1.0, 0.0, -2.0], [0.0, -1.0, 0.0, -2.0]]) ; print(input) 
        ae = Ae.AE()
        ae.Build(input)
        ae.Train(True)
        res = ae.Output()
        print(res)
        pass

    def train(self, nnArch, mnist=None, lrate=0.001, bSize=100, epochs=10, printStep=5, nSamples=100):
    
        vae = Vae.VAE(nnArch, lrate=lrate, bSize=bSize)
        
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = int(nSamples / bSize)
            for i in range(total_batch):
                batch_xs, _ = mnist.train.next_batch(bSize)
                cost = vae.Fit(batch_xs)
                avg_cost += cost[1] / nSamples * bSize
                if epoch % printStep == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    
 # Tests VAE
 # ---------------------------------------------------------------
   
    def testVAE(self):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        
        print ("Test VAE")
        id.read_data_sets()
        mnist = id.read_data_sets('MNIST_data', one_hot=True)
        nnArch = dict(nHrec1=500, nHrec2=500, nHgen1=500, nHgen2=500, nInput=784, nZ=20)
        self.train(nnArch, mnist, epochs=5, nSamples=mnist.train.num_examples)
        pass

    def testName(self):
        df = pd.read_csv('TsToyAnomaly.csv', sep='\t', header=None)
        dataset = df.transpose()
        nnArch = dict(nHrec1=25, nHrec2=25, nHgen1=25, nHgen2=25, nInput=50, nZ=20)
        vae = Vae.VAE(nnArch, lrate=0.001, bSize=100)
        vae.train(epochs=10)
        pass

    

         
    

    
