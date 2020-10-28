import unittest
import numpy as np
import pylab as plt
import networkx as nx
import QLearning.QLearn as QL


class TestQLearning(unittest.TestCase):
    
    def test10_QLearnLaberith(self):
        # Define graph laberinth (fully connected). Starts in 0, ends in 7
        nodes = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
        goal = 7

        # Plot
        G=nx.Graph()
        G.add_edges_from(nodes)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G,pos)
        nx.draw_networkx_edges(G,pos)
        nx.draw_networkx_labels(G,pos)
        plt.show()

        # Reward matrix
        nNodes = 8
        R = np.matrix(np.ones(shape=(nNodes, nNodes)))
        R *= -1

        # assign zeros to paths and 100 to goal-reaching point
        for n in nodes:
            print(n)
            if n[1] == goal: R[n] = 100
            else: R[n] = 0

            if n[0] == goal: R[n[::-1]] = 100
            else: R[n[::-1]]= 0 # reverse

        # add goal 
        R[goal,goal]= 100
        print(R)



        # Q Reinforcement Algorithm
        Q = np.matrix(np.zeros([nNodes, nNodes]))
        ql = QL.QLearn(R, Q, gamma=0.8, ini=1)
    
        # Training
        scores = ql.Train(epochs=700)
        print("Trained Q matrix:")
        print(Q/np.max(Q)*100)

        # Testing
        currState = 0
        steps = [currState]
        while currState != 7:
            nextStepIdx = np.where(Q[currState,] == np.max(Q[currState,]))[1]
            if nextStepIdx.shape[0] > 1: nextStepIdx = int(np.random.choice(nextStepIdx, size = 1))
            else: nextStepIdx = int(nextStepIdx)
            steps.append(nextStepIdx)
            currState = nextStepIdx

        print("Most efficient path:", steps)
        plt.plot(scores)
        plt.show()



if __name__ == '__main__':
    unittest.main()
