import random
import numpy as np

reward = np.array([[0, -1, 0, -10],
                   [0, -1, -1, -1],
                   [0, -10, -10, -1],
                   [0, 100, -1, 0],
                   
                   [-1, -1, 0, -1],
                   [-10, -1, -1, -10],
                   [-1, -1, -1, 100],
                   [-1, -1, -1, 0],
                   
                   [-1, -1, 0, -1],
                   [-1, -1, -1, -1],
                   [-10, -1, -1, -1],
                   [100, -1, -1, 0],
                   
                   [-1, 0, 0, -1],
                   [-1, 0, -1, -1],
                   [-1, 0, -1, -1],
                   [-1, 0, -1, 0]])

States = np.array([[-1,4,-1,1],
                [-1,5,0,2],
                [-1,6,1,3],
                [-1,7,2,-1],
                
                [0,8,-1,5],
                [1,9,4,6],
                [2,10,5,7],
                [3,11,6,-1],
                
                [4,12,-1,9],
                [5,13,8,10],
                [6,14,9,11],
                [7,15,10,-1],
                
                [8,-1,-1,13],
                [9,-1,12,14],
                [10,-1,13,15],
                [11,-1,14,-1]])

action = np.array([[1,3],
                   [1, 2, 3],
                   [1, 2, 3],
                   [1, 2],
                   
                   [0,1,3],
                   [0,1,2,3],
                   [0,1,2,3],
                   [0,1,2],
                   
                   [0,1,3],
                   [0,1,2,3],
                   [0,1,2,3],
                   [0,1,2],
                   
                   [0,3],
                   [0,2,3],
                   [0,2,3],
                   [0,2]])


epsilon = 0.3
decEpsilon = 0.03
gamma = 0.7
lrA = 0.7
goalState = 7

print("\nQ Learning\n")

Q = np.zeros((16,4))
nextState = 0
episode = 1
Path = []

for i in range(10):
    re = 0
    currState = 0
    Path = []

    while(currState != 7):
        Qx = -999
        for i in action[currState]:
            if Q[currState][i] > Qx:
                currAct = i
                Qx = Q[currState][i]
        nextState = States[currState][currAct]
        "print(i_state,"  ",n_state)"
        
        nextValues = []
        for i in action[nextState]:
            nextValues.append(Q[nextState][i])
        Max = max(nextValues)
        re = re + reward[currState][currAct] + gamma * Max - Q[currState][currAct]
        Q[currState][currAct] = Q[currState][currAct] + lrA * (reward[currState][currAct] + gamma * Max - Q[currState][currAct])
        Path.append(currState)
        currState = nextState
    Path.append(currState)
    episode += 1

print(Path)
print("The Final Q Matrix is: \n", np.divide(Q,np.amax(Q)))

print("\nSARSA Learning\n")

Q = np.zeros((16,4))
nextState = 0
episode = 1
Path = []

def SelectAction(cState, nState):
    i = random.random()
    if(i > epsilon):
        MaxRew = [-9, -9, -9, -9]
        for j in action[cState]: MaxRew[j] = Q[nState][j]
        cAct = np.argmax(MaxRew)
    else:
        cAct = random.choice(action[cState])
    return cAct

for i in range(10):
    re = 0
    currState = 0
    Path = []
    currAct = SelectAction(currState, nextState)

    while(currState != goalState):
        nextState = States[currState][currAct]
        nextAct = SelectAction(nextState, nextState)

        Q[currState][currAct] = Q[currState][currAct] + lrA * (reward[currState][currAct] + gamma * Q[nextState][nextAct] - Q[currState][currAct])
        Path.append(currState)
        currState = nextState
        currAct = nextAct
    
    Path.append(currState)
    episode += 1
    if epsilon > 0: epsilon -= decEpsilon

print(Path)
print("The Final Q Matrix is: \n", np.divide(Q,np.amax(Q)))



