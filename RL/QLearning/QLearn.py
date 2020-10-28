import numpy as np

class QLearn:

    def __init__(self, R, Q, gamma, ini):
        self.R = R
        self.Q = Q
        self.gamma = gamma
        self.ini = ini
       
        self.candidateActs = self.CandidateActions(ini) 
        action = self.SampleNextAction(self.candidateActs)
        self.Update(ini, action, self.gamma)

    
    def CandidateActions(self, state):
        currStateRow = self.R[state,]
        ca = np.where(currStateRow >= 0)[1]
        return ca

    def SampleNextAction(self, candidateActionsRange):
        na = int(np.random.choice(self.candidateActs, 1))
        return na

    def Update(self, currState, action, gamma):
      maxIdx = np.where(self.Q[action,] == np.max(self.Q[action,]))[1]
      if maxIdx.shape[0] > 1: maxIdx = int(np.random.choice(maxIdx, size = 1))
      else: maxIdx = int(maxIdx)
      maxValue = self.Q[action, maxIdx]
  
      self.Q[currState, action] = self.R[currState, action] + self.gamma * maxValue
      print('max_value', self.R[currState, action] + gamma * maxValue)
  
      if (np.max(self.Q) > 0): return(np.sum(self.Q/np.max(self.Q)*100))
      else: return (0)

    def Train(self, epochs):
        scores = []
        for i in range(epochs):
            currState = np.random.randint(0, int(self.Q.shape[0]))
            self.candidateActs = self.CandidateActions(currState)
            action = self.SampleNextAction(self.candidateActs)
            score = self.Update(currState, action, self.gamma)
            scores.append(score)
        return(scores)
            
     


