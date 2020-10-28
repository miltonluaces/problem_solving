# Backtracking example: Hamiltonian Cycle
  

def CreateGraph(vertices): 
    graph = [[0 for column in range(vertices)] for row in range(vertices)] 
    return(graph)
  
# Check if this vertex is an adjacent vertex of the previously added vertex and is not included in the path earlier 
def IsSafe(graph, nVert, pos, path): 
    # Check if current vertex and last vertex in path are adjacent 
    if graph[path[pos-1]][nVert] == 0: 
        return False 
    # Check if current vertex not already in path 
    for v in path: 
        if v == nVert: return False  
    return True
  
# A recursive utility function to solve hamiltonian cycle problem 
def HamCycleBack(graph, nVert, path, pos): 
    # base case: if all vertices are included in the path 
    if pos == nVert: 
        # Last vertex must be adjacent to the first vertex in path to make a cyle 
        if graph[path[pos-1]][ path[0] ] == 1: return True
        else: return False
  
    # Recursive case: Try vertices as next candidate in Hamiltonian Cycle. (0 already included)
    for v in range(1,nVert): 
        if IsSafe(graph, v, pos, path) == True: 
            path[pos] = v
            if HamCycleBack(graph, nVert, path, pos+1) == True: return True
            # Backtracking step: Remove current vertex if it doesn't lead to a solution 
            path[pos] = -1
    return False
  
def HamCycle(graph, nVert): 
    path = [-1] * nVert
    # Put vertex 0 as the first vertex in the path. If there is a Hamiltonian Cycle in an undirected graph, the cycle can be started from any point
    path[0] = 0  
    if HamCycleBack(graph, nVert, path, 1) == False: print("Solution does not exist\n"); return False 
    else: Print(path) 
    return True
  
def Print(path): 
    print("Solution Exists: Following is one Hamiltonian Cycle")
    for vertex in path: print(vertex,) ; print(path[0], "\n")
  
# Test
  
# Create the following graph 
#      (0)--(1)--(2) 
#       |   / \   | 
#       |  /   \  | 
#       | /     \ | 
#      (3)-------(4)  

nVert = 5
g1 = CreateGraph(nVert) 
g1 = [ [0, 1, 0, 1, 0], [1, 0, 1, 1, 1], [0, 1, 0, 0, 1,], [1, 1, 0, 0, 1], [0, 1, 1, 1, 0], ] 
  
# Print the solution 
HamCycle(g1, nVert); 
  
# Same graph without the 3-4 edge (no cycle)

g2 = CreateGraph(nVert) 
g2 = [ [0, 1, 0, 1, 0], [1, 0, 1, 1, 1], [0, 1, 0, 0, 1,], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], ] 
HamCycle(g2, nVert); 
