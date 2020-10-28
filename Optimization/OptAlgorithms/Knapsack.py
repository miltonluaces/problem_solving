import knapsack

size = [21, 11, 15, 9, 34, 25, 41, 52]
weight = [22, 12, 16, 10, 35, 26, 42, 53]
capacity = 100

sol = knapsack.knapsack(size, weight).solve(capacity)
print(sol)

