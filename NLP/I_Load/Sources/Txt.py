import json

file = open('Data/json1.json')
data = json.load(file)
print(data['size'])