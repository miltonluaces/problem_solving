

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from apyori import apriori 
#from Utils.Admin.Standard import *


storeData = pd.read_csv('D:/data/csv/storeData.csv')[:7501] 
print(storeData.head(10))

# Data Preprocessing
records = []  
for i in range(0, 7501):  
    records.append([str(storeData.values[i,j]) for j in range(0, 20)])

# Apply apriori
assocRules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)  
assocRes = list(assocRules)  
print(assocRules[0])  

# Show support, confidence, lift
for item in association_rules:

    # 1st index of the inner list Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    # 2nd index of the inner list
    print("Support: " + str(item[1]))

    # 3rd index of the list located at 0th of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print(" ")