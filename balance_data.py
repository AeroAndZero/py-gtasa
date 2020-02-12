import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data-1.npy', allow_pickle=True)

onlyForwards = []
onlyLefts = []
onlyRights = []
everythingElse = []

forwardCount = 0
leftCount = 0
rightCount = 0

shuffle(train_data)

for data in train_data:
    dists = data[0]
    keys = data[1]
    
    if keys == [1,0,0,0,0]:
        onlyForwards.append([dists,keys])
        forwardCount += 1
    elif keys == [0,1,0,0,0]:
        onlyLefts.append([dists,keys])
        leftCount += 1
    elif keys == [0,0,0,1,0]:
        onlyRights.append([dists,keys])
        rightCount += 1
    else:
        everythingElse.append([dists,keys])


minCount = min(forwardCount,leftCount,rightCount)

onlyForwards = onlyForwards[:minCount]
onlyLefts = onlyLefts[:minCount]
onlyRights = onlyRights[:minCount]

final_data = onlyForwards + onlyLefts + onlyRights + everythingElse

df = pd.DataFrame(final_data)
print(df.head())
print(Counter(df[1].apply(str)))

shuffle(final_data)

np.save('training_data-balanced.npy', final_data)
