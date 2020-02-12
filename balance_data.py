import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data-1.npy', allow_pickle=True)

onlyForwards = []
everythingElse = []

shuffle(train_data)

for data in train_data:
    dists = data[0]
    keys = data[1]
    
    if keys == [1,0,0,0,0]:
        onlyForwards.append([dists,keys])
    else:
        everythingElse.append([dists,keys])

final_data = everythingElse

df = pd.DataFrame(final_data)
print(df.head())
print(Counter(df[1].apply(str)))
df = pd.DataFrame(onlyForwards)
print("Only Forwards : ")
print(Counter(df[1].apply(str)))

shuffle(final_data)

np.save('training_data-balanced.npy', final_data)
