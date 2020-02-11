import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data-1.npy', allow_pickle=True)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []
backwards = []
spaces = []


#dists = np.array([i[0] for i in train_data])
#keys = np.array([j[1] for j in train_data])



shuffle(train_data)

for data in train_data:
    dists = data[0]
    keys = data[1]
    
    if keys == [1,0,0,0,0]:
        forwards.append([dists,keys])
    elif keys == [0,1,0,0,0]:
        backwards.append([dists,keys])
    elif keys == [0,0,1,0,0]:
        lefts.append([dists,keys])
    elif keys == [0,0,0,1,0]:
        rights.append([dists,keys])
    elif keys == [0,0,0,0,1]:
        spaces.append([dists,keys])
    else:
        print('no matches')

#print(keys)
forwards = forwards[:len(lefts)][:len(rights)][:len(backwards)][:len(spaces)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
backwards = backwards[:len(forwards)]
spaces = spaces[:len(forwards)]


final_data = forwards + lefts + rights + backwards + spaces

shuffle(final_data)

np.save('training_data_v2.npy', final_data)
#"""