import json
# import tensorflow as tf
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

# Prints only high acc pitches 
def print_pitches_high(pitches):
    for p in pitches:
        for e in p:
            if (e > 0.99999):
                print(e, end = " ")
            else: 
                print("...", end = " ")
        print("")

# Prints app pitches
def print_pitches_all(pitches):
    for p in pitches:
        print(p)

# Reading a track file
with open('mess.json') as f:
    track1 = json.load(f)
#with open('track2.json') as f:
#    track2 = json.load(f)
#with open('track3.json') as f:
#    track3 = json.load(f)
#with open('track4.json') as f:
#    track4 = json.load(f)
#with open('track5.json') as f:
#    track5 = json.load(f)
#with open('track6.json') as f:
#    track6 = json.load(f)
#with open('track7.json') as f:
#    track7 = json.load(f)
#with open('track8.json') as f:
#    track8 = json.load(f)

# Save all pitch values to pitches array
pitches = []
for segment in track1['segments']:
    pitches.append(np.asarray(segment['pitches']))
#for segment in track2['segments']:
#    pitches.append(np.asarray(segment['pitches']))
#for segment in track3['segments']:
#    pitches.append(np.asarray(segment['pitches']))
#for segment in track4['segments']:
#    pitches.append(np.asarray(segment['pitches']))
#for segment in track5['segments']:
#    pitches.append(np.asarray(segment['pitches']))
#for segment in track6['segments']:
#    pitches.append(np.asarray(segment['pitches']))
#for segment in track7['segments']:
#    pitches.append(np.asarray(segment['pitches']))
#for segment in track8['segments']:
#    pitches.append(np.asarray(segment['pitches']))
pitches = np.asarray(pitches)

# For visual representation 
# pitches_visual = []
# for segment in track['segments']:
#    for i in range(0,int(segment['duration']*10)):
#        pitches_visual.append(np.asarray(segment['pitches']))
# pitches_visual = np.asarray(pitches_visual)

# print(len(pitches))

# im = x.reshape(1500, 1500)
im = pitches[:20]

plt.gray()
plt.imshow(im)
plt.savefig(fname = "mess_old.png", dpi = 120)
# plt.show()

pred = []

# large rank for first solution that I done
# 1 rank = 1 chunk = all data
ran = 1 
chunk = len(pitches) / ran

for i in range(ran):
    val_x = pitches[int(i * chunk): int((i+1) * chunk)]
    # n_clusters = 5 -- first problem
    # large n_cluster for second
    km = KMeans(n_jobs = -1, n_clusters = 80, n_init = 20)
    km.fit(val_x)
    pred.append(list(km.predict(val_x)))
print("First Kmean over...")

indexlist = []
# for p in pred:
#    print(p, end = ", ")

#### Generate list out of all indexes
for pre in pred:
    for p in pre:
        indexlist.append(p)
# print(indexlist)
print("Generating list over...")

#### Generate all possible chunks
chunk = 6
chunklist = []

for i in range(len(indexlist) - chunk):
    chunklist.append(indexlist[i:i+chunk])
print("Generating chunks over...")
# print(chunklist)

### Apply Kmeans to chunklist with large n_cluster number

patternpred = []

km = KMeans(n_jobs = -1, n_clusters = int(len(chunklist) / 4), n_init = 20)
km.fit(chunklist)
patternpred.append(list(km.predict(chunklist)))
print("Second Kmean over...")

# print(patternpred)
duplicates = set([x for x in patternpred[0] if patternpred[0].count(x) > 1])

print("Chunk number, " + str(chunk) + ": "+ str(duplicates))

for dup in duplicates:
    print("Cluster Num : " + str(dup) + ", Count : " + str(patternpred[0].count(dup)) + " Indexes : " + str([index for index, value in enumerate(patternpred[0]) if value == dup]))


output = {}

output['xdatas'] = []

plt.gray()
plt.hist(pred[0])
# plt.show()




#for pre in pred:
#    for p in pre:
#        output['xdatas'].append({'index' : int(p)})

#with open(str('1NeLwFETswx8Fzxl2AFll91'+'_xdata_onlykmean.json'), 'w') as outfile:
#    json.dump(output, outfile)

#input_arr = Input(shape = (12,))

#encoded = Dense(500, activation = "relu")(input_arr)

#encoded = Dense(500, activation = "relu")(encoded)

#encoded = Dense(2000, activation = "relu")(encoded)

#encoded = Dense(5, activation = "sigmoid")(encoded)
# *************************************************
#decoded = Dense(2000, activation = "relu")(encoded)

#decoded = Dense(500, activation = "relu")(decoded)

#decoded = Dense(500, activation = "relu")(decoded)

#decoded = Dense(12)(decoded)

#autoencoder = Model(input_arr, decoded)

#autoencoder.summary()

#encoder = Model(input_arr, encoded)

#autoencoder.compile(optimizer="adam", loss="mse")

#print(len(val_x))
#train_history = autoencoder.fit(val_x, val_x , epochs = 10, batch_size = 2048, validation_data = (val_x, val_x))

#pred_auto_train = encoder.predict(val_x[:])

#pred_auto = encoder.predict(val_x[:])

#km.fit(pred_auto_train)

#pred = km.predict(pred_auto)

#for p in pred:
#    print(p)

#output = {}

#output['xdatas'] = []

#for p in pred:
#    output['xdatas'].append({'index' : int(p)})

#with open(str('1NeLwFETswx8Fzxl2AFll91'+'_xdata.json'), 'w') as outfile:
#    json.dump(output, outfile)

### PATTERN ANALYSIS ###
# Generate pattern search from this numbers

# Specify max pattern number

# Generate pattern arrays for 