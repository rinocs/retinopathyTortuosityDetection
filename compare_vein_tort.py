import cv2
import numpy as np
import pandas as pd
import json
from utils.math import tortuosity
import scipy.io
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


from pandas.io.json import json_normalize


def myNormalize(lst):
    s = sum(lst)
    return map(lambda x: float(x)/s, lst)


# loading file
with open("DATA/veins_tort.json", "r") as read_file:
    data_vein = json.load(read_file)
    

    
entries = sorted(data_vein.items(), key=lambda items: items[1]['rank'])



name_vein = []
dist_infl_vein_tort = []
squared_vein_tort = []
distance_vein_tort = []
linear_reg_vein_tort = []
density_vein_tort = []
rank = []
for key, value in entries:
    name_vein.append(key)
    dist_infl_vein_tort.append(value['distance_inflection_tort'])
    distance_vein_tort.append(value['distance_tortuosity'])
    linear_reg_vein_tort.append(value['linear_reg_tort'])
    density_vein_tort.append(value['squared_tort'])
    squared_vein_tort.append(value['tortuosity_density'])
    rank.append(int(value['rank']))
    
print(rank)
 
rank = np.array(rank)
dist_infl_vein_tort = np.array(dist_infl_vein_tort)
squared_vein_tort = np.array(squared_vein_tort)
distance_vein_tort = np.array(distance_vein_tort)
linear_reg_vein_tort = np.array(linear_reg_vein_tort)
density_vein_tort = np.array(density_vein_tort)
# norm1 = rank / np.linalg.norm(rank)
# norm1 = rank /30


#### sklearn normalize --- min-max scaling
# rank_norm = normalize(rank[:,np.newaxis], axis=0).ravel()
# dist_infl_vein_tort_norm = normalize(dist_infl_vein_tort[:,np.newaxis], axis=0).ravel()
# squared_vein_tort_norm = normalize(squared_vein_tort[:,np.newaxis], axis=0).ravel()
# distance_vein_tort_norm = normalize(distance_vein_tort[:,np.newaxis], axis=0).ravel()
# linear_reg_vein_tort_norm = normalize(linear_reg_vein_tort[:,np.newaxis], axis=0).ravel()
# density_vein_tort_norm = normalize(density_vein_tort[:,np.newaxis], axis=0).ravel()




####  my normalize function 
# rank_norm = np.array(list(myNormalize(rank)))
# dist_infl_vein_tort_norm = np.array(list(myNormalize(dist_infl_vein_tort)))
# squared_vein_tort_norm =   np.array(list(myNormalize(squared_vein_tort)))
# distance_vein_tort_norm =  np.array(list(myNormalize(distance_vein_tort)))
# linear_reg_vein_tort_norm =np.array(list(myNormalize(linear_reg_vein_tort)))
# density_vein_tort_norm =   np.array(list(myNormalize(density_vein_tort)))
# print(rank_norm)


### min -max scaler
# define min max scaler
scaler = MinMaxScaler()
# transform data
rank_norm = scaler.fit_transform(rank.reshape(-1,1))
dist_infl_vein_tort_norm = scaler.fit_transform(dist_infl_vein_tort.reshape(-1,1))
squared_vein_tort_norm = scaler.fit_transform(squared_vein_tort.reshape(-1,1))
distance_vein_tort_norm = scaler.fit_transform(distance_vein_tort.reshape(-1,1))
linear_reg_vein_tort_norm = scaler.fit_transform(linear_reg_vein_tort.reshape(-1,1))
density_vein_tort_norm = scaler.fit_transform(density_vein_tort.reshape(-1,1))



## MSE computation
mse_dist_infl_vein_tort = mean_squared_error(rank_norm, dist_infl_vein_tort_norm)
mse_squared_vein_tort = mean_squared_error(rank_norm, squared_vein_tort_norm)
mse_distance_vein_tort = mean_squared_error(rank_norm, distance_vein_tort_norm)
mse_linear_reg_vein_tort = mean_squared_error(rank_norm, linear_reg_vein_tort_norm)
mse_density_vein_tort = mean_squared_error(rank_norm, density_vein_tort_norm)


##R2SCORE   
r2_dist_infl_vein_tort = r2_score(rank_norm, dist_infl_vein_tort_norm)
r2_squared_vein_tort = r2_score(rank_norm, squared_vein_tort_norm)
r2_distance_vein_tort = r2_score(rank_norm, distance_vein_tort_norm)
r2_linear_reg_vein_tort = r2_score(rank_norm, linear_reg_vein_tort_norm)
r2_density_vein_tort = r2_score(rank_norm, density_vein_tort_norm)

# norm2 = vein_tort/30
# norm2 = vein_tort/ np.linalg.norm(vein_tort)

# norm1 = normalize([rank])
# norm2 = normalize([vein_tort])
# print (np.all(norm1 == norm2))
# True    
    




fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
axes[1][2].set_visible(False)
ax = axes.ravel()

ax[0].scatter(name_vein, rank_norm)
ax[0].scatter(name_vein, density_vein_tort_norm)
ax[0].set_title('MSE:'+str(mse_density_vein_tort)+ "R2" + str(r2_density_vein_tort))
ax[0].set_xlabel("image name")
ax[0].set_ylabel("tortuosity_density")
ax[0].set_xticklabels(name_vein ,rotation = 45)


ax[1].scatter(name_vein, rank_norm)
ax[1].scatter(name_vein, dist_infl_vein_tort_norm)
ax[1].set_title('MSE:' +str(mse_dist_infl_vein_tort)+ "R2" + str(r2_dist_infl_vein_tort))
ax[1].set_xlabel("image name")
ax[1].set_ylabel("distance_inflection_tort")
ax[1].set_xticklabels(name_vein ,rotation = 45)



ax[2].scatter(name_vein, rank_norm)
ax[2].scatter(name_vein, squared_vein_tort_norm)
ax[2].set_title(' MSE:'+str(mse_squared_vein_tort)+ "R2" + str(r2_squared_vein_tort))
ax[2].set_xlabel("image name")
ax[2].set_ylabel("squared_tort")
ax[2].set_xticklabels(name_vein ,rotation = 45)


ax[3].scatter(name_vein, rank_norm)
ax[3].scatter(name_vein, distance_vein_tort_norm)
ax[3].set_title(' MSE:' +str(mse_distance_vein_tort)+ "R2" + str(r2_distance_vein_tort))
ax[3].set_xlabel("image name")
ax[3].set_ylabel("distance_tortuosity")
ax[3].set_xticklabels(name_vein ,rotation = 45)


ax[4].scatter(name_vein, rank_norm)
ax[4].scatter(name_vein, linear_reg_vein_tort_norm)
ax[4].set_title(' MSE:' + str(mse_linear_reg_vein_tort) + "R2" + str(r2_linear_reg_vein_tort))
ax[4].set_xlabel("image name")
ax[4].set_ylabel("linear_reg_tort")
ax[4].set_xticklabels(name_vein ,rotation = 45)



fig.tight_layout()
# plt.savefig('compare_vein_tort_sk_norm.png')
plt.show()






# fig, ax = plt.subplots() # Create the figure and axes object

# ax.plot(name_vein, norm1, marker="o")
# ax.set_xlabel("image name")
# ax.set_ylabel("tortuosity_density")
# ax.set_xticklabels(name_vein ,rotation = 45)
# ax.plot(name_vein, norm2, marker="o")
# plt.show()

# plt.scatter(name_vein, norm1)
# plt.scatter(name_vein, norm2)

# plt.show()