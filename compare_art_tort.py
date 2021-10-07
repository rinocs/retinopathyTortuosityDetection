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

    
with open("Data/artery_tort_test_1.json", "r") as read_file:
    data_art = json.load(read_file)    
    
entries = sorted(data_art.items(), key=lambda items: items[1]['rank'])


name_art = []
dist_infl_art_tort = []
squared_art_tort = []
distance_art_tort = []
linear_reg_art_tort = []
density_art_tort = []
rank = []
vti = []
vti_mat = []
for key, value in entries:
    name_art.append(key)
    dist_infl_art_tort.append(value['distance_inflection_tort'])
    distance_art_tort.append(value['distance_tortuosity'])
    linear_reg_art_tort.append(value['linear_reg_tort'])
    density_art_tort.append(value['squared_tort'])
    squared_art_tort.append(value['tortuosity_density'])
    vti.append(value["vti"])
    vti_mat.append(value["vti_mat"])
    rank.append(int(value['rank']))
    
print(rank)
 
rank = np.array(rank)
dist_infl_art_tort = np.array(dist_infl_art_tort)
squared_art_tort = np.array(squared_art_tort)
distance_art_tort = np.array(distance_art_tort)
linear_reg_art_tort = np.array(linear_reg_art_tort)
density_art_tort = np.array(density_art_tort)
vti = np.array(vti)
vti_mat = np.array(vti_mat)
# norm1 = rank / np.linalg.norm(rank)
# norm1 = rank /30

#### sklearn normalize --- min-max scaling
# rank_norm = normalize(rank[:,np.newaxis], axis=0).ravel()
# dist_infl_art_tort_norm = normalize(dist_infl_art_tort[:,np.newaxis], axis=0).ravel()
# squared_art_tort_norm = normalize(squared_art_tort[:,np.newaxis], axis=0).ravel()
# distance_art_tort_norm = normalize(distance_art_tort[:,np.newaxis], axis=0).ravel()
# linear_reg_art_tort_norm = normalize(linear_reg_art_tort[:,np.newaxis], axis=0).ravel()
# density_art_tort_norm = normalize(density_art_tort[:,np.newaxis], axis=0).ravel()
# vti_norm = normalize(vti[:,np.newaxis],axis=0).ravel()
####  my normalize function 
# rank_norm = np.array(list(myNormalize(rank)))
# dist_infl_art_tort_norm = np.array(list(myNormalize(dist_infl_art_tort)))
# squared_art_tort_norm =   np.array(list(myNormalize(squared_art_tort)))
# distance_art_tort_norm =  np.array(list(myNormalize(distance_art_tort)))
# linear_reg_art_tort_norm =np.array(list(myNormalize(linear_reg_art_tort)))
# density_art_tort_norm =   np.array(list(myNormalize(density_art_tort)))
# vti_norm = np.array(list(myNormalize(vti)))
# print(rank_norm)


### min -max scaler
# define min max scaler
scaler = MinMaxScaler()
# transform data
rank_norm = scaler.fit_transform(rank.reshape(-1,1))
dist_infl_art_tort_norm = scaler.fit_transform(dist_infl_art_tort.reshape(-1,1))
squared_art_tort_norm = scaler.fit_transform(squared_art_tort.reshape(-1,1))
distance_art_tort_norm = scaler.fit_transform(distance_art_tort.reshape(-1,1))
linear_reg_art_tort_norm = scaler.fit_transform(linear_reg_art_tort.reshape(-1,1))
density_art_tort_norm = scaler.fit_transform(density_art_tort.reshape(-1,1))
vti_norm= scaler.fit_transform(vti.reshape(-1,1))
vti_mat_norm= scaler.fit_transform(vti_mat.reshape(-1,1))


mse_dist_infl_art_tort = mean_squared_error(rank_norm, dist_infl_art_tort_norm)
mse_squared_art_tort = mean_squared_error(rank_norm, squared_art_tort_norm)
mse_distance_art_tort = mean_squared_error(rank_norm, distance_art_tort_norm)
mse_linear_reg_art_tort = mean_squared_error(rank_norm, linear_reg_art_tort_norm)
mse_density_art_tort = mean_squared_error(rank_norm, density_art_tort_norm)
mse_vti = mean_squared_error(rank_norm, vti_norm)
mse_vti_mat = mean_squared_error(rank_norm, vti_mat_norm)


###R2score
##R2SCORE   
r2_dist_infl_art_tort = r2_score(rank_norm, dist_infl_art_tort_norm)
r2_squared_art_tort = r2_score(rank_norm, squared_art_tort_norm)
r2_distance_art_tort = r2_score(rank_norm, distance_art_tort_norm)
r2_linear_reg_art_tort = r2_score(rank_norm, linear_reg_art_tort_norm)
r2_density_art_tort = r2_score(rank_norm, density_art_tort_norm)
r2_vti = r2_score(rank_norm, vti_norm)
r2_vti_mat = r2_score(rank_norm, vti_mat_norm)



# norm2 = art_tort/30
# norm2 = art_tort/ np.linalg.norm(art_tort)

# norm1 = normalize([rank])
# norm2 = normalize([art_tort])
# print (np.all(norm1 == norm2))
# True    
    




fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].scatter(name_art, rank_norm)
ax[0].scatter(name_art, density_art_tort_norm)
ax[0].set_title(' MSE:'+"{0:.5f}".format(mse_density_art_tort) + "r2score:" +"{0:.5f}".format(r2_density_art_tort))
ax[0].set_xlabel("image name")
ax[0].set_ylabel("tortuosity_density")
ax[0].set_xticklabels(name_art ,rotation = 45)


ax[1].scatter(name_art, rank_norm)
ax[1].scatter(name_art, dist_infl_art_tort_norm)
ax[1].set_title(' MSE:' +"{0:.5f}".format(mse_dist_infl_art_tort) + "r2score:" +"{0:.5f}".format(r2_dist_infl_art_tort))
ax[1].set_xlabel("image name")
ax[1].set_ylabel("distance_inflection_tort")
ax[1].set_xticklabels(name_art ,rotation = 45)



ax[2].scatter(name_art, rank_norm)
ax[2].scatter(name_art, squared_art_tort_norm)
ax[1].set_title(' MSE:' +"{0:.5f}".format(mse_squared_art_tort) + "r2score:" +"{0:.5f}".format(r2_squared_art_tort))
ax[2].set_xlabel("image name")
ax[2].set_ylabel("squared_tort")
ax[2].set_xticklabels(name_art ,rotation = 45)


ax[3].scatter(name_art, rank_norm)
ax[3].scatter(name_art, distance_art_tort_norm)
ax[1].set_title(' MSE:' +"{0:.5f}".format(mse_distance_art_tort) + "r2score:" +"{0:.5f}".format(r2_distance_art_tort))
ax[3].set_xlabel("image name")
ax[3].set_ylabel("distance_tortuosity")
ax[3].set_xticklabels(name_art ,rotation = 45)


ax[4].scatter(name_art, rank_norm)
ax[4].scatter(name_art, linear_reg_art_tort_norm)
ax[1].set_title(' MSE:' +"{0:.5f}".format(mse_linear_reg_art_tort) + "r2score:" +"{0:.5f}".format(r2_linear_reg_art_tort))
ax[4].set_xlabel("image name")
ax[4].set_ylabel("linear_reg_tort")
ax[4].set_xticklabels(name_art ,rotation = 45)

ax[5].scatter(name_art, rank_norm)
ax[5].scatter(name_art, vti_norm)
ax[1].set_title(' MSE:' +"{0:.5f}".format(mse_vti) + "r2score:" +"{0:.5f}".format(r2_vti))
ax[5].set_xlabel("image name")
ax[5].set_ylabel("vti ")
ax[5].set_xticklabels(name_art ,rotation = 45)

ax[6].scatter(name_art, rank_norm)
ax[6].scatter(name_art, vti_mat_norm)
ax[1].set_title(' MSE:' +"{0:.5f}".format(mse_vti_mat) + "r2score:" +"{0:.5f}".format(r2_vti_mat))
ax[6].set_xlabel("image name")
ax[6].set_ylabel("vti_mat ")
ax[6].set_xticklabels(name_art ,rotation = 45)



fig.tight_layout()
# plt.savefig('compare_art_tort_mm_scaler.png')
plt.show()


plt.scatter(name_art, rank_norm)
plt.scatter(name_art, distance_art_tort_norm)
plt.title("distance_tort")
plt.xlabel("image")
plt.ylabel("distance_tort")
# plt.savefig('art_dist_tort.png')
plt.show()

plt.scatter(name_art, rank_norm)
plt.scatter(name_art, vti_norm)
plt.title("vti")
plt.xlabel("image")
plt.ylabel("vti")
# plt.savefig('art_dist_tort.png')
plt.show()

plt.scatter(name_art, rank_norm)
plt.scatter(name_art, vti_mat_norm)
plt.title("vti_mat")
plt.xlabel("image")
plt.ylabel("vti_mat")
# plt.savefig('art_dist_tort.png')
plt.show()




# fig, ax = plt.subplots() # Create the figure and axes object

# ax.plot(name_art, norm1, marker="o")
# ax.set_xlabel("image name")
# ax.set_ylabel("tortuosity_density")
# ax.set_xticklabels(name_art ,rotation = 45)
# ax.plot(name_art, norm2, marker="o")
# plt.show()

# plt.scatter(name_art, norm1)
# plt.scatter(name_art, norm2)

# plt.show()