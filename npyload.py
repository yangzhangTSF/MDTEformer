import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

name = "Australia_il48_ol48"
#name = "European_il24_ol24"
#name = "GEFC2012_il24_ol24"
#name = "UK_il48_ol48"

pre_file =np.load("results/MDTEformer_"+name+"_sl6_win2_fa10_dm256_nh4_el3_itr0/pred.npy")
tre_file =np.load("results/MDTEformer_"+name+"_sl6_win2_fa10_dm256_nh4_el3_itr0/true.npy")

print(pre_file.shape)
print(tre_file.shape)
pre_file = pre_file[:,:1,:1].ravel()
tre_file = tre_file[:,:1,:1].ravel()
print(pre_file.shape)
print(tre_file.shape)

data = {'pre_file': pre_file, 'tre_file': tre_file}
df = pd.DataFrame(data)
df.to_csv("MDTEformer"+name+'pre.csv', index=False)

plt.plot(pre_file[:600], color='red', label='pre', marker='o')
plt.plot(tre_file[:600], color='b', label='true', marker='*')
plt.legend()
plt.show()