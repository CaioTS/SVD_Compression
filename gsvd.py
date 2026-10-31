#%%
import numpy as np
import scipy as signal
from  easygsvd import gsvd

def format_W(W,C):
    R = int(W.shape[0]/C)
    
    n_pad = W.shape[0] % R
    W_pad = np.zeros(int(W.shape[0] + (R - n_pad)))
    W_pad[:W.shape[0]] = W
    
    lower_dim = min(R,C)
    return  W_pad.reshape((int(lower_dim),-1),order = 'F')


#%%

A = np.loadtxt("wsecimpulse.txt")
B = np.loadtxt("wfbkimpulse.txt")

#A_m = format_W(A,50)
#B_m = format_W(B,50)
A_m = A.reshape((-1,50))
B_m = B.reshape((-1,50))

print(A.shape,B.shape)
print(A_m.shape,B_m.shape)
# %%
gsvdres = gsvd(A_m, B_m, full_matrices=True)
# %%
print(A)
#gsvdres.U @ gsvdres.U.T @ gsvdres.A @ gsvdres.X @ np.linalg.inv(gsvdres.X)
# %%
C = np.zeros_like(A_m)
np.fill_diagonal(C,gsvdres.c)

gsvdres.U @ C @ np.linalg.inv(gsvdres.X)
# %%
import matplotlib.pyplot as plt
plt.plot(gsvdres.c)
plt.plot(gsvdres.s)
plt.plot(gsvdres.s + gsvdres.c)
#gsvdres.V @ gsvdres.V.T @ B @ gsvdres.X @ np.linalg.inv(gsvdres.X)
# %%
S = np.zeros_like(B_m)
print(S.shape)
s_start = (gsvdres.s.shape[0])
np.fill_diagonal(S,gsvdres.s[-50+20:])
gsvdres.V @ S @ np.linalg.inv(gsvdres.X)
# %%
B
# %%
print(S)

# %%
