#%%
#File Created to test SVDs implentation in Python.

#   Objective is to create a random matrix and use SVD to make approximates matrix on.

#   Create an X array of 10x1 to multiply by the H that will pass trough SVD

#   Calculate error for Differents Ranks used in ~H.

#   Adapt calculation of matrix to optimize efficiency by adding X in the calculation instead.(Simulate Layer operation in Neural Network)


#%%
#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
#%%
H = np.random.rand(10,10)
X = np.random.rand(10,1)

Y = H @ X # Y is 10 x 1

U , S , VT = np.linalg.svd(H)

print(f"U size  = {U.shape}")
print(f"S size  = {S.shape}")
print(f"VT size = {VT.shape}")
print(f"S = {S}")

#%%

#Check if Reconstruction of H is perfect
S_matrix = np.zeros_like(H, dtype=float)
np.fill_diagonal(S_matrix, S)

H_recon = U @ S_matrix @ VT

print(sum(sum(abs(H - H_recon)))) #Absolute Sum of Error between reconstruction and original H
print(len(S))

#%%
error = []
error_Y = []
for i in range(len(S)) :
    U_r = np.delete(U,[range(len(S)-i,len(S),1)], axis=1)
    VT_r  = np.delete(VT,[range(len(S)-i,len(S),1)], axis=0)
    S_r = S if i == 0 else S[:-i] 
    S_r_matrix = np.zeros((len(S) - i,len(S) - i))
    np.fill_diagonal(S_r_matrix, S_r)
    #print(f"Sizes U ({U_r.shape}) VT ({VT_r.shape}) S ({S_r_matrix.shape})")
    H_r = U_r @ S_r_matrix @ VT_r
    error.append(sum(sum(abs(H - H_r))))
    error_Y.append(sum(sum(abs(Y - H_r @ X))))
    side_by_side = np.hstack((Y.reshape(-1, 1), (H_r @ X).reshape(-1, 1)))

plt.plot(error_Y)
plt.xlabel("Rank")
plt.ylabel("Absolute Difference")
# %%

U , S , VT = np.linalg.svd(H)

Y_test = np.zeros((Y.shape[0],Y.shape[1]))
H_test = np.zeros_like(H)
print(U.shape,VT.shape,S.shape, X.shape)
US = np.zeros_like(U)

for i in range(len(S)):
    temp = S[i] * np.outer(U[:,i] , VT[i,:])
    H_test += temp

plt.plot(H - H_test)
# %%
U , S , VT = np.linalg.svd(H)

Y_test = np.zeros((Y.shape[0],Y.shape[1]))
H_test = np.zeros_like(H)
print(U.shape,VT.shape,S.shape, X.shape)
US = np.zeros_like(U)
for i in range(len(S)):
    US[:,i] = U[:,i]*S[i]

for i in range(len(S)):
    VTX = VT[i,:] @ X
    temp = np.outer(US[:,i] , VTX)
    Y_test += temp

plt.plot(Y - Y_test)
# %%
# %%
U , S , VT = np.linalg.svd(H)

Y_test = np.zeros((Y.shape[0],Y.shape[1]))
H_test = np.zeros_like(H)
print(U.shape,VT.shape,S.shape, X.shape)
US = np.zeros_like(U)
for i in range(len(S)):
    US[:,i] = U[:,i]*S[i]

for i in range(len(S)):
    VTX = VT[i,:] @ X
    temp = np.outer(US[:,i] , VTX)
    Y_test += temp

plt.plot(Y - Y_test)
# %%
#Plotar Número de Operações do método original e com somatória
import matplotlib.pyplot as plt

H = np.random.rand(784,300)
X = np.random.rand(300,1)

m,n = H.shape
l = X.shape[1]

n_sum = []
n_original = []

compression = []
for k in range(n):
    n_sum.append(k * (m + (2*n -1))+ k*m)
    n_original.append( m*l*(2*n -1))

compression.append(np.array(n_original)/np.array(n_sum))

plt.plot(n_sum)
plt.plot(n_original)
plt.title(f"Numero de Operações H({m}x{n}) @ X({n}x{l})")

# %%

H = np.random.rand(300,100)
X = np.random.rand(100,1)

m,n = H.shape
l = X.shape[1]

n_sum = []
n_original = []
for k in range(n):
    n_sum.append(k * (m + (2*n -1))+ k*m)
    n_original.append( m*l*(2*n -1))

compression.append(np.array(n_original)/np.array(n_sum))

plt.plot(n_sum)
plt.plot(n_original)
plt.title(f"Numero de Operações H({m}x{n}) @ X({n}x{l})")

# %%

H = np.random.rand(100,10)
X = np.random.rand(10,1)

m,n = H.shape
l = X.shape[1]

n_sum = []
n_original = []
for k in range(n):
    n_sum.append(k * (m + (2*n -1)) + k*m)
    n_original.append( m*l*(2*n -1))

compression.append(np.array(n_original)/np.array(n_sum))

plt.plot(n_sum)
plt.plot(n_original)
plt.title(f"Numero de Operações H({m}x{n}) @ X({n}x{l})")

# %%
fig, axes = plt.subplots(3, 1, figsize=(8, 8))

titles = ["H(784x300) @ X(300x1)","H(300x100) @ X(100x1)","H(100x10) @ X(10x1)"]

for i in range(3):
    axes[i].plot(compression[i])
    axes[i].set_title(f'Compression {titles[i]}')

plt.tight_layout()
plt.show()
# %%

def calc_compression_per_layer(H,X):
    m,n = H.shape   
    l = X.shape[1]

    n_sum = []
    n_original = []
    compression = []
    for k in range(n):
        n_sum.append(k * (m + (2*n -1)) + k*m)
        n_original.append( m*l*(2*n -1))

    compression.append(np.array(n_original)/np.array(n_sum))

    return compression

#%%
import numpy as np
import matplotlib.pyplot as plt

H = np.random.rand(784,300)
X = np.random.rand(300,1)

U , S , VT = np.linalg.svd(H)


#%%
import numpy as np
X = np.zeros((1,256))
VT = np.zeros((256,1))
for i in range(256):
    X[0,i] = 65537
    VT[i,0] = 131072

in_prod = X @ VT
print(
    in_prod/2**32
)

for i in range(256):
    X[0,i] = i * 1024 * 1024
    VT[i,0] = i * 1024 * 1024

in_prod = X @ VT
print(
    in_prod/2**32 - 1423278080
)
# %%
