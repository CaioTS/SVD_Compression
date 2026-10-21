# %%
import numpy as np
import plotly.express as px
from scipy.linalg import convolution_matrix, inv, pinv
from scipy.signal.windows import triang

from ActVibModules.CantileverBeam import CantileverBeam
from ActVibModules.Adaptive import FIRNLMS


# %% Básico
h = np.sin(np.arange(0, 10, 0.1)*2) * np.exp(-np.arange(0, 10, 0.1)*0.3)
px.line(x=np.arange(len(h)), y=h)

# %%
g = np.array([0.5, 1.0, 0.5])
G = convolution_matrix(g, len(h), mode='same')

# W ótimo que minimiza ||h - G @ w||^2 é dado por w = (G^T G)^{-1} G^T h
w = inv(G.T @ G) @ G.T @ h
print(np.sum((h - G @ w)**2))

fig = px.line()
fig.add_scatter(y=h, mode='lines', name='h')
fig.add_scatter(y=G@w, mode='lines', name='w')
fig.show()


# %% Com w esparso
h = np.sin(np.arange(0, 10, 0.1)*2) * np.exp(-np.arange(0, 10, 0.1)*0.3)
L = 10
M = 2*L - 1
N = len(h)

mask = np.ones(N)
mask[1::L] = 0
P = np.diag(mask)

g = np.array(triang(M))

G = convolution_matrix(g, len(h), mode='same')

w_masked = pinv(G.T @ P @ G) @ G.T @ P @ h
print(np.sum((h - G @ w_masked)**2))

fig = px.line()
fig.add_scatter(y=h, mode='lines', name='h')
fig.add_scatter(y=G@w_masked, mode='lines', name='w_masked')
fig.show()


# %% Cascata alternante

h = np.sin(np.arange(0, 10, 0.1)*2) * np.exp(-np.arange(0, 10, 0.1)*0.3)

M = 20
N = len(h) - M + 1

g = np.zeros(M)
g[0] = 1.0
w = np.zeros_like(h)

for k in range(150):
    G = convolution_matrix(g, N, mode='full')
    w = inv(G.T @ G) @ G.T @ h
    print(np.sum((h - G @ w)**2))
    W = convolution_matrix(w, M, mode='full')
    g = inv(W.T @ W) @ W.T @ h
    print(np.sum((h - W @ g)**2))

fig = px.line()
fig.add_scatter(y=h, mode='lines', name='h')
fig.add_scatter(y=W@g, mode='lines', name='w_masked')
fig.show()




# %%
print(w.shape)
print(g.shape)

# %%
fs = 416

npoints = 100 # Number of points in the beam (finite element method)
beamlength = 0.58 # Length of the beam in meters
beamwidth = 0.05 # Width of the beam in meters
beamthickness = 0.006 # Thickness of the beam in meters
#dampingfactors = [0.015, 0.0023, 0.01, 0.01, 0.01] 
dampingfactors = [0.01, 0.0023, 0.01, 0.01, 0.01] 

# Positions of sensors and forces:
perturbpos = 30 # Position of the perturbation force, which causes beam vibration.
referencepos = 75 # Position of the acceleration measurement at the beam.
controlpos = 60 # Position of the control force
errorpos = 95 # Position of the error acceleration measurement in the beam

cbeam = CantileverBeam(npoints=npoints, width=beamwidth, thickness=beamthickness, 
                        length=beamlength, Tsampling=1.0/fs,
                        damp=dampingfactors, nmodes=5)
cbeam.reset()
print("Natural frequencies are:\n",
      ",\n".join(cbeam.freqsHz.astype(str).tolist()),
      " (all in Hz).")

x = np.zeros(1000) # Random noise as input
x[0] = 1.0 # Initial impulse
h = np.zeros_like(x) # Output of the system (acceleration at referencepos)
cbeam.reset()
for k in range(len(x)):
    h[k] = cbeam.setforce(controlpos, x[k])    
    h[k] = cbeam.getaccel(referencepos)
    cbeam.update()
    
px.line(x=np.arange(len(h)), y=h, title="Impulse response at reference position").show()

print(cbeam.freqsHz)
taus = 1.0 / (cbeam.freqsHz * 2 * np.pi * cbeam.zeta)
print(taus)
print("3 x taus:", 3*taus)
print("3 x taus in samples:", (3*taus * fs).astype(int))




# %%
L = 5
M = 2*L - 1
N = len(h) - M + 1

mask = np.zeros(N)
mask[0::L] = 1
P = np.diag(mask)

g = np.array(triang(M))

G = convolution_matrix(g, N, mode='full')

w_masked = P @ pinv(P @ G.T @ G @ P) @ P @ G.T @ h
print((np.sum((h - G @ P @ w_masked)**2)))

fig = px.line()
fig.add_scatter(y=h, mode='lines', name='h')
fig.add_scatter(y=G@w_masked, mode='lines', name='w_masked')
fig.show()

# %%

L = 2
M = 2*L - 1
N = len(h) - M + 1

mask = np.zeros(N)
mask[0::L] = 1
P = np.diag(mask)

# g = np.random.randn(M)/10
# g = np.array(triang(M))
g = np.zeros(M)
g[0] = 1.0

for k in range(10):
    G = convolution_matrix(g, N, mode='full')    
    w_masked = P @ pinv(P @ G.T @ G @ P) @ P @ G.T @ h
    #w_new = P @ pinv(P @ G.T @ G @ P) @ P @ G.T @ h
    #w_masked = (w_new + w_masked) / 2
    print((np.sum((h - G @ w_masked)**2)))
    W = convolution_matrix(w_masked, M, mode='full')
    g = inv(W.T @ W) @ W.T @ h
    #g_new = inv(W.T @ W) @ W.T @ h
    #g = (g_new + g) / 2
    print(np.sum((h - W @ g)**2))

fig = px.line()
fig.add_scatter(y=h, mode='lines', name='h')
fig.add_scatter(y=W @ g, mode='lines', name='w_masked')
fig.show()
# %%
L = 2
M = 2*L - 1
N = len(h)

mask = np.zeros(N)
mask[0::L] = 1
P = np.diag(mask)

# g = np.random.randn(M)/10
# g = np.array(triang(M))
g = np.zeros(M)
g[0] = 1.0

for k in range(10):
    G = convolution_matrix(g, N, mode='same')    
    w_masked = P @ pinv(P @ G.T @ G @ P) @ P @ G.T @ h
    #w_new = P @ pinv(P @ G.T @ G @ P) @ P @ G.T @ h
    #w_masked = (w_new + w_masked) / 2
    print((np.sum((h - G @ w_masked)**2)))
    W = convolution_matrix(w_masked, M, mode='same')
    g = inv(W.T @ W) @ W.T @ h
    #g_new = inv(W.T @ W) @ W.T @ h
    #g = (g_new + g) / 2
    print(np.sum((h - W @ g)**2))

fig = px.line()
fig.add_scatter(y=h, mode='lines', name='h')
fig.add_scatter(y=W @ g, mode='lines', name='w_masked')
fig.show()
# %%
