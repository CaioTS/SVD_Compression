# %% 
import numpy as np
import polars as pl
import plotly.express as px
from scipy import signal
from ActVibModules.Adaptive import FIRNLMS
from ActVibModules.DSPFuncs import easyFourier
from scipy.linalg import convolution_matrix, inv, pinv

# %%
data = pl.read_ipc("dados/R1_A1_0.75.feather")
data.show(10)

# %%
x = data["dac1"].to_numpy()
d = data["imu1accz"].to_numpy()
d = d - np.mean(d)

fnlms = FIRNLMS(1000,0.1,1e-2)
fnlms.run(x, d)

px.line(fnlms.ww).show()

magww,freqww = easyFourier(fnlms.ww, fs=416)
px.line(x=freqww, y=10**(magww/20)).show()

ww = fnlms.ww

# %% Amortecimento
picos = signal.find_peaks(magww,height=-40,distance=25,width=10)
hts = picos[1]['peak_heights'][0:3]
picos = picos[0][0:3]
print(hts)
print(picos)

hts3db = hts[0:3]-3
finf = []
fsup = []
fdelta = 0
for k in range(hts3db.shape[0]):
    finf.append( freqww[(magww >= hts3db[k]) & (freqww > fdelta)][0] )
    fdelta = finf[-1]
    fsup.append( freqww[(magww <= hts3db[k]) & (freqww > fdelta)][0] )
    fdelta = fsup[-1]
print(finf)
print(fsup)

amort = []
taus = []
for k in range(hts3db.shape[0]):
    amort.append( (fsup[k]-finf[k])/(2*freqww[picos[k]]) )
    taus.append(1/(2*np.pi*amort[k]*freqww[picos[k]]))
print(f"Amortecimentos: {amort}")
print(f"Taus: {taus} s")

print(f"Amostras para tau: {np.array(taus)*416} amostras")

# %%
# Matrix formatting with zero padding:
def format_W(W,R):
    n_pad = W.shape[0] % R
    W_pad = np.zeros(int(W.shape[0] + (R - n_pad)))
    W_pad[:W.shape[0]] = W
    C = W_pad.shape[0]/R
    lower_dim = min(R,C)
    return  W_pad.reshape((int(lower_dim),-1),order = 'F')

# %%
WW = format_W(ww, 45)
print(WW.shape)

U, s, Vt = np.linalg.svd(WW)
print(U.shape)
print(Vt.shape)

fig = px.line()
fig2 = px.line()
for k in range(10):
    vt2 = (Vt[k,:]*s[k])
    print(f"Singular value {k}: {s[k]} - Energy: {np.max(vt2)}")
    fig.add_scatter(x=np.arange(vt2.shape[0])*10, y=vt2, mode='lines', name=f"Vt[{k},:]**2")
    mag,freq = easyFourier(Vt[k,:], fs=416)
    fig2.add_scatter(x=freq, y=mag, mode='lines', name=f"Vt[{k},:]")
fig.show()
fig2.show()

print(s)

# %%
totenergy = []
energyn = []
N = 6
ks = range(N,np.sqrt(ww.shape[0]).astype(int)*2)
compl = []
for k in ks:
    WW = format_W(ww, k)
    U, s, Vt = np.linalg.svd(WW)
    totenergy.append(np.sum(s**2))
    energy = np.sum(s[0:N]**2)
    energyn.append(energy)
    compl.append(U.shape[0] + Vt.shape[0])

fig = px.line()
fig.add_scatter(x=np.array(ks), y=np.array(totenergy), mode='lines', name='Total energy')
fig.add_scatter(x=np.array(ks), y=np.array(energyn), mode='lines', name=f"First {N} sv")
fig.show()

fig2 = px.line()
fig2.add_scatter(x=np.array(ks), y=np.array(compl*N), mode='lines', name='Complexity')
fig2.add_scatter(x=np.array(ks), y=ww.shape[0]*np.ones(len(ks)), mode='lines', name='Ref')
fig2.show()

# %%
h = ww

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

for k in range(1):
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

magh,freqh = easyFourier(h, fs=416)
magwg,freqwg = easyFourier(W @ g, fs=416)
fig2 = px.line()
fig2.add_scatter(x=freqh, y=magh, mode='lines', name='h')
fig2.add_scatter(x=freqwg, y=magwg, mode='lines', name='W @ g')
fig2.show()
# %%
