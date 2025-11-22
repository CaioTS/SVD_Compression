# %%
import numpy as np
import plotly.express as px
from scipy import signal

from ActVibModules.ActVibSystem import ActVibData
from ActVibModules.DSPFuncs import easyFourier
from Adaptive import FIRNLMS

# %% General Definitions:

data = ActVibData("test_1911.feather")
print(data.columns)

# %%

fnlms = FIRNLMS(4000,0.05,1e-2,wwavgwindow=4000)
fnlms.run(data['ctrl'].values,data['err'].values)
px.line(fnlms.wwavg).show()

# %%
mag,freq = easyFourier(fnlms.wwavg,fs=416)

picos = signal.find_peaks(mag,height=-100,distance=25,width=20)
hts = picos[1]['peak_heights']
picos = picos[0]
print(picos)

fig = px.line()
fig.add_scatter(x=freq,y=mag,mode='lines',name='Magnitude')
fig.add_scatter(x=freq[picos],y=hts,mode='markers',name='Peaks')
fig.show()

# %%
halfhts = hts - 3 # -3 dB
limssup = []
limsinf = []

print(halfhts)
for pico in picos:
    print(mag[pico-20:pico+20])
    # Find upper limit
    for i in range(pico,len(mag)):
        if mag[i] <= halfhts[np.where(picos==pico)[0][0]]:
            limssup.append(i)
            break
    # Find lower limit
    for i in range(pico,0,-1):
        if mag[i] <= halfhts[np.where(picos==pico)[0][0]]:
            limsinf.append(i)
            break

limssup = np.array(limssup)
limsinf = np.array(limsinf)

print(picos)
print(limssup)
print(limsinf)

freqssup = freq[limssup]
freqsinf = freq[limsinf]
print(freqssup)
print(freqsinf)

amort = (freqssup - freqsinf) / (2 * freq[picos])
print(amort)

# %%
