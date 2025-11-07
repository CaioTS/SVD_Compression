# %%
import numpy as np
from ctypes import CDLL, c_float, c_int, POINTER

import plotly.express as px
import plotly.graph_objects as go

from CantileverBeam import CantileverBeam
from Adaptive import FIRNLMS
from AdaptiveOO import FIR

# %% General Definitions:

fs = 100.0 # Sampling frequency in Hertz
pathmem = 400 # Number of samples for modeling the secondary and feedback paths
maxtimeformodeling = 100.0 # in seconds
nsteps = int(maxtimeformodeling * fs)

errwocompression = None # Variable to store the error signal without compression
errcompressed = None # Variable to store the compressed error signal

# %% Beam definitions:

npoints = 100 # Number of points in the beam (finite element method)
beamlength = 0.58 # Length of the beam in meters
beamwidth = 0.05 # Width of the beam in meters
beamthickness = 0.006 # Thickness of the beam in meters
dampingfactors = [0.015, 0.0023, 0.01, 0.01, 0.01] 

# Positions of sensors and forces:
perturbpos = 30 # Position of the perturbation force, which causes beam vibration.
referencepos = 75 # Position of the acceleration measurement at the beam.
controlpos = 60 # Position of the control force
errorpos = 95 # Position of the error acceleration measurement in the beam

cbeam = CantileverBeam(npoints=npoints, width=beamwidth, thickness=beamthickness, 
                        length=beamlength, Tsampling=1.0/fs,
                        damp=dampingfactors)
cbeam.reset()
print("Natural frequencies are:\n",
      ",\n".join(cbeam.freqsHz.astype(str).tolist()),
      " (all in Hz).")

xcoords = np.linspace(0.0, cbeam.length, npoints)
xcoords = np.concatenate((xcoords, xcoords[::-1]))
ycoords = np.array([0.0]*npoints + [beamthickness]*npoints)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xcoords, y=ycoords, fill='toself', mode='lines'))
fig.add_annotation(x=perturbpos*beamlength/npoints, y=beamthickness*1.1, 
            ax=0, ay=-30, text="Perturbation",
            showarrow=True, arrowhead=1)
fig.add_annotation(x=referencepos*beamlength/npoints, y=beamthickness*1.1, 
            ax=0, ay=-50, text="Accel. Measurement",
            showarrow=True, arrowhead=1, arrowside="start")
fig.add_annotation(x=controlpos*beamlength/npoints, y=0, 
            ax=0, ay=30, text="Control Force",
            showarrow=True, arrowhead=1)
fig.add_annotation(x=errorpos*beamlength/npoints, y=0, 
            ax=0, ay=50, text="Error Accel.",
            showarrow=True, arrowhead=1, arrowside="start")
fig.update_layout(title="Cantilever Beam", xaxis_title="x (m)", yaxis_title="y (m)")
fig.update_layout(xaxis=dict(range=[0, beamlength*1.1]), yaxis=dict(range=[-(beamthickness + 0.1), beamthickness + 0.1]))
fig.update_layout(width=600, height=350)
fig.show()


# %% Active control requires modeling both the secondary and feedback paths:
# The secondary path is the path from the control force to the error sensor.
# The feedback path is the path from the control force to the reference sensor.
# Modeling carried out using the FIRNLMS algorithm.

firnlms = FIRNLMS(memorysize=pathmem,stepsize=0.15,regularization=1e-3) # Create the FIRNLMS object

# Secondary path via adaptive modeling (the practical way):
cbeam.reset()
xrandom = np.random.randn(nsteps) # Random force vector
yerror = np.zeros(nsteps) # Error signal vector
for k in range(nsteps):
  cbeam.setforce(controlpos,xrandom[k]) # Force is applied at the control position
  cbeam.update() # Update the beam for 1 sampling period.
  yerror[k] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position

firnlms.run(insignal=xrandom,outsignal=yerror,maxiter=nsteps) # Run the FIRNLMS algorithm
wsecadaptive = firnlms.ww # Adaptive model of the secondary path

# Comparing the two methods:
fig = px.line()
fig.add_scatter(y=wsecadaptive, name="Adaptive model", mode="lines")
fig.update_layout(title="Secondary path response (FIR)")
fig.show() # Plot the secondary path coefficients


# %% FILTER COMPRESSION

# Matrix formatting with zero padding:
def format_W(W,R):
    n_pad = W.shape[0] % R
    W_pad = np.zeros(int(W.shape[0] + (R - n_pad)) if n_pad != 0 else W.shape[0])
    W_pad[:W.shape[0]] = W
    C = W_pad.shape[0]/R
    print("uepa: ",C)
    lower_dim = min(R,C)
    return  W_pad.reshape((int(lower_dim),-1),order = 'F')

# FIRSVD Implementation: 
class FIRSVDFilterPy(FIR):
    def __init__(self, C_weights, R_weights):
        self.R  =  R_weights.shape[1] # Rows
        self.C  =  C_weights.shape[1] # Collunms
        self.B  =  C_weights.shape[0] # Number of bases
        self.N = self.R*self.C
        self.vdot = C_weights        
        self.inputbuffer = np.zeros(self.N)
        self.util = []
        for k in range(B):
           self.util.append(FIR(R_weights[k,:]))
        self.reset()

    def reset(self):
        self.y = 0
        self.inputbuffer[:] = 0.0
        for k in range(self.B):
            self.util[k].reset()
            
    def filterstep(self, xn):
        self.inputbuffer[1:] = self.inputbuffer[:-1]
        self.inputbuffer[0] = xn
        self.youts = self.vdot @ self.inputbuffer[::self.R]
        for k in range(self.B):
            self.youts[k] = self.util[k].filterstep(self.youts[k])
        self.y = np.sum(self.youts)
        return self.y
    

# %% Secondary path compression:
C_chosen = 10
B = 3

print("Shape for wsecadaptive: ",wsecadaptive.shape)

W = format_W(wsecadaptive,C_chosen)
print(f'{W.shape = }')
R = W.shape[0]
U,S,VT = np.linalg.svd(W)

SM = np.zeros((R,C_chosen))
np.fill_diagonal(SM,S)
US = U @ SM
C_weights = np.zeros((B,VT.shape[1]))
R_weights = np.zeros((B,U.shape[0]))
print(f'{C_chosen = }\n',
      f'{R = }\n',
      f'{S.shape = }\n',
      f'{U.shape = }\n',
      f'{VT.shape = }\n')
for i in range(B):
    C_weights[i,:] = VT.T[:,i]
    R_weights[i,:] = US[:,i]

print(f'{C_weights.shape = }')
print(f'{R_weights.shape = }')
print(f'Total number of coefficients: {C_weights.size + R_weights.size} vs {wsecadaptive.size} ({100*(1 - (C_weights.size + R_weights.size)/wsecadaptive.size):.2f}% reduction)')

px.line(y=S, title='Singular values of the secondary path').show()

y = np.zeros(pathmem)
firsvdsec = FIRSVDFilterPy(C_weights, R_weights)
firsvdsec.reset()
y[0] = firsvdsec.filterstep(1.0)
for k in range(1,pathmem):
    y[k] = firsvdsec.filterstep(0.0)

fig = px.line(title='Impulse response from FIRSVDFilterPy')
fig.add_scatter(y=wsecadaptive, name="ideal", mode="lines")
fig.add_scatter(y=y, name="FIRSVDFilter2", mode="lines")
fig.show()

# %%
# print(f"{C = }, {R = }")
print("C shape:",C_weights.shape)
print("R shape:",R_weights.shape)
print(pathmem)
print("R = ",pathmem / C_chosen)
print(R)


# %%
# !g++ -fPIC -shared -o firsvd.so firsvd.cpp 

# %%
class firsvdc():

    # Wrapper for the C implementation of the FIRSVD class.
    def __init__(self,mem=400, B=3, R=10, C=41, ww=None):
        self.firsvdlib = CDLL("./firsvd.so")
        self.ww = np.ones(2000, dtype=np.float32) if ww is None else ww.astype(np.float32)
        self.xx = np.zeros(2000, dtype=np.float32)
        self.firsvdlib.createFilter.argtypes = [c_int, c_int, c_int, c_int, 
                                     POINTER(c_float), POINTER(c_float)]
        self.firsvdlib.createFilter.restype = None  # void return type
        self.firsvdlib.Filter.argtypes = [c_float]
        self.firsvdlib.Filter.restype = c_float
        self.firsvdlib.FilterWithFIR.argtypes = [c_float]
        self.firsvdlib.FilterWithFIR.restype = c_float

        # int mem, int nbranches, int nR, int nC, float *ptrw, float *ptrx
        self.firsvdlib.createFilter(c_int(mem), c_int(B), c_int(R), c_int(C), 
                                 self.ww.ctypes.data_as(POINTER(c_float)), 
                                 self.xx.ctypes.data_as(POINTER(c_float)))
        
    def filter(self, xin):
        yout = self.firsvdlib.Filter(c_float(xin))
        return yout

    def filter_with_fir(self, xin):
        yout = self.firsvdlib.FilterWithFIR(c_float(xin))
        return yout

# %%
wwaux = np.concatenate((C_weights.T.flatten(order='F'), R_weights.T.flatten(order='F')))
ww = np.zeros(2000, dtype=np.float32)
ww[:wwaux.size] = wwaux.astype(np.float32)

mycfirsvd = firsvdc(mem=pathmem, B=B, R=R_weights.shape[1], C=C_weights.shape[1], ww=ww)


# %%
y = np.zeros(pathmem)
y[0] = mycfirsvd.filter(1.0)
for k in range(1,pathmem):
    y[k] = mycfirsvd.filter(0.0) 

fig = px.line(title='Impulse response from FIRSVDFilterPy')
fig.add_scatter(y=wsecadaptive, name="ideal", mode="lines")
fig.add_scatter(y=y, name="C Implementation", mode="lines")
fig.show()

# %%
