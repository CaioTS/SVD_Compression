# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from CantileverBeam import CantileverBeam
from Adaptive import FIRNLMS
from AdaptiveOO import FIRFxNLMS, FIR

# %% General Definitions:

fs = 100.0 # Sampling frequency in Hertz

pathmem = 400 # Number of samples for modeling the secondary and feedback paths
maxtimeformodeling = 100.0 # in seconds
nsteps = int(maxtimeformodeling * fs)

# General controller definitions:
controllermem = 300


# Simulation definitions:
maxtimeformodeling = 120.0
vibstart = 0.0 # Start time for the vibration
controlstart = 30.0 # Start time for the control
vibfreq = 14.5 # Hertz
vibampl = 1.0

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

# Secondary path via impulse response (ideal but not practical):
wsecimpulse = np.zeros(pathmem) # Impulse response vector
cbeam.reset()
cbeam.setforce(controlpos,1.0) # Force is applied at the control position
cbeam.update()
wsecimpulse[0] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position
cbeam.setforce(controlpos,0.0) # Force is removed
for k in range(1,pathmem):
  cbeam.update() # Update the beam for 1 sampling period.
  wsecimpulse[k] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position

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
fig.add_scatter(y=wsecimpulse, name="Impulse response", mode="lines")
fig.add_scatter(y=wsecadaptive, name="Adaptive model", mode="lines")
fig.update_layout(title="Secondary path response (FIR)")
fig.show() # Plot the secondary path coefficients

# %% Feedback path via impulse response (ideal but not practical):

wfbkimpulse = np.zeros(pathmem) # Impulse response vector
cbeam.reset()
cbeam.setforce(controlpos,1.0) # Force is applied at the control position
cbeam.update()
wfbkimpulse[0] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position
cbeam.setforce(controlpos,0.0) # Force is removed
for k in range(1,pathmem):
  cbeam.update() # Update the beam for 1 sampling period.
  wfbkimpulse[k] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position

# Secondary path via adaptive modeling (the practical way):
cbeam.reset()
xrandom = np.random.randn(nsteps) # Random force vector
yerror = np.zeros(nsteps) # Error signal vector
for k in range(nsteps):
  cbeam.setforce(controlpos,xrandom[k]) # Force is applied at the control position
  cbeam.update() # Update the beam for 1 sampling period.
  yerror[k] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position

firnlms.run(insignal=xrandom,outsignal=yerror,maxiter=nsteps) # Run the FIRNLMS algorithm
wfbkadaptive = firnlms.ww # Adaptive model of the secondary path

# Comparing the two methods:
fig = px.line()
fig.add_scatter(y=wfbkimpulse, name="Impulse response", mode="lines")
fig.add_scatter(y=wfbkadaptive, name="Adaptive model", mode="lines")
fig.update_layout(title="Feedback path response")
fig.show() # Plot the secondary path coefficients



# %% Now, after obtaining the secondary and feedback responses,
# the active control using the FIRFxNLMS algorithm can be performed:
stdpathmem = 40
stdmu = 0.006
stdfi = 1e-3

nsteps = int(maxtimeformodeling * fs) # Total number of steps

controller = FIRFxNLMS(mem=controllermem, memsec=stdpathmem) # Create the controller
# controller.setSecondary(wsecimpulse) # Set the secondary path
controller.setSecondary(FIR(wsecadaptive[:stdpathmem])) # Set the secondary path
controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
controller.mu = stdmu # Set the step size
controller.fi = stdfi # Set the regularization parameter
controller.reset() # Reset the controller

feedbackfilter = FIR(wfbkadaptive[:stdpathmem]) # Create the feedback filter
feedbackfilter.reset() # Reset the filter


th = np.linspace(0.0,maxtimeformodeling,nsteps) # Time vector
xh = vibampl*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
xh[0:int(fs*vibstart)] = 0.0 # Force is zero for the first 10 seconds

cbeam.reset()
err = np.zeros(nsteps) # Vibration response
yfbk = np.zeros(nsteps) # Vibration response

# Running the simulation:
for k in range(nsteps):
  cbeam.setforce(perturbpos,xh[k]) # force is applied
  cbeam.setforce(controlpos,-controller.y) # Control force is applied

  if th[k] >= controlstart: # Control starts at 30 seconds
    controller.update(cbeam.getaccelms2(errorpos)) 
  yfbk[k] = feedbackfilter.filterstep(-controller.y) # Get the feedback force
  controller.evalout(cbeam.getaccelms2(referencepos) - yfbk[k])

  err[k] = cbeam.getaccelms2(errorpos) # Error acceleration is read

  cbeam.update() # beam is updated

errwocompression = err.copy()

# Plotting the results:
fig = px.line()
fig.add_scatter(x=th, y=xh**2, name="Perturbation force (N)", mode="lines")
fig.add_scatter(x=th, y=errwocompression**2, name="Vib. w.o. compression (m/s²)", mode="lines")
if errcompressed is not None:
  fig.add_scatter(x=th, y=errcompressed**2, name="Vib. with compression (m/s²)", mode="lines")
fig.update_xaxes(range=[controlstart-2, controlstart+20])
fig.show()



# %% FILTER COMPRESSION

# Matrix formatting with zero padding:
def format_W(W,R):
    n_pad = W.shape[0] % R
    W_pad = np.zeros(int(W.shape[0] + (R - n_pad)) if n_pad != 0 else W.shape[0])
    W_pad[:W.shape[0]] = W
    C = W_pad.shape[0]/R
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
C_chosen = 20
B = 1

print("Shape for wsecimpulse: ",wsecimpulse.shape)

W = format_W(wsecimpulse,C_chosen)
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
print(f'Total number of coefficients: {C_weights.size + R_weights.size} vs {wsecimpulse.size} ({100*(1 - (C_weights.size + R_weights.size)/wsecimpulse.size):.2f}% reduction)')

px.line(y=S, title='Singular values of the secondary path').show()

y = np.zeros(pathmem)
firsvdsec = FIRSVDFilterPy(C_weights, R_weights)
firsvdsec.reset()
y[0] = firsvdsec.filterstep(1.0)
for k in range(1,pathmem):
    y[k] = firsvdsec.filterstep(0.0)

fig = px.line(title='Impulse response from FIRSVDFilterPy')
fig.add_scatter(y=wsecimpulse, name="ideal", mode="lines")
fig.add_scatter(y=y, name="FIRSVDFilter2", mode="lines")
fig.show()

# %% Feedback path compression:
C_chosen = 20
B = 1

print("Shape for wfbkimpulse: ",wfbkimpulse.shape)

W = format_W(wfbkimpulse,C_chosen)
print(f'{W.shape = }')
R = W.shape[0]
U,S,VT = np.linalg.svd(W)

SM = np.zeros((R,C_chosen))
np.fill_diagonal(SM,S)
US = U @ SM
C_weightsfbk = np.zeros((B,VT.shape[1]))
R_weightsfbk = np.zeros((B,U.shape[0]))
print(f'{C_chosen = }\n',
      f'{R = }\n',
      f'{S.shape = }\n',
      f'{U.shape = }\n',
      f'{VT.shape = }\n')
for i in range(B):
    C_weightsfbk[i,:] = VT.T[:,i]
    R_weightsfbk[i,:] = US[:,i]

print(f'{C_weightsfbk.shape = }')
print(f'{R_weightsfbk.shape = }')
print(f'Total number of coefficients: {C_weightsfbk.size + R_weightsfbk.size} vs {wfbkimpulse.size} ({100*(1 - (R_weightsfbk.size + R_weightsfbk.size)/wfbkimpulse.size):.2f}% reduction)')

px.line(y=S, title='Singular values of the feedback path').show()

y = np.zeros(pathmem)
firsvdfbk = FIRSVDFilterPy(C_weightsfbk, R_weightsfbk)
firsvdfbk.reset()
y[0] = firsvdfbk.filterstep(1.0)
for k in range(1,pathmem):
    y[k] = firsvdfbk.filterstep(0.0)

fig = px.line(title='Impulse response from FIRSVDFilterPy')
fig.add_scatter(y=wfbkimpulse, name="ideal", mode="lines")
fig.add_scatter(y=y, name="FIRSVDFilter2", mode="lines")
fig.show()

fig = px.line(title='C basis vectors')
fig2 = px.line(title='R basis vectors')
for nb in range(B):
    fig.add_scatter(y=C_weightsfbk[nb,:], name=f"Basis {nb+1}", mode="lines") 
    fig2.add_scatter(y=R_weightsfbk[nb,:], name=f"Basis {nb+1}", mode="lines")
fig.show()
fig2.show()

# %%
cprmu = 0.01
cprfi = 1e-3

nsteps = int(maxtimeformodeling * fs) # Total number of steps

controller = FIRFxNLMS(mem=controllermem, memsec=pathmem) # Create the controller
# controller.setSecondary(wsecimpulse) # Set the secondary path
firsvdsec.reset()
controller.setSecondary(firsvdsec) # Set the secondary path
controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
controller.mu = cprmu # Set the step size
controller.fi = cprfi # Set the regularization parameter
controller.reset() # Reset the controller

# feedbackfilter = FIR(wfbkimpulse) # Create the feedback filter
feedbackfilter = firsvdfbk
feedbackfilter.reset() # Reset the filter

th = np.linspace(0.0,maxtimeformodeling,nsteps) # Time vector
xh = 1.0*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
xh[0:int(fs*vibstart)] = 0.0 # Force is zero for the first 10 seconds

cbeam.reset()
err = np.zeros(nsteps) # Vibration response
yfbk = np.zeros(nsteps) # Vibration response

# Running the simulation:
for k in range(nsteps):
  cbeam.setforce(perturbpos,xh[k]) # force is applied
  cbeam.setforce(controlpos,-controller.y) # Control force is applied

  if th[k] >= controlstart: # Control starts at 30 seconds
    controller.update(cbeam.getaccelms2(errorpos)) 
  yfbk[k] = feedbackfilter.filterstep(-controller.y) # Get the feedback force
  controller.evalout(cbeam.getaccelms2(referencepos) - yfbk[k])

  err[k] = cbeam.getaccelms2(errorpos) # Error acceleration is read

  cbeam.update() # beam is updated

errcompressed = err.copy()

# Plotting the results:
fig = px.line()
# fig.add_scatter(x=th, y=xh, name="Perturbation force (N)", mode="lines")
if errwocompression is not None:
  fig.add_scatter(x=th, y=errwocompression**2, name="Vib. w.o. compr. (m/s²)", mode="lines")
fig.add_scatter(x=th, y=errcompressed**2, name="Vib. with compr. (m/s²)", mode="lines")
fig.update_xaxes(range=[controlstart-2, controlstart+50])
fig.show()

print(np.sum((err-errwocompression)**2))

# %%
rp = (errcompressed[int((controlstart-10)*fs):int(controlstart*fs)]**2).max()
nivel2pct = rp*0.02
nivel5pct = rp*0.05
print("\nNivel 2%: ",nivel2pct,"-------------------------------")
ondewoc = np.where(errwocompression[int(controlstart*fs):]**2 > nivel2pct)
print("Tau 2% w.o. compr.",ondewoc[0][-1]/fs)
ondewc = np.where(errcompressed[int(controlstart*fs):]**2 > nivel2pct)
print("Tau 2% with compr.",ondewc[0][-1]/fs)

print("\nNivel 5%: ",nivel5pct,"-------------------------------")
ondewoc5 = np.where(errwocompression[int(controlstart*fs):]**2 > nivel5pct)
print("Tau 5% w.o. compr.",ondewoc5[0][-1]/fs)
ondewc5 = np.where(errcompressed[int(controlstart*fs):]**2 > nivel5pct)
print("Tau 5% with compr.",ondewc5[0][-1]/fs)


# %% FILTER COMPRESSION WITH SHARED BASIS


# %% Let's try to use the basis from WK in WS:
print("Shape for wfbkimpulse: ",wfbkimpulse.shape)
C_chosen = 25
B = 3

W = format_W(wfbkimpulse,C_chosen)
print(f'{W.shape = }')
R = W.shape[0]
U,S,VT = np.linalg.svd(W)

SM = np.zeros((R,C_chosen))
np.fill_diagonal(SM,S)
US = U @ SM
C_weightsfbk = np.zeros((B,VT.shape[1]))
R_weightsfbk = np.zeros((B,U.shape[0]))
print(f'{C_chosen = }\n',
      f'{R = }\n',
      f'{S.shape = }\n',
      f'{U.shape = }\n',
      f'{VT.shape = }\n')
for i in range(B):
    C_weightsfbk[i,:] = VT.T[:,i]
    R_weightsfbk[i,:] = US[:,i]

print(f'{C_weightsfbk.shape = }')
print(f'{R_weightsfbk.shape = }')
print(f'Total number of coefficients: {C_weightsfbk.size + R_weightsfbk.size} vs {wfbkimpulse.size} ({100*(1 - (R_weightsfbk.size + R_weightsfbk.size)/wfbkimpulse.size):.2f}% reduction)')

px.line(y=S, title='Singular values of the feedback path').show()

y = np.zeros(pathmem)
firsvdfbk = FIRSVDFilterPy(C_weightsfbk, R_weightsfbk)
firsvdfbk.reset()
y[0] = firsvdfbk.filterstep(1.0)
for k in range(1,pathmem):
    y[k] = firsvdfbk.filterstep(0.0)

fig = px.line(title='Impulse response from FIRSVDFilterPy')
fig.add_scatter(y=wfbkimpulse, name="ideal", mode="lines")
fig.add_scatter(y=y, name="FIRSVDFilter2", mode="lines")
fig.show()

# %%
print(C_weightsfbk.shape)
fig = px.line(title='Feedback path basis vectors')
for k in range(B):
    fig.add_scatter(y=(C_weightsfbk[k,:]))
fig.show()

# %%
Wsec = format_W(wsecimpulse,C_chosen)
USsec = Wsec @ np.linalg.inv(VT)
px.line(y=(USsec**2).sum(axis=0), title='Energy of U branchs for the secondary path').show()

C_weightssec = np.zeros((B,VT.shape[1]))
R_weightssec = np.zeros((B,U.shape[0]))
for i in range(B):
    C_weightssec[i,:] = VT.T[:,i]
    R_weightssec[i,:] = USsec[:,i]

print(f'{C_weightssec.shape = }')
print(f'{R_weightssec.shape = }')
print(f'Total number of coefficients: {C_weightssec.size + R_weightssec.size} vs {wsecimpulse.size} ({100*(1 - (C_weightssec.size + R_weightssec.size)/wsecimpulse.size):.2f}% reduction)')


y = np.zeros(pathmem)
firsvdsec = FIRSVDFilterPy(C_weightssec, R_weightssec)
firsvdsec.reset()
y[0] = firsvdsec.filterstep(1.0)
for k in range(1,pathmem):
    y[k] = firsvdsec.filterstep(0.0)

fig = px.line(title='Impulse response from FIRSVDFilterPy')
fig.add_scatter(y=wsecimpulse, name="ideal", mode="lines")
fig.add_scatter(y=y, name="FIRSVDFilter2", mode="lines")
fig.show()



# %%
print("Shape for wfbkimpulse: ",wfbkimpulse.shape)
print("Shape for wsecimpulse: ",wsecimpulse.shape)

WK = format_W(wfbkimpulse,50)
WS = format_W(wsecimpulse,50)

Uk,Sk,VkT = np.linalg.svd(WK)
Us,Ss,VsT = np.linalg.svd(WS)

fig = px.line()
fig.add_scatter(y=Sk, name="Feedback path", mode="lines")
fig.add_scatter(y=Ss, name="Secondary path", mode="lines")
fig.update_layout(title="Singular values comparison")
fig.show()


# %%
X = np.linalg.inv(Uk) @ WS
Xen = np.sqrt((X**2).sum(axis=0))

fig = px.line()
fig.add_scatter(y=Xen, name="Energy projection of secondary path on feedback path", mode="lines")
fig.add_scatter(y=Ss, name="Secondary path singular values", mode="lines")
fig.update_layout(title="Energy projection of secondary path on feedback path")
fig.show()


# %%
print(X.shape)
print((np.diag(Ss) @ VsT).shape)
# %%
print(np.diag(Ss).shape)
print(VsT.shape)
# %%
