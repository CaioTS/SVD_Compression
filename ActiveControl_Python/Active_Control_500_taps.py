# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from CantileverBeam import CantileverBeam
from Adaptive import FIRNLMS
from AdaptiveOO import FIRFxNLMS, FIR


# %%
fs = 416.0 # Sampling frequency in Hertz

# Beam characteristics:
npoints = 100 # Number of points in the beam (finite element method)
beamlength = 0.58 # Length of the beam in meters
beamwidth = 0.05 # Width of the beam in meters
beamthickness = 0.006 # Thickness of the beam in meters
dampingfactors = [0.01, 0.01, 0.01, 0.01, 0.01] 


# Positions of sensors and forces:
perturbpos = 30 # Position of the perturbation force, which causes beam vibration.
referencepos = 75 # Position of the acceleration measurement at the beam.
controlpos = 60 # Position of the control force
errorpos = 95 # Position of the error acceleration measurement in the beam

firmem = 500 # Number of samples for the secondary and feedback paths


# %% Creating Beam instance with 100 points:
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



# %% Simulating the beam response to a sinusoidal force:

maxtime = 60.0
vibstart = 10.0 # Start time of the vibration
nsteps = int(maxtime * fs) # Total number of steps
vibfreq = 10.0 # Hertz
th = np.linspace(0.0,maxtime,nsteps) # Time vector
xh = 0.3*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
xh[0:int(fs*vibstart)] = 0.0 # Force set to zero for the first 10 seconds

cbeam.reset()
err = np.zeros(nsteps) # Vibration response
# Running simulation:
for k in range(nsteps):
  err[k] = cbeam.getaccelms2(referencepos)  
  cbeam.setforce(perturbpos,xh[k]) 
  cbeam.update() # Updata for 1 sampling period.

# Plotting the results:
fig = px.line()
fig.add_scatter(x=th, y=xh, name="Força (N)", mode="lines")
fig.add_scatter(x=th, y=err, name="Aceleração (m/s²)", mode="lines")
fig.show()


# %% FILTER COMPRESSION

# Matrix formatting with zero padding:
def format_W(W,R):
    n_pad = W.shape[0] % R
    W_pad = np.zeros(int(W.shape[0] + (R - n_pad)))
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
        for k in range(self.B):
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

# %%
"""
- GENERATE THE 500 TAPS FOR WSEC AND WFBK
- GENERATE BEAM ACELLERATION GRAPH WITHOUT SVD
- CHANGE CONTROLLER FILTERS TO SVD TYPE (SO THAT THEY HAVE ALSO 500 COEFFICIENTTS)
- GENERATE BEAM ACELLEARITIO GRAPH WITH SVD 

"""
def gen_wsec_wfbk_filters(firmem):
  # Secondary path via impulse response (ideal but not practical):
  wsecimpulse = np.zeros(firmem) # Impulse response vector
  cbeam.reset()
  cbeam.setforce(controlpos,1.0) # Force is applied at the control position
  cbeam.update()
  wsecimpulse[0] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position
  cbeam.setforce(controlpos,0.0) # Force is removed
  for k in range(1,firmem):
    cbeam.update() # Update the beam for 1 sampling period.
    wsecimpulse[k] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position


  wfbkimpulse = np.zeros(firmem) # Impulse response vector
  cbeam.reset()
  cbeam.setforce(controlpos,1.0) # Force is applied at the control position
  cbeam.update()
  wfbkimpulse[0] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position
  cbeam.setforce(controlpos,0.0) # Force is removed
  for k in range(1,firmem):
    cbeam.update() # Update the beam for 1 sampling period.
    wfbkimpulse[k] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position

  fig = px.line()
  fig.add_scatter(y = wfbkimpulse)
  fig.add_scatter(y = wsecimpulse)
  
  return wsecimpulse,wfbkimpulse

# %%
"""
Calculate and organize filter weights for SVD
"""
firmem = 6400
wsec_impulse_6400 , wfbkimpulse_6400 = gen_wsec_wfbk_filters(firmem)
wsecimpulse = wsec_impulse_6400
wfbkimpulse = wfbkimpulse_6400

print("Shape for wsecimpulse: ",wsecimpulse.shape)
C_chosen_s = 60
B_s = 3

W = format_W(wsecimpulse,C_chosen_s)
print(f'{W.shape = }')
R = W.shape[0]
U,S,VT = np.linalg.svd(W)

SM = np.zeros((R,C_chosen_s))
np.fill_diagonal(SM,S)
US = U @ SM
C_weights = np.zeros((B_s,VT.shape[1]))
R_weights = np.zeros((B_s,U.shape[0]))
for i in range(B_s):
    C_weights[i,:] = VT.T[:,i]
    R_weights[i,:] = US[:,i]

print(f'Total number of coefficients: {C_weights.size + R_weights.size} vs {wsecimpulse.size} ({100*(1 - (C_weights.size + R_weights.size)/wsecimpulse.size):.2f}% reduction)')

px.line(y=S, title='Singular values of the secondary path').show()

y = np.zeros(firmem)
firsvdsec = FIRSVDFilterPy(C_weights, R_weights)
firsvdsec.reset()
y[0] = firsvdsec.filterstep(1.0)
for k in range(1,firmem):
    y[k] = firsvdsec.filterstep(0.0)

# Feedback path compression:

C_chosen_fb = 80
B_fb = 3
W = format_W(wfbkimpulse,C_chosen_fb)
print(f'{W.shape = }')
R = W.shape[0]
U,S,VT = np.linalg.svd(W)

SM = np.zeros((R,C_chosen_fb))
np.fill_diagonal(SM,S)
US = U @ SM
C_weightsfbk = np.zeros((B_fb,VT.shape[1]))
R_weightsfbk = np.zeros((B_fb,U.shape[0]))

for i in range(B_fb):
    C_weightsfbk[i,:] = VT.T[:,i]
    R_weightsfbk[i,:] = US[:,i]

print(f'Total number of coefficients: {C_weightsfbk.size + R_weightsfbk.size} vs {wfbkimpulse.size} ({100*(1 - (R_weightsfbk.size + R_weightsfbk.size)/wfbkimpulse.size):.2f}% reduction)')

px.line(y=S, title='Singular values of the feedback path').show()

y = np.zeros(firmem)
firsvdfbk = FIRSVDFilterPy(C_weightsfbk, R_weightsfbk)
firsvdfbk.reset()
y[0] = firsvdfbk.filterstep(1.0)
for k in range(1,firmem):
    y[k] = firsvdfbk.filterstep(0.0)

# %%
"""
General Parameters for simulation comparison
"""

mu = 0.004
force_amplitude = 1

"""
Run Filter without SVD 
"""
firmem = 500
wsecimpulse_500 , wfbkimpulse_500 = gen_wsec_wfbk_filters(500)

maxtime = 120.0
nsteps = int(maxtime * fs) # Total number of steps
vibstart = 0.0 # Start time of the vibration
controlstart = 30.0 # Start time of the control

controller = FIRFxNLMS(mem=300, memsec=firmem) # Create the controller
# controller.setSecondary(wsecimpulse) # Set the secondary path
controller.setSecondary(FIR(wsecimpulse_500)) # Set the secondary path
controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
controller.mu = mu # Set the step size
controller.psi = 1e-3 # Set the regularization parameter
controller.reset() # Reset the controller

feedbackfilter = FIR(wfbkimpulse_500) # Create the feedback filter
feedbackfilter.reset() # Reset the filter

vibfreq = 12.0 # Hertz
th = np.linspace(0.0,maxtime,nsteps) # Time vector
xh = force_amplitude*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
xh[0:int(fs*vibstart)] = 0.0 # Force is zero for the first 10 seconds

cbeam.reset()
err_500 = np.zeros(nsteps) # Vibration response
yfbk = np.zeros(nsteps) # Vibration response

# Running the simulation:
for k in range(nsteps):
  cbeam.setforce(perturbpos,xh[k]) # force is applied
  cbeam.setforce(controlpos,-controller.y) # Control force is applied

  if th[k] >= controlstart: # Control starts at 30 seconds
    controller.update(cbeam.getaccelms2(errorpos)) 
  yfbk[k] = feedbackfilter.filterstep(-controller.y) # Get the feedback force
  controller.evalout(cbeam.getaccelms2(referencepos) - yfbk[k])

  err_500[k] = cbeam.getaccelms2(errorpos) # Error acceleration is read

  cbeam.update() # beam is updated

# Plotting the results:
#fig = px.line()
#fig.add_scatter(x=th, y=xh, name="Perturbation force (N)", mode="lines")
#fig.add_scatter(x=th, y=err_500, name="Beam accelaration (m/s²)", mode="lines")
#fig.show()
#

# 
"""
Run SVD Filter
"""
firmem = 6400
controller = FIRFxNLMS(mem=firmem, memsec=firmem) # Create the controller
# controller.setSecondary(wsecimpulse) # Set the secondary path
firsvdsec.reset()
controller.setSecondary(FIRSVDFilterPy(C_weights,R_weights)) # Set the secondary path
controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
controller.mu = mu # Set the step size
controller.psi = 1e-3 # Set the regularization parameter
controller.reset() # Reset the controller

feedbackfilter = FIRSVDFilterPy(C_weightsfbk,R_weightsfbk)  # Create the feedback filter
feedbackfilter.reset() # Reset the filter

vibfreq = 12.0 # Hertz
th = np.linspace(0.0,maxtime,nsteps) # Time vector
xh = force_amplitude*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
xh[0:int(fs*vibstart)] = 0.0 # Force is zero for the first 10 seconds

cbeam.reset()
err_6400 = np.zeros(nsteps) # Vibration response
yfbk = np.zeros(nsteps) # Vibration response

# Running the simulation:
for k in range(nsteps):
  cbeam.setforce(perturbpos,xh[k]) # force is applied
  cbeam.setforce(controlpos,-controller.y) # Control force is applied

  if th[k] >= controlstart: # Control starts at 30 seconds
    controller.update(cbeam.getaccelms2(errorpos)) 
  yfbk[k] = feedbackfilter.filterstep(-controller.y) # Get the feedback force
  controller.evalout(cbeam.getaccelms2(referencepos) - yfbk[k])

  err_6400[k] = cbeam.getaccelms2(errorpos) # Error acceleration is read

  cbeam.update() # beam is updated
# Plotting the results:
fig = px.line()
fig.add_scatter(x=th, y=xh, name="Perturbation force (N)", mode="lines")
fig.add_scatter(x=th, y=err_500, name=f"Beam accelaration (m/s²) 500 taps mu = {mu}", mode="lines")
fig.add_scatter(x=th, y=err_6400, name=f"Beam accelaration (m/s²)500 taps SVD mu = {mu}", mode="lines")

def find_local_maxima(signal):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return np.array(peaks)

# For err_500
peaks_500 = find_local_maxima(err_500)
peaks_500_below = peaks_500[err_500[peaks_500] < 0.02]
fig.add_vline(x=th[peaks_500_below[7]], line_width=1, line_dash="dash", line_color="red")

# For err_6400
peaks_6400 = find_local_maxima(err_6400)
peaks_6400_below = peaks_6400[err_6400[peaks_6400] < 0.02]
fig.add_vline(x=th[peaks_6400_below[7]], line_width=1, line_dash="dash", line_color="blue")

fig.update_layout(
    legend=dict(
        orientation="h", # Horizontal legend
        yanchor="bottom",
        y=-0.3, # Adjust this value to move the legend further down
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=100) # Increase bottom margin if needed
)

fig.show()
print(f"Parameters: wsec (B/C) = ({B_s}/{C_chosen_s}) wfbk (B/C) = ({B_fb}/{C_chosen_fb}) | ")
print(f'WFBK: Total number of coefficients: {C_weightsfbk.size + R_weightsfbk.size} vs {wfbkimpulse.size} ({100*(1 - (R_weightsfbk.size + R_weightsfbk.size)/wfbkimpulse.size):.2f}% reduction)')
print(f'WSEC: Total number of coefficients: {C_weights.size + R_weights.size} vs {wsecimpulse.size} ({100*(1 - (C_weights.size + R_weights.size)/wsecimpulse.size):.2f}% reduction)')


# %%
