# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from CantileverBeam import CantileverBeam
from Adaptive import FIRNLMS
from AdaptiveOO import FIRFxNLMS, FIR


# %%
fs = 100.0 # Sampling frequency in Hertz

# Beam characteristics:
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

firmem = 200 # Number of samples for the secondary and feedback paths


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
vibfreq = 14.5 # Hertz
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

# Matrix formatting with zero padding:
def format_W_raw(W,R):
    n_pad = W.shape[0] % R
    W_pad = np.zeros(int(W.shape[0] + (R - n_pad)))
    W_pad[:W.shape[0]] = W
    C = W_pad.shape[0]/R
    return  W_pad.reshape(R,-1,order = 'F')


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

def find_local_maxima(signal):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return np.array(peaks)
# %%
"""
Calculate and organize filter weights for SVD
If cut the filtes in V in  50% , i can add how much branches?



"""

import matplotlib.pyplot as plt

firmem = 1000
wsec_impulse_6400 , wfbkimpulse_6400 = gen_wsec_wfbk_filters(firmem)
wsecimpulse = wsec_impulse_6400
wfbkimpulse = wfbkimpulse_6400

print("Shape for wsecimpulse: ",wsecimpulse.shape)
C_chosen_s = 34
B_s = 3

C_chosen_s_cut = 18
B_s_cut = 4
#W = format_W(wsecimpulse,C_chosen_s)
W = format_W(wsecimpulse,C_chosen_s)
print(f'{W.shape = }')
R = W.shape[0]
U,S,VT = np.linalg.svd(W)

SM = np.zeros((R,C_chosen_s))

SM = np.zeros((R,C_chosen_s))
min_len = min(SM.shape[0], len(S))
indices = np.arange(min_len)
SM[indices, indices] = S[:min_len]

US = U @ SM
C_weights = np.zeros((B_s,VT.shape[1]))
R_weights = np.zeros((B_s,U.shape[0]))
for i in range(B_s):
    C_weights[i,:] = VT.T[:,i]
    R_weights[i,:] = US[:,i]


y = np.zeros(firmem)
firsvdsec = FIRSVDFilterPy(C_weights, R_weights)
firsvdsec.reset()
y[0] = firsvdsec.filterstep(1.0)
for k in range(1,firmem):
    y[k] = firsvdsec.filterstep(0.0)

#Generate cut filters

#fig.show()
C_weights_cut = np.zeros((B_s_cut,C_chosen_s_cut))
R_weights_cut = np.zeros((B_s_cut,U.shape[0]))
for i in range(B_s_cut):
    C_weights_cut[i,:] = VT.T[:C_chosen_s_cut,i]
    R_weights_cut[i,:] = US[:,i]
    

y = np.zeros(firmem)
firsvdsec_cut = FIRSVDFilterPy(C_weights_cut, R_weights_cut)
firsvdsec_cut.reset()
y[0] = firsvdsec_cut.filterstep(1.0)
for k in range(1,firmem):
    y[k] = firsvdsec_cut.filterstep(0.0)


print(f'Total number of coefficients: {C_weights.size + R_weights.size} vs {wsecimpulse.size} ({100*(1 - (C_weights.size + R_weights.size)/wsecimpulse.size):.2f}% reduction)')
print(f'Total number of coefficients: {C_weights_cut.size + R_weights_cut.size} vs {wsecimpulse.size} ({100*(1 - (C_weights_cut.size + R_weights_cut.size)/wsecimpulse.size):.2f}% reduction)')

px.line(y=S, title='Singular values of the secondary path').show()



# Feedback path compression:

C_chosen_fb = 34
B_fb = 3

C_chosen_fb_cut = 18
B_fb_cut = 4

#W = format_W(wfbkimpulse,C_chosen_fb)
W = format_W(wfbkimpulse,C_chosen_fb)
print(f'{W.shape = }')
R = W.shape[0]
U,S,VT = np.linalg.svd(W)

SM = np.zeros((R,C_chosen_fb))
min_len = min(SM.shape[0], len(S))
indices = np.arange(min_len)
SM[indices, indices] = S[:min_len]

US = U @ SM
C_weightsfbk = np.zeros((B_fb,VT.shape[1]))
R_weightsfbk = np.zeros((B_fb,U.shape[0]))

for i in range(B_fb):
    C_weightsfbk[i,:] = VT.T[:,i]
    R_weightsfbk[i,:] = US[:,i]


y = np.zeros(firmem)
firsvdfbk = FIRSVDFilterPy(C_weightsfbk, R_weightsfbk)
firsvdfbk.reset()
y[0] = firsvdfbk.filterstep(1.0)
for k in range(1,firmem):
    y[k] = firsvdfbk.filterstep(0.0)

#Generate cut filter

C_weightsfbk_cut = np.zeros((B_fb_cut,C_chosen_fb_cut))
R_weightsfbk_cut = np.zeros((B_fb_cut,U.shape[0]))
for i in range(B_fb_cut):
    C_weightsfbk_cut[i,:] = VT.T[:C_chosen_fb_cut,i]
    R_weightsfbk_cut[i,:] = US[:,i]
    

#print(C_weightsfbk_cut == C_weightsfbk , R_weightsfbk_cut == R_weightsfbk)
#print(C_weights_cut == C_weights , R_weights_cut == R_weights)

y = np.zeros(firmem)
firsvdfbk_cut = FIRSVDFilterPy(C_weightsfbk_cut, R_weightsfbk_cut)
firsvdfbk_cut.reset()
y[0] = firsvdfbk_cut.filterstep(1.0)
for k in range(1,firmem):
    y[k] = firsvdfbk_cut.filterstep(0.0)


print(f'Total number of coefficients: {C_weightsfbk.size + R_weightsfbk.size} vs {wfbkimpulse.size} ({100*(1 - (R_weightsfbk.size + R_weightsfbk.size)/wfbkimpulse.size):.2f}% reduction)')
print(f'Total number of coefficients: {C_weightsfbk_cut.size + R_weightsfbk_cut.size} vs {wfbkimpulse.size} ({100*(1 - (C_weightsfbk_cut.size + R_weightsfbk_cut.size)/wfbkimpulse.size):.2f}% reduction)')

#px.line(y=S, title='Singular values of the feedback path').show()

fig , ax = plt.subplots(2,2,figsize = (12,8))

for i in range(B_fb):
    ax[0,0].plot(C_weightsfbk[i,:])
    ax[1,0].plot(R_weightsfbk[i,:])
  
for i in range(B_fb_cut):
    ax[0,1].plot(C_weightsfbk_cut[i,:])
    ax[1,1].plot(R_weightsfbk_cut[i,:])

fig.show()

fig , ax = plt.subplots(2,2,figsize = (12,8))

for i in range(B_s):
    ax[0,0].plot(C_weights[i,:])
    ax[1,0].plot(R_weights[i,:])
  
for i in range(B_s_cut):
    ax[0,1].plot(C_weights_cut[i,:])
    ax[1,1].plot(R_weights_cut[i,:])

fig.show()

# %%
"""
General Parameters for simulation comparison
"""


def run_step_comparison(mu,force_amplitude,firsvdseccut,firsvdfbkcut,firsvdsec,firsvdfbk,firmem6400):

    maxtime = 120.0
    nsteps = int(maxtime * fs) # Total number of steps
    vibstart = 0.0 # Start time of the vibration
    controlstart = 30.0 # Start time of the control
    vibfreq = 90 # Hertz

    controller = FIRFxNLMS(mem=firmem6400, memsec=firmem6400) # Create the controller
    # controller.setSecondary(wsecimpulse) # Set the secondary path
    firsvdseccut.reset()
    controller.setSecondary(firsvdseccut) # Set the secondary path
    controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
    controller.mu = mu # Set the step size
    controller.psi = 1e-3 # Set the regularization parameter
    controller.reset() # Reset the controller

    feedbackfilter = firsvdfbkcut # Create the feedback filter
    feedbackfilter.reset() # Reset the filter

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

    """
    Run SVD Filter
    """
    firmem = firmem6400
    controller = FIRFxNLMS(mem=firmem, memsec=firmem) # Create the controller
    # controller.setSecondary(wsecimpulse) # Set the secondary path
    firsvdsec.reset()
    controller.setSecondary(firsvdsec) # Set the secondary path
    controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
    controller.mu = mu # Set the step size
    controller.psi = 1e-3 # Set the regularization parameter
    controller.reset() # Reset the controller

    feedbackfilter = firsvdfbk  # Create the feedback filter
    feedbackfilter.reset() # Reset the filter

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
    fig.add_scatter(x=th, y=err_500, name= f"Beam accelaration (m/s²) cut SVD taps mu = {mu}", mode="lines")
    fig.add_scatter(x=th, y=err_6400, name=f"Beam accelaration (m/s²)     SVD taps mu = {mu}", mode="lines")

    # For err_500
    energy_500  = err_500**2
    energy_6400 = err_6400**2

    max_energy = max(energy_500[int(fs*10): int(fs*20)])
    max_index_original = np.argmax(energy_500[int(fs*10): int(fs*20)]) + int(fs*10)

    peaks_500 = find_local_maxima(energy_500)
    #max_energy = energy_500[peaks_500_mean[int(len(peaks_500_mean)/2)]]
    print(max_energy)

    peaks_500_below = peaks_500[energy_500[peaks_500] > 0.05 * max_energy]


    fig.add_vline(x=th[peaks_500_below[-1]], line_width=1, line_dash="dash", line_color="red")
    fig.add_vline(x=th[max_index_original], line_width=1, line_dash="dash", line_color="yellow")

    # For err_6400
    peaks_6400 = find_local_maxima(energy_6400)
    peaks_6400_below = peaks_6400[energy_6400[peaks_6400] > 0.05 * max_energy]
    fig.add_vline(x=th[peaks_6400_below[-1]], line_width=1, line_dash="dash", line_color="blue")

    #Get times to Reach the 5% of energy

    dt_500 = th[peaks_500_below[-1]] - 30
    dt_6400 = th[peaks_6400_below[-1]] - 30

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

    print("Time to stabilize under 5% energy \n Using Same Number of coeffients:")
    print(f"SVD     : {dt_6400:.2f} s\n cut SVD : {dt_500:.2f} s\nAccelleration: {dt_500/dt_6400:.2f}x")

    return dt_6400,dt_500
# %%


mu = np.arange(0.001,0.051,0.005)
d_svd = np.zeros_like(mu)
d_nsvd = np.zeros_like(mu)
force_amplitude = 1

firmem = 200
#wsecimpulse_500 , wfbkimpulse_500 = gen_wsec_wfbk_filters(200)


for i,m in enumerate(mu):
    d_svd[i],d_nsvd[i] = run_step_comparison(m,force_amplitude,firsvdsec_cut,firsvdfbk_cut,firsvdsec,firsvdfbk,192)

# %%
fig = px.line()
fig.add_scatter(y = d_svd , x = mu)
fig.add_scatter(y = d_nsvd , x = mu)
fig.show()
# %%
fig = px.line()
fig.add_scatter(y = d_nsvd/d_svd , x = mu)
fig.show()
#
# %%
