#%%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from CantileverBeam import CantileverBeam
from Adaptive import FIRNLMS
from AdaptiveOO import FIRFxNLMS, FIR

#%%

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


#  Creating Beam instance with 100 points:
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
#%%
def calculate_convergence_time (err, threshold,step,nmem):
    energy  = err**2
    fs = 416
    th = np.linspace(0.0,len(err)/fs,len(err)) # Time vector

    max_energy = max(energy[int(fs*10): int(fs*20)])
    max_index_original = np.argmax(energy[int(fs*10): int(fs*20)]) + int(fs*10)

    overshoot = max(energy[int(fs*30):])/max_energy
    overshoot_index = np.argmax(energy[int(fs*30):]) + int(fs*30)
    indeces = np.arange(0,len(th),1)

    indeces_above = indeces[energy[indeces] >= threshold * max_energy]
    point_above = indeces_above[-1] + 1 
    if point_above >= len(th):
       point_above = len(th) - 1
    #Get times to Reach the 5% of energy

    dt = th[indeces_above[-1]] - 30
    if dt < 20:
      converged = True
    else:
      converged = False

    #fig = px.line(title=f"Vibration Control nmem = {nmem} mu = {step}")
    #fig.add_scatter(x=th, y=energy, name=f"Beam accelaration (m/s²)", mode="lines")
    #fig.add_vline(x=th[point_above], line_width=1, line_dash="dash", line_color="red")
    #fig.add_hline(y=energy[max_index_original], line_width=1, line_dash="dash", line_color="yellow")
    #fig.add_scatter(x=[th[overshoot_index]], y=[overshoot*max_energy], name=f"Overshoot", mode="markers")


    #fig.update_layout(
    #    legend=dict(
    #        orientation="h", # Horizontal legend
    #        yanchor="bottom",
    #        y=-0.3, # Adjust this value to move the legend further down
    #        xanchor="center",
    #        x=0.5
    #    ),
    #    margin=dict(b=100) # Increase bottom margin if needed
    #)

    #fig.show()
    return overshoot,converged,dt

def run_sim(firmem,wsecimpulse,wfbkimpulse,mu):
    force_amplitude = 0.3
    maxtime = 120.0
    nsteps = int(maxtime * fs) # Total number of steps
    vibstart = 0.0 # Start time of the vibration
    controlstart = 30.0 # Start time of the control

    controller = FIRFxNLMS(mem=firmem, memsec=firmem) # Create the controller
    # controller.setSecondary(wsecimpulse) # Set the secondary path
    controller.setSecondary(FIR(wsecimpulse)) # Set the secondary path
    controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
    controller.mu = mu # Set the step size
    controller.psi = 1e-2 # Set the regularization parameter
    controller.reset() # Reset the controller

    feedbackfilter = FIR(wfbkimpulse) # Create the feedback filter
    feedbackfilter.reset() # Reset the filter

    vibfreq = 13.5 # Hertz
    th = np.linspace(0.0,maxtime,nsteps) # Time vector
    xh = force_amplitude*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
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
    return err
#%%
firmem = 6400
wsecimpulse , wfbkimpulse = gen_wsec_wfbk_filters(firmem)
mu = 0.01
err = run_sim(firmem,wsecimpulse,wfbkimpulse,mu)

fig = px.line()
fig.add_scatter(y = err)
# %%
df = pd.DataFrame(columns=["nmem", "step", "dt","overshoot", "converged"])
nmems = [6400,4000,3500,3000,2500,2000,1000,500,300]

for nmem in nmems:
  step = 0.015
  fig = px.line(title=f"Only Secondary nmem = {nmem} mu = {step} f_vib = 13.5Hz")
  fig.update_layout(
    xaxis_title="Step",
    yaxis_title="Time of Convergence (s)"
)
  converged = True
  overshoot = 0
  while  step >= 0.001 :
    err = run_sim(nmem,wsecimpulse[:nmem],wfbkimpulse,step)
    overshoot,converged ,dt = calculate_convergence_time(err,0.05,step,nmem)
    df.loc[len(df)] = [nmem,step,dt,overshoot,converged]
    fig.add_scatter(y=err, name= f"mu = {step}")
    step = step - 0.001
    
  fig.show()
# %%
fig = px.line()

fig = px.line(title=f"Comparisons Changing only secondary f_vib = 13.5Hz")
fig.update_layout(
    xaxis_title="Step",
    yaxis_title="Time of Convergence (s)"
)
for nmem in nmems:
    temp_df = df[df["nmem"] == nmem]

    # Main trace (default color)
    fig.add_scatter(
        x=temp_df["step"],
        y=temp_df["dt"],
        name=f"nmem = {nmem}",
        mode="lines+markers"
    )

fig.show()
# %%
df_wsec_wfbk = pd.DataFrame(columns=["nmem", "step", "dt","overshoot", "converged"])
nmems = [6400,4000,3500,3000,2500,2000,1000,500,300]

for nmem in nmems:
  step = 0.015
  fig = px.line(title=f"Secondary and Feedback nmem = {nmem} mu = {step} f_vib = 13.5Hz")
  fig.update_layout(
    xaxis_title="Step",
    yaxis_title="Time of Convergence (s)"
)
  converged = True
  overshoot = 0
  while step >= 0.001  :
    err = run_sim(nmem,wsecimpulse[:nmem],wfbkimpulse[:nmem],step)
    overshoot,converged ,dt = calculate_convergence_time(err,0.05,step,nmem)
    df_wsec_wfbk.loc[len(df_wsec_wfbk)] = [nmem,step,dt,overshoot,converged]
    fig.add_scatter(y=err, name= f"mu = {step}")
    step = step - 0.001
  fig.show()
#%%
fig = px.line()

fig = px.line(title=f"Comparisons Changing secondary and feedback f_vib = 13.5Hz")
fig.update_layout(
    xaxis_title="Step",
    yaxis_title="Time of Convergence (s)"
)
for nmem in nmems:
  temp_df = df_wsec_wfbk[df_wsec_wfbk["nmem"] == nmem]
  fig.add_scatter(x = temp_df["step"], y = temp_df["dt"],name=f"nmem = {nmem}" )
fig.show()
# %%
df_wfbk = pd.DataFrame(columns=["nmem", "step", "dt","overshoot", "converged"])
nmems = [6400,4000,3500,3000,2500,2000,1000,500,300]

for nmem in nmems:
  step = 0.001
  converged = True
  overshoot = 0
  while converged == True and step <= 0.018  :
    err = run_sim(6400,wsecimpulse,wfbkimpulse[:nmem],step)
    overshoot,converged ,dt = calculate_convergence_time(err,0.05,step,nmem)
    df_wfbk.loc[len(df_wfbk)] = [nmem,step,dt,overshoot,converged]
    step = step + 0.002

#%%
fig = px.line()

fig = px.line(title=f"Comparisons Changing only feedback f_vib = 13.5Hz")
fig.update_layout(
    xaxis_title="Step",
    yaxis_title="Time of Convergence (s)"
)
for nmem in nmems:
  temp_df = df_wfbk[df_wfbk["nmem"] == nmem]
  fig.add_scatter(x = temp_df["step"], y = temp_df["dt"],name=f"nmem = {nmem}" )
fig.show()
# %%
