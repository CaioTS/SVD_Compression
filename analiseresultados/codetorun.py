#%%
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# import pandas as pd
from ActVibModules.CantileverBeam import CantileverBeam
# from ActVibModules.Adaptive import FIRNLMS
from ActVibModules.DSPFuncs import easyFourier
from ActVibModules.AdaptiveOO import FIRFxNLMS, FIR
from itertools import product
import pandas as pd

#%%

fs = 416.0 # Sampling frequency in Hertz

beamconfig = {
    'npoints': 100, # Number of points in the beam (finite element method)
    'length': 0.58, # Length of the beam in meters
    'width': 0.05, # Width of the beam in meters
    'thickness': 0.006, # Thickness of the beam in meters
    'damp': [0.005] * 5, # Damping factors for the first 5 modes
    'Tsampling': 1.0/fs # Sampling period
}

experimentconfig = {
    'perturbpos': 30, # Position of the perturbation force, which causes beam vibration.
    'referencepos': 75, # Position of the acceleration measurement at the beam.
    'controlpos': 60, # Position of the control force
    'errorpos': 95, # Position of the error acceleration measurement in the beam
} 


firmem = 2000 # Number of samples for the secondary and feedback paths


#  Creating Beam instance with 100 points:
cbeam = CantileverBeam(**beamconfig)
cbeam.reset()

print("Natural frequencies are:\n",
      ",\n".join(cbeam.freqsHz.astype(str).tolist()),
      " (all in Hz).")

xcoords = np.linspace(0.0, cbeam.length, beamconfig['npoints'])
xcoords = np.concatenate((xcoords, xcoords[::-1]))
ycoords = np.array([0.0]*beamconfig['npoints'] + [beamconfig['thickness']]*beamconfig['npoints'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=xcoords, y=ycoords, fill='toself', mode='lines'))
fig.add_annotation(x=experimentconfig['perturbpos']*beamconfig['length']/beamconfig['npoints'], y=beamconfig['thickness']*1.1, 
            ax=0, ay=-30, text="Perturbation",
            showarrow=True, arrowhead=1)
fig.add_annotation(x=experimentconfig['referencepos']*beamconfig['length']/beamconfig['npoints'], y=beamconfig['thickness']*1.1, 
            ax=0, ay=-50, text="Accel. Measurement",
            showarrow=True, arrowhead=1, arrowside="start")
fig.add_annotation(x=experimentconfig['controlpos']*beamconfig['length']/beamconfig['npoints'], y=0, 
            ax=0, ay=30, text="Control Force",
            showarrow=True, arrowhead=1)
fig.add_annotation(x=experimentconfig['errorpos']*beamconfig['length']/beamconfig['npoints'], y=0, 
            ax=0, ay=50, text="Error Accel.",
            showarrow=True, arrowhead=1, arrowside="start")
fig.update_layout(title="Cantilever Beam", xaxis_title="x (m)", yaxis_title="y (m)")
fig.update_layout(xaxis=dict(range=[0, beamconfig['length']*1.1]), yaxis=dict(range=[-(beamconfig['thickness'] + 0.1), beamconfig['thickness'] + 0.1]))
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
  err[k] = cbeam.getaccelms2(experimentconfig['referencepos'])  
  cbeam.setforce(experimentconfig['perturbpos'],xh[k]) 
  cbeam.update() # Updata for 1 sampling period.

# Plotting the results:
fig = px.line()
fig.add_scatter(x=th, y=xh, name="Força (N)", mode="lines")
fig.add_scatter(x=th, y=err, name="Aceleração (m/s²)", mode="lines")
fig.show()

cbeam.reset()
impulse_response = np.zeros(firmem) # Impulse response vector
cbeam.setforce(experimentconfig['perturbpos'],1.0) # Force is applied at the perturbation position
cbeam.update()
impulse_response[0] = cbeam.getaccelms2(experimentconfig['referencepos']) # Read the acceleration at the reference position
cbeam.setforce(experimentconfig['perturbpos'],0.0) # Force is removed
for k in range(1,firmem):
  cbeam.update() # Update the beam for 1 sampling period.
  impulse_response[k] = cbeam.getaccelms2(experimentconfig['referencepos']) # Read the acceleration at the reference position
fig = px.line(y=impulse_response, title="Impulse Response at Reference Position")
fig.update_layout(xaxis_title="Samples (n)", yaxis_title="Acceleration (m/s²)")
fig.show()


# %% Funções

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


def gen_wsec_wfbk_filters(cbeam, firmem, controlpos, errorpos, referencepos, perturbpos, verbose=False):
    # Secondary path via impulse response (ideal but not practical):
    wsecimpulse = np.zeros(firmem) # Impulse response vector
    cbeam.reset()
    cbeam.setforce(controlpos,1.0) # Force is applied at the control position
    cbeam.update()
    wsecimpulse[0] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position
    #cbeam.update()
    cbeam.setforce(controlpos,0.0) # Force is removed
    for k in range(1,firmem):
        cbeam.update() # Update the beam for 1 sampling period.
        wsecimpulse[k] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position


    wfbkimpulse = np.zeros(firmem) # Impulse response vector
    cbeam.reset()
    cbeam.setforce(controlpos,1.0) # Force is applied at the control position
    cbeam.update()
    wfbkimpulse[0] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position
    #cbeam.update()
    cbeam.setforce(controlpos,0.0) # Force is removed
    for k in range(1,firmem):
        cbeam.update() # Update the beam for 1 sampling period.
        wfbkimpulse[k] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position

    if verbose:
        fig = px.line()
        fig.add_scatter(y = wfbkimpulse)
        fig.add_scatter(y = wsecimpulse)

    return wsecimpulse,wfbkimpulse


def calculate_convergence_time(err, threshold, controlstart=30.0, fs=416, refwindow=5.0):
    energy  = err**2    
    # th = np.linspace(0.0,len(err)/fs,len(err)) # Time vector

    refenergy = np.max(energy[int(fs*(controlstart-refwindow)): int(fs*controlstart)])    
    # max_energy = max(energy[int(fs*10): int(fs*20)])
    # max_index_original = np.argmax(energy[int(fs*10): int(fs*20)]) + int(fs*10)

    overshoot = np.max(energy[int(fs*controlstart):])/refenergy
    # overshoot_index = np.argmax(energy[int(fs*30):]) + int(fs*30)
    # indeces = np.arange(0,len(th),1)

    # indeces_above = indeces[energy[indeces] >= threshold * refenergy]
    # point_above = indeces_above[-1] + 1 
    # if point_above >= len(th):
    #    point_above = len(th) - 1
    #Get times to Reach the threshold of energy

    # dt = th[indeces_above[-1]] - 30
    convthres = threshold * refenergy
    lastabovethres = np.where(energy[int(fs*controlstart):] >= convthres)
    if lastabovethres[0].size > 0:
        lastindex = lastabovethres[0][-1] + int(fs*controlstart)
        dt = lastindex / fs - controlstart
    else:
        dt = float('NaN')

    return overshoot,dt

def sqerror_smooth(err, alpha=0.95):
    smoothed = np.zeros_like(err)
    smoothed[0] = err[0]**2
    alpham1 = 1 - alpha
    for k in range(1, len(err)):
        smoothed[k] = alpha * smoothed[k-1] + alpham1 * err[k]**2
    return smoothed

"""
Calculate and organize filter weights for SVD
"""
def gen_SVD_weights(weights,num_coef):
    """
    Args:
        weights (array): Array of filters weights
        num_coef: Number of coefficiets desired for SVD filter num_branches*( num_collunms + num_rows)
    """
    
    #Will assume the most squared matrix for the weights
    full_size = len(weights)
    C_chosen_s = int(np.sqrt(full_size))

    #W = format_W(wsecimpulse,C_chosen_s)
    W = format_W(weights,C_chosen_s)
    # print(f'{W.shape = }')
    R_s = W.shape[0]
    B_s = int(np.round(num_coef/(C_chosen_s + R_s)))

    U,S,VT = np.linalg.svd(W)

    SM = np.zeros((R_s,C_chosen_s))
    np.fill_diagonal(SM,S)
    US = U @ SM
    C_weights = np.zeros((B_s,VT.shape[1]))
    R_weights = np.zeros((B_s,U.shape[0]))

    for i in range(B_s):
        C_weights[i,:] = VT.T[:,i]
        R_weights[i,:] = US[:,i]

    return C_weights,R_weights


def run_sim(threshold,filter_size,wsecimpulse,wfbkimpulse,isSVD=False,
            compressed_filters='both',verbose=False,
            beamconfig=beamconfig, experimentconfig=experimentconfig,
            perturbconfig=None, controllerconfig=None):
    """
    Simulation for the beam and different filters implementations

    Args:
        threshold (float): Convergence threshold for the error signal. Defaults to 0.05.
        controllersize (int): Size of the controller memory.
        filter_size (int): Number of taps (memory length) for the FIR filter.
        wsecimpulse (array_like): Impulse response of the secondary path (S).
        wfbkimpulse (array_like): Impulse response of the feedback path (F).
        mu (float): Step size (learning rate) for the adaptive algorithm update.
        psi (float): Regularization parameter.
        isSVD (bool): If True, performs the update using Singular Value Decomposition.
        compressed_filters(string): Choose what filters are compressed: (both, sec, fbk)
        verbose (bool): If True, prints detailed information during the simulation.
        perturbconfig (dict): Configuration for the perturbation signal.
    Returns:
        tuple: A tuple containing (overshoot , convergence_time).
    """
    
    np.seterr(over='raise', invalid='raise', divide='raise') #Operation
    
    #  Creating Beam instance with 100 points:    
    cbeam = CantileverBeam(**beamconfig)
    cbeam.reset()
    
    force_amplitude = pertb_amp
    maxtime = 120.0
    nsteps = int(maxtime * fs) # Total number of steps
    controlstart = 30.0 # Start time of the control
    
    if compressed_filters == "both":
        wsec = wsecimpulse[:filter_size]
        wfbk = wfbkimpulse[:filter_size]
    elif compressed_filters == "sec":
        wsec = wsecimpulse[:filter_size]
        wfbk = wfbkimpulse
    elif compressed_filters == "fbk":
        wsec = wsecimpulse
        wfbk = wfbkimpulse[:filter_size]
    else:
        raise ValueError("compressed filter not compatible with function")

    controller = FIRFxNLMS(mem=controllerconfig['mem'], memsec=1) # Create the controller
    if isSVD :
        C_weights_sec,R_weights_sec = gen_SVD_weights(wsecimpulse,filter_size)
        C_weights_fbk,R_weights_fbk = gen_SVD_weights(wfbkimpulse,filter_size)
        controller.setSecondary(FIRSVDFilterPy(C_weights_sec,R_weights_sec)) # Set the secondary path
        feedbackfilter = FIRSVDFilterPy(C_weights_fbk,R_weights_fbk)  # Create the feedback filter
    else:
      controller.setSecondary(FIR(wsec)) # Set the secondary path
      feedbackfilter = FIR(wfbk)
    controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
    controller.mu = controllerconfig['mu'] # Set the step size
    controller.psi = controllerconfig['psi'] # Set the regularization parameter
    controller.reset() # Reset the controller
    feedbackfilter.reset() # Reset the filter

    vibfreq = pertb_freq # Hertz
    th = np.linspace(0.0,maxtime,nsteps) # Time vector
    if perturbconfig['type'] == 'harmonic':
        force_amplitude = perturbconfig['amp']
        vibfreq = perturbconfig['freq']
        xh = force_amplitude*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector        

    cbeam.reset()
    err = np.zeros(nsteps) # Vibration response
    # yfbk = np.zeros(nsteps) # Vibration response
    yfbk = 0
    ypf = np.zeros(nsteps) # Plant output without control

    try:
        # Running the simulation:
        perturbpos = experimentconfig['perturbpos']
        controlpos = experimentconfig['controlpos']
        errorpos = experimentconfig['errorpos']
        referencepos = experimentconfig['referencepos']
        for k in range(nsteps):
            cbeam.setforce(perturbpos,xh[k]) # force is applied
            cbeam.setforce(controlpos,-controller.y) # Control force is applied    
            if th[k] >= controlstart: # Control starts at 30 seconds
                controller.update(cbeam.getaccelms2(errorpos))        
            yfbk = feedbackfilter.filterstep(-controller.y) # Get the feedback force
            ypf[k] = cbeam.getaccelms2(referencepos) - yfbk
            controller.evalout(ypf[k])
            err[k] = cbeam.getaccelms2(errorpos) # Error acceleration is read
            cbeam.update() # beam is updated            
        ov, dt = calculate_convergence_time(err,threshold)
        if ov > 1000:
           ov = 1000
    except (ValueError, FloatingPointError, OverflowError) as ex:
        print(f"Warning: numerical error at step {k}: {ex}")
        #Added symbolic values informing it failed
        ov = 1000 
        dt = float('NaN')
    
    if verbose:
        fig = px.line(title=f"Vibration Control nmem = {filter_size} mu = {mu}",x=th,y=sqerror_smooth(err))
        fig.update_yaxes(type="log", title="Smoothed Squared Error (log scale)")
        fig.update_xaxes(title="Time (s)")
        fig.add_vline(x=dt + 30, line_width=1, line_dash="dash", line_color="yellow")
        fig.add_hline(y=np.sqrt(ov), line_width=1, line_dash="dash", line_color="red")
        fig.show()
        fig2 = px.line()
        fig2.add_scatter(x=th, y=ypf, name="reference minus feedback", mode="lines")
        fig2.show()

        
    return ov,dt




#%%
# if __name__ == "__main__":
    #Steps to use this function , Generate Beam(First lines in file)
    #Generate full secondary and feedback weights
firmem = 4000
wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(cbeam,firmem,verbose=True,**experimentconfig)
px.line(y=wfbkimpulse, title="Feedback Path Impulse Response").show()
pertb_freq , pertb_amp = 15.0 , 1.0
threshold = 0.1
filter_size = 650
controllersize = 300
mu = 0.002
psi = 1e-2
controllerconfig = {'mem': controllersize, 'mu': mu, 'psi': psi}
perturbconfig = {"type":'harmonic', 'freq':pertb_freq, 'amp':pertb_amp}
ov , dt = run_sim(threshold,filter_size,wsecimpulse,wfbkimpulse,
                  isSVD=True,compressed_filters='fbk',verbose=True,
                  beamconfig=beamconfig,experimentconfig=experimentconfig,
                  perturbconfig=perturbconfig, controllerconfig=controllerconfig)
    # ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,mu,psi,isSVD=True,compressed_filters='sec')
    # ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,mu,psi,isSVD=False,compressed_filters='sec')
print(ov,dt)

# %%
wsecimpulse1 = wfbkimpulse[:2000]
wsecimpulse2 = wfbkimpulse[:650]
larger = max(wsecimpulse1.shape[0],wsecimpulse2.shape[0])
mag,freq,pha = easyFourier(wsecimpulse1,fs,phasealso=True)
mag2,freq2,pha2 = easyFourier(np.concatenate([wsecimpulse2]),
                              fs,phasealso=True)
fig = px.line()
fig.add_scatter(y=wsecimpulse1, mode='lines', name=f'Secondary Path Impulse Response ({wsecimpulse1.shape[0]} samples)')
fig.add_scatter(y=wsecimpulse2, mode='lines', name=f'Secondary Path Impulse Response ({wsecimpulse2.shape[0]} samples)')
fig.update_layout(title="Secondary Path Impulse Response", xaxis_title="Samples (n)", yaxis_title="Acceleration (m/s²)")
fig.show()
fig = px.line()
fig.add_scatter(x=freq, y=mag, mode='lines', name='Magnitude Spectrum')
fig.add_scatter(x=freq2, y=mag2, mode='lines', name=f'Magnitude Spectrum ({wsecimpulse2.shape[0]} samples)')
fig.update_layout(title="Magnitude Spectrum of Secondary Path Impulse Response", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude") 
fig.show()
fig2 = px.line()
fig2.add_scatter(x=freq, y=np.unwrap(pha), mode='lines', name='Phase Spectrum')
fig2.add_scatter(x=freq2, y=np.unwrap(pha2), mode='lines', name=f'Phase Spectrum ({wsecimpulse2.shape[0]} samples)')
fig2.update_layout(title="Phase Spectrum of Secondary Path Impulse Response", xaxis_title="Frequency (Hz)", yaxis_title="Phase (degrees)")
fig2.show()

# %%
perturbfreqs = [13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0]
filtersizes = [200, 400, 600, 800, 1000, 1500, 2000]
mus = [a * 0.001 for a in range(1, 11)] + [a * 0.01 for a in range(1, 11)] + [a * 0.1 for a in range(1, 5)]
psis = [1e-4, 1e-3, 1e-2, 1e-1]
issvds = [True, False]

# perturbfreqs = [13.5, 14.0]
# filtersizes = [200,]
# mus = [a * 0.001 for a in range(1, 3)]
# psis = [1e-2]
# issvds = [True]

combinations = list(product(perturbfreqs, filtersizes, mus, psis, issvds))
print(combinations.__len__())

# results = []
# for pertb_freq, filter_size, mu, psi, isSVD in combinations:
#     print(f"Running simulation with pertb_freq={pertb_freq}, filter_size={filter_size}, mu={mu}, psi={psi}, isSVD={isSVD}")
#     ov , dt = run_sim(pertb_freq, pertb_amp, threshold, filter_size, wsecimpulse, wfbkimpulse, mu, psi, isSVD=isSVD, compressed_filters='both')    
#     print(f"Results: Overshoot = {ov}, Convergence Time = {dt} seconds\n")
#     results.append((pertb_freq, filter_size, mu, psi, isSVD, ov, dt))

# df = pd.DataFrame(results, columns=['Perturb Freq', 'Filter Size', 'Mu', 'Psi', 'Is SVD', 'Overshoot', 'Convergence Time'])
# df.to_csv('simulation_results.csv', index=False)

# %%
# Multiprocessing version of the simulation loop

def run_simulation_wrapper(args):
    """Wrapper function to run a single simulation with multiprocessing."""
    pertb_freq, filter_size, mu, psi, isSVD, wsecimpulse, wfbkimpulse, pertb_amp, threshold = args
    print(f"(PID {os.getpid()}) Running simurulation with pertb_freq={pertb_freq}, filter_size={filter_size}, mu={mu}, psi={psi}, isSVD={isSVD}")
    ov, dt = run_sim(pertb_freq, pertb_amp, threshold, filter_size, wsecimpulse, wfbkimpulse, mu, psi, isSVD=isSVD, compressed_filters='both')
    print(f"(PID {os.getpid()}) Results: Overshoot = {ov}, Convergence Time = {dt} seconds\n")
    return (pertb_freq, filter_size, mu, psi, isSVD, ov, dt)

if __name__ == '__main__':
    # Prepare arguments for multiprocessing
    mp_args = [(pertb_freq, filter_size, mu, psi, isSVD, wsecimpulse, wfbkimpulse, pertb_amp, threshold) 
               for pertb_freq, filter_size, mu, psi, isSVD in combinations]
    
    # Run simulations using multiprocessing
    print("Available CPUs:",os.cpu_count())
    results_mp = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count()-2)) as executor:
        futures = [executor.submit(run_simulation_wrapper, args) for args in mp_args]
        for future in as_completed(futures):
            result = future.result()
            results_mp.append(result)
            print(f"Finished simulation: pertb_freq={result[0]}, filter_size={result[1]}, mu={result[2]}, psi={result[3]}, isSVD={result[4]}")
    
    # Create DataFrame from results
    df_mp = pd.DataFrame(results_mp, columns=['Perturb Freq', 'Filter Size', 'Mu', 'Psi', 'Is SVD', 'Overshoot', 'Convergence Time'])
    df_mp.to_csv('simulation_results_multiprocessing_filtermem2000.csv', index=False)
    print("Multiprocessing results saved to 'simulation_results_multiprocessing.csv'")

# %%
