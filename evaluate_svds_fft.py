#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import plotly.express as px
from CantileverBeam import CantileverBeam
from Filters import FIR
from Adaptive import FIRNLMS, FIRFxNLMS
#%%
def compare_spectrums(signal, fs, title):
    n = len(signal)
    yfs = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    #phase_signal = np.unwrap(np.angle(yfs[0:n//2]))%np.pi
    #phase_signal = (np.angle(yfs[0:n//2]))

    plt.plot(xf, 2.0/n * np.abs(yfs[0:n//2]),label = title,linewidth=3)
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    #plt.plot(phase_signal,label= title)
    plt.grid(True)
    plt.legend()
    plt.xlim(0, fs/2)


def compare_spectrums_phase(signal, ref, fs, title):
    n_signal = len(signal)
    n_ref = len(ref)
    
    # Ensure both signals have the same length for proper comparison
    min_length = min(n_signal, n_ref)
    signal = signal[:min_length]
    ref = ref[:min_length]
    n = min_length
    
    # Calculate FFTs
    yfs = fft(signal)
    yfr = fft(ref)
    xf = fftfreq(n, 1/fs)[:n//2]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Magnitude spectrum (log scale)
    ax1.plot(xf, 2.0/n * np.abs(yfs[0:n//2]), 
             label='Signal Filter Response', linewidth=3, alpha=0.8)
    ax1.plot(xf, 2.0/n * np.abs(yfr[0:n//2]), 
             label='Reference Filter Response', linewidth=2, alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_title(f'{title} - Magnitude Spectrum')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, fs/2)
    
    # Phase spectrum
    phase_signal = np.unwrap(np.angle(yfs[0:n//2]))
    phase_ref = np.unwrap(np.angle(yfr[0:n//2]))
    
    ax2.plot(xf, phase_signal, 
             label='Signal Phase', linewidth=3, alpha=0.8)
    ax2.plot(xf, phase_ref, 
             label='Reference Phase', linewidth=2, alpha=0.8)
    ax2.set_title(f'{title} - Phase Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (radians)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, fs/2)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate error metric (using only positive frequencies)
    error = abs(np.sum(((yfs[0:n//2] - yfr[0:n//2])**2) / len(xf)))
    
    return error



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


def get_wsec_filter(N,fs):
    fs = fs # Sampling frequency

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
    # Creating Beam instance with 100 points:

    cbeam = CantileverBeam(npoints=npoints, width=beamwidth, thickness=beamthickness, 
                            length=beamlength, Tsampling=1.0/fs,
                            damp=dampingfactors)
    cbeam.reset()
    #print("Natural frequencies are:\n",
    #      ",\n".join(cbeam.freqsHz.astype(str).tolist()),
    #      " (all in Hz).")

    xcoords = np.linspace(0.0, cbeam.length, npoints)
    xcoords = np.concatenate((xcoords, xcoords[::-1]))
    ycoords = np.array([0.0]*npoints + [beamthickness]*npoints)
    
    #Simulating the beam response to a sinusoidal force:

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

    # Active control requires modeling both the secondary and feedback paths:
    # The secondary path is the path from the control force to the error sensor.
    # The feedback path is the path from the control force to the reference sensor.
    # Modeling carried out using the FIRNLMS algorithm.

    maxtime = 100.0
    nsteps = int(maxtime * fs)

    firmem = N # Number of samples for the secondary and feedback paths
    firnlms = FIRNLMS(memorysize=firmem,stepsize=0.15,regularization=1e-3) # Create the FIRNLMS object


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


    # Comparing the two methods:
    #fig = px.line()
    #fig.add_scatter(y=wsecimpulse, name="Impulse response", mode="lines")
    #fig.update_layout(title="Secondary path response (FIR)")
    #fig.show() # Plot the secondary path coefficients

    return wsecimpulse

# Filter parameters
fs = 800  # Sampling frequency (Hz)
fc = 100  # Cutoff frequency (Hz)
numtaps = 4001  # Number of filter taps (odd number for better performance)

# 1. Generate low-pass FIR filter using window method
taps = get_wsec_filter(numtaps,fs)
n = np.arange(taps.shape[0]) # From -5 to 5
    # Create an array of zeros with the same length as n
impulse_signal = np.zeros_like(n, dtype=float)
impulse_signal[0] = 1

filtered_signal_full = signal.lfilter(taps,1.0,impulse_signal)

compare_spectrums(filtered_signal_full,fs,"Frequency Response")


# %%
print("Shape for wsecimpulse: ",taps.shape)
C_chosen = 50
B = 5

W = format_W(taps,C_chosen)
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

px.line(y=S, title='Singular values of the taps filter ').show()

compare_spectrums(filtered_signal_full,fs,"Frequency Response FULL")

for i in range(3):
    temp = signal.lfilter(C_weights[i,:],1.0,impulse_signal)
    filtered_signal = signal.lfilter(R_weights[i,:],1.0,temp)
    compare_spectrums(filtered_signal,fs,f"Filtered Signal of Branch {i+1}")

#

y = np.zeros(numtaps)
firsvdsec = FIRSVDFilterPy(C_weights, R_weights)
firsvdsec.reset()
y[0] = firsvdsec.filterstep(1.0)
for k in range(1,numtaps):
    y[k] = firsvdsec.filterstep(0.0)
#fig = px.line(title='Impulse response from FIRSVDFilterPy')
#fig.add_scatter(y=taps, name="ideal", mode="lines")
#fig.add_scatter(y=y, name="FIRSVDFilter2", mode="lines")
#fig.show()

compare_spectrums(y,fs,"SVD Implementation")

plt.show()
# %%
