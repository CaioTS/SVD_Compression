#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq


def compare_spectrums(signal,ref, fs, title):
    n = len(signal)
    yfs = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    plt.plot(xf, 2.0/n * np.abs(yfs[0:n//2]),label = 'Signal Filter Response',linewidth=3)

    n = len(ref)
    yfr = fft(ref)
    plt.plot(xf, 2.0/n * np.abs(yfr[0:n//2]),label= 'Reference Filter Response',linewidth=2)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, fs/2)
    plt.show()
    norm = np.linalg.norm(yfr)
    print(f"{norm = }")
    return abs(np.sum(((yfs - yfr)**2)/len(xf)))/norm


class FIRFilter:
    def __init__(self, mem, ptrw, ptrx):
        self.wptr = ptrw
        self.x = ptrx
        self.N = mem
        self.reset()

    def set_mem(self, mem):
        self.N = mem
        self.reset()

    def reset(self):
        self.y = 0
        for k in range(self.N):
            self.x[k] = 0
        self.ptr = self.N - 1

    def filter(self, xn):
        self.ptr += 1
        if self.ptr >= self.N:
            self.ptr = 0
        self.x[self.ptr] = xn
        self.y = 0
        for k in range(self.N):
            self.y += self.x[(self.ptr - k + self.N) % self.N] * self.wptr[k]
        return self.y
    
#%%
# Filter parameters
fs = 800  # Sampling frequency (Hz)
fc = 100  # Cutoff frequency (Hz)
numtaps = 4001  # Number of filter taps (odd number for better performance)

# 1. Generate low-pass FIR filter using window method
taps = signal.firwin(numtaps, fc, fs=fs, window='hamming')
print(f"Generated FIR filter with {numtaps} taps")

n = np.arange(taps.shape[0]) 
# Create an array of zeros with the same length as n
impulse_signal = np.zeros_like(n, dtype=float)
impulse_signal[0] = 0.5*numtaps
full_size = taps.shape[0]
filtered_signal = signal.lfilter(taps,1.0,impulse_signal)

#Test with FIR class
mem_w  = taps
mem_x  = np.zeros_like(taps)

N = len(mem_w)
fir = FIRFilter(N,mem_w,mem_x)

y = []
for x in impulse_signal:
    y.append(fir.filter(x))

compare_spectrums(filtered_signal,np.array(y),fs,"Test with FIR class")
# %%

"""
Memory Structure:
ptrW  = (C + R ) B times
ptrx  = (C*R) 
N will be a list of all filters size in the same order as stored in memory

Get offset for R sparsity in ptrx 

Filter will be a double loop, one counts the branchs, 
the other makes the sparse and small filter calculations, 
sum them at the end of each bigger loop.

I can call the FIRFILTER CLASS and operate with it (and just control 
list of these classes)

"""

class FIRSVDFilter:
    def __init__(self,mem, ptrw, ptrx, R, C, B):
        self.R  =  R # Rows
        self.C  =  C # Collunms
        self.B  =  B # Branches
        self.filters = [] #Stores all filters class
        self.v_ptr   = [] # Buffers of intermdiate filter exit
        self.y  =  0
        self.N = mem #Full size of weights

        for i in range(B):
            self.filters.append(FIRFilter(C,ptrw[i*(R+C):i*(R+C) + C],ptrx))
            
            self.v_ptr.append(np.zeros(R))
            self.filters.append(FIRFilter(R,ptrw[i*(R + C) + C :i*(R + C) + C + R],self.v_ptr[-1]))

        self.reset()

    def set_mem(self, mem):
        self.N = mem
        self.reset()

    def reset(self):
        self.y = 0

    def filter(self, xn):
        self.y = 0
        for i in range(self.B):
           tmp = self.filters[2*i].filter(xn)
           self.y += self.filters[2*i+1].filter(tmp)
        return self.y
           
# %%
"""
- Test a dual filter with  random weights and see if this 
implementation matches the one with scipy

"""

# Filter parameters
fs = 800  # Sampling frequency (Hz)
fc1 = 100  # Cutoff frequency (Hz)
numtaps1 = 201  # Number of filter taps (odd number for better performance)

fc2 = 250
numtaps2 = 101

#  Generate low-pass FIR filter using window method
taps_low  = signal.firwin(numtaps1, fc1, fs=fs, window='hamming')
taps_low1 = signal.firwin(numtaps2, fc1 + 50, fs=fs, window='hamming')
#  Generate High-pass FIR filter using window method
taps_high  = signal.firwin(numtaps1, fc2, fs=fs, window='hamming',pass_zero=False)
taps_high1 = signal.firwin(numtaps2, fc2 - 50, fs=fs, window='hamming',pass_zero=False)

n = np.arange(taps_low.shape[0]) 
# Create an array of zeros with the same length as n
impulse_signal = np.zeros_like(n, dtype=float)
impulse_signal[0] = 0.5*numtaps

filtered_low_signal_0 = signal.lfilter(taps_low,1.0,impulse_signal)
filtered_low_signal = signal.lfilter(taps_low,1.0,filtered_low_signal_0)

filtered_high_signal0 = signal.lfilter(taps_high,1.0,impulse_signal)
filtered_high_signal = signal.lfilter(taps_high,1.0,filtered_high_signal0)

filtered_signal = filtered_low_signal + filtered_high_signal
compare_spectrums(impulse_signal,filtered_signal,fs,"Theoric")
compare_spectrums(filtered_high_signal,filtered_low_signal,fs,"Branches")

ptrw = np.concatenate((taps_low,taps_low1,taps_high,taps_high1))
ptrw_x  = np.zeros_like(taps)


print(len(ptrw))
filt = FIRSVDFilter(len(ptrw),ptrw,ptrw_x,numtaps2,numtaps1,2)


y = []
for x in impulse_signal:
    y.append(filt.filter(x))

compare_spectrums(filtered_signal,np.array(y),fs,"Test with FIRSVD class")

# %%
"""
Remember to create outside of the class the proper buffer aligment

Organize weights from SVD as the class needs

"""