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

n = np.arange(taps.shape[0]) # From -5 to 5
# Create an array of zeros with the same length as n
impulse_signal = np.zeros_like(n, dtype=float)
impulse_signal[0] = 1
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