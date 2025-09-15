"""
Steps:

- Generate a low-pass FIR filter with fs 800 Hz and fc = 100 Hz
- Verify the Transfer functio of this filter
- Test with a 60Hz signal with some white-noise

- Convert to SVD
- Test with no Decomposition
- Test with Decomposition

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# Set random seed for reproducibility
np.random.seed(42)
#%%


# %%

# %%
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


# Download coefficients from Viga
#Metodologia de escolha de construção de filtro via SVD
#Escolher decomposição de rank via 10% do valor de somatório total
def rank_choser(S,threshold):
    sum_s = np.sum(S)
    step = 0
    i = 0
    while (step < (1-threshold)*sum_s):
        step+=S[i]
        i+=1
        #print(sum_s, step, step/sum_s , threshold )
    return i 

def format_W(W,R):
    n_pad = W.shape[0] % R
    W_pad = np.zeros(int(W.shape[0] + (R - n_pad)))
    W_pad[:W.shape[0]] = W
    C = W_pad.shape[0]/R
    lower_dim = min(R,C)
    return  W_pad.reshape((int(lower_dim),-1),order = 'F')


def calc_num_svd_coefs(W,R,threshold):
    W_p = format_W(W,R)
    #if (W_p.shape[1] < R):
    #    W_p = W_p.T
    U , S , VT = np.linalg.svd(W_p)
    R_compressed = rank_choser(S,threshold)
    num_coefs = R_compressed* (U.shape[0] + VT.shape[0])
    num_ops = (2*(U.shape[0] + VT.shape[0])*R_compressed )+ R_compressed - 1 
    #print(f'{S.shape = },{W_p.shape = }, {R_compressed = } , {VT.shape[0] = }, {num_coefs = }')
    return num_coefs ,num_ops, R_compressed
    
def analyze_FIR_compression(file,fs,threshold,taps,notFILE,parameter):

    if (notFILE):
        wsec_impulse = taps
    else:
        wsec_impulse = np.loadtxt(file, delimiter=',')
    
    n_coefs = []
    n_ops = []
    R_comps = []
    n_coefs.append(100000)
    n_coefs.append(100000)
    n_ops.append(100000)
    n_ops.append(100000)
    R_comps.append(100000)
    R_comps.append(100000)

    for R in range(2,501):
        n_coef , n_op , R_comp = calc_num_svd_coefs(wsec_impulse,R,threshold)
        n_coefs.append(n_coef)
        n_ops.append(n_op)
        R_comps.append(R_comp)

    opxcoefs  = np.array(n_coefs) * np.array(n_ops)

    fs = fs
    #Set parameter = 0 to check for Compression with less coefficients, 1 for less operations
    param_chosen = n_coefs if parameter == 0 else n_ops
    
    match parameter:
        case 0:
            param_chosen = n_coefs
        case 1:
            param_chosen = n_ops
        case 2: 
            param_chosen = opxcoefs

    R_chosen = np.argmin(param_chosen)
    num_coef = n_coefs[R_chosen]
    num_op   = n_ops[R_chosen]
    opxcoef  = opxcoefs[R_chosen]
    S_decomp = R_comps[R_chosen]
    plt.plot(param_chosen[2:])
    plt.xscale("log")
    plt.plot(R_chosen,param_chosen[R_chosen],"ro")
    plt.show()
    print(R_chosen,S_decomp)

    n = np.arange(wsec_impulse.shape[0]) # From -5 to 5
    # Create an array of zeros with the same length as n
    impulse_signal = np.zeros_like(n, dtype=float)
    impulse_signal[0] = 1

    full_size = wsec_impulse.shape[0]

    filtered_signal = signal.lfilter(wsec_impulse,1.0,impulse_signal)
    FIR_ch = format_W(wsec_impulse,R_chosen)


    U , S ,VT = np.linalg.svd(FIR_ch)

    SM = np.zeros((U.shape[0],VT.shape[0]))
    np.fill_diagonal(SM,S)

    US = U @ SM
    #US.reshape(1,63*64)

    kron_total = U.shape[0] * VT.shape[0]
    print(kron_total,R_chosen,S.shape,VT.shape)
    I = np.identity(len(S))
    impulse_filtered = np.zeros_like(impulse_signal)

    for i in range(S_decomp):
        vd = np.kron(VT[i,:],I).reshape(R_chosen*kron_total)
        svd_filt_Vt = signal.lfilter(vd[:kron_total], 1.0, impulse_signal)
        svd_filt_US = signal.lfilter(US[:,i], 1.0, svd_filt_Vt)

        impulse_filtered+=svd_filt_US
    error = compare_spectrums_phase(impulse_filtered,filtered_signal,fs,f"Filters Impulse Response N branches = {S_decomp} N coefs = {num_coef} N ops = {num_op} CxO = {opxcoef}")
    print(error)
    #plt.plot(S)
# %%
analyze_FIR_compression('../../ActVibModules/wsecimpulse.txt',400,0.1,0,False,0)
#analyze_FIR_compression('./ActVibModules/wsecimpulse.txt',400,0.1,0,False,1)
#analyze_FIR_compression('./ActVibModules/wsecimpulse.txt',400,0.1,0,False,2)

# %%
analyze_FIR_compression('./ActVibModules/wfbkimpulse.txt',400,0.01,0,False,0)

# %%
# %%
# Filter parameters
fs = 800  # Sampling frequency (Hz)
fc = 100  # Cutoff frequency (Hz)
numtaps = 4001  # Number of filter taps (odd number for better performance)

# 1. Generate low-pass FIR filter using window method
taps = signal.firwin(numtaps, fc, fs=fs, window='hamming',)
print(f"Generated FIR filter with {numtaps} taps")

analyze_FIR_compression(None,800,0.1,taps,True,2)

# %%
