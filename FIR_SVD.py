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

# Filter parameters
fs = 800  # Sampling frequency (Hz)
fc = 100  # Cutoff frequency (Hz)
numtaps = 4001  # Number of filter taps (odd number for better performance)

# 1. Generate low-pass FIR filter using window method
taps = signal.firwin(numtaps, fc, fs=fs, window='hamming',)
print(f"Generated FIR filter with {numtaps} taps")

# 2. Verify the transfer function
# Frequency response
w, h = signal.freqz(taps, fs=fs)

plt.figure(figsize=(12, 8))

# Plot frequency response
plt.subplot(2, 2, 1)
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.axvline(fc, color='red', linestyle='--', alpha=0.7, label=f'Cutoff ({fc} Hz)')
plt.title('Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.legend()
plt.xlim(0, fs/2)

# Plot phase response
plt.subplot(2, 2, 2)
plt.plot(w, np.unwrap(np.angle(h)))
plt.title('Phase Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.xlim(0, fs/2)

plt.tight_layout()
plt.show()
#%%
# 3. Test with a 60Hz signal with white noise
# Generate test signal
duration = 8.0  # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Create 60Hz sine wave with some noise
signal_60hz = 1.0 * np.sin(2 * np.pi * 60 * t)
noise = 0.2 * np.random.randn(len(t))
test_signal = signal_60hz + noise
# Apply the filter
filtered_signal = signal.lfilter(taps, 1.0, test_signal)

# Apply group delay compensation (shift signal by half the filter length)
group_delay = (numtaps - 1) // 2
filtered_signal_compensated = np.roll(filtered_signal, -group_delay)
filtered_signal_compensated[-group_delay:] = 0  # Zero out the invalid samples

filtered_signal_compensated_rev = filtered_signal_compensated[0:-group_delay]
test_signal_rev = test_signal[0:-group_delay]
t_rev = t[0:-group_delay]
print(test_signal.shape,filtered_signal.shape)
# Frequency analysis
def plot_spectrum(signal, fs, title):
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    plt.plot(xf, 2.0/n * np.abs(yf[0:n//2]))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0, fs/2)

# Plot results
plt.figure(figsize=(15, 10))

# Time domain signals
plt.subplot(3, 2, 1)
plt.plot(t_rev, test_signal_rev, alpha=0.7, label='Noisy Signal')
#plt.plot(t_rev, signal_60hz, 'r-', linewidth=2, label='Clean 60Hz Signal')
plt.title('Input Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t_rev, filtered_signal_compensated_rev, 'g-', linewidth=2, label='Filtered Signal')
#plt.plot(t, signal_60hz, 'r--', alpha=0.7, label='Original 60Hz Signal')
plt.title('Filtered Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Frequency domain signals
plt.subplot(3, 2, 3)
plot_spectrum(test_signal_rev, fs, 'Input Signal Spectrum')

plt.subplot(3, 2, 4)
plot_spectrum(filtered_signal_compensated_rev, fs, 'Filtered Signal Spectrum')


plt.tight_layout()
plt.show()

# Calculate SNR improvement
def calculate_snr(signal, noise_component):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise_component**2)
    return 10 * np.log10(signal_power / noise_power)

# Extract noise components
original_noise = test_signal_rev - signal_60hz[0:-group_delay]
filtered_noise = filtered_signal_compensated_rev - signal_60hz[0:-group_delay]

input_snr = calculate_snr(signal_60hz[0:-group_delay], original_noise)
output_snr = calculate_snr(signal_60hz[0:-group_delay], filtered_noise)

print(f"Input SNR: {input_snr:.2f} dB")
print(f"Output SNR: {output_snr:.2f} dB")
print(f"SNR Improvement: {output_snr - input_snr:.2f} dB")

# Show filter specifications
print(f"\nFilter Specifications:")
print(f"Sampling Frequency: {fs} Hz")
print(f"Cutoff Frequency: {fc} Hz")
print(f"Number of Taps: {numtaps}")
print(f"Transition Bandwidth: ~{fs/numtaps:.2f} Hz")

# %%
#Organize w array in W matrix with padding, in the collumn separation
print(taps,taps.shape)
print(np.sqrt(4000))
w = taps.copy()
#Taps Size = 4001, better factoring is 63x64
W = np.zeros((63,64))

index = 0
for i in range(64):
    for j in range(63):
        if index < len(w):
            W[j, i] = w[index]
            index += 1

print(W.shape,w.shape)

# %%
#Calculate the SVD of W
U , S , VT = np.linalg.svd(W)

print(U.shape,S.shape,VT.shape)

plt.plot(S)

#%%

#Cada coluna de VT é um filtro FIR 
#Cada coluna de US é um filtro FIR 
SM = np.zeros((U.shape[0],VT.shape[0]))
np.fill_diagonal(SM,S)

print(U.shape,VT.shape,SM.shape)
US = U @ SM
US.reshape(1,63*64)
#US is OK

# Gerar US_k = Sk * U[:,k]
#US = np.zeros_like(U)
#for k,s in enumerate(S):
#    US[:,k] = s * U[:,k]
#Fazer laço para filtrar o sinal x com os respectivos VT e US
I = np.identity(len(S))
VTI = np.kron(VT,I)
print(VT.shape,I.shape, VTI.shape)
test = np.kron(VT[0,:],I)
print(test.shape)
print(test == VTI[0:63,:])

plt.plot(test[0,:])
plt.plot(test[1,:])

# %%
#Filtrar de forma paralela com 
SVD_filtered = np.zeros_like(test_signal)
for i in range(len(S)):
    vd = VTI[64*i:64*i + 63,:].reshape(63*4032)
    svd_filt_Vt = signal.lfilter(vd, 1.0, test_signal)
    svd_filt_US = signal.lfilter(US[:,i], 1.0, svd_filt_Vt)

    SVD_filtered+=svd_filt_US
plt.plot(SVD_filtered)
#plt.plot(test_signal)

# %%
svd_group_delay = 64 // 2 +  63  // 2
print(svd_group_delay)
#plot_spectrum(test_signal,fs,'Original Signal SNR')
plot_spectrum(filtered_signal_compensated_rev, fs, 'Filtered Signal Spectrum')
plot_spectrum(SVD_filtered[2000:5000], fs, 'Filtered Signal Spectrum')


# %%
svd_test_signal_rev = test_signal[svd_group_delay:]

original_noise = svd_test_signal_rev - signal_60hz[svd_group_delay:]
filtered_noise = SVD_filtered[svd_group_delay:] - signal_60hz[svd_group_delay:]

input_snr = calculate_snr(signal_60hz[group_delay:], original_noise)
output_snr = calculate_snr(signal_60hz[group_delay:], filtered_noise)

SNR_improv = output_snr - input_snr
print(SNR_improv)
# %%
# %%
# Define the range for the discrete time axis
n = np.arange(4001) # From -5 to 5
# Create an array of zeros with the same length as n
impulse_signal = np.zeros_like(n, dtype=float)
impulse_signal[0] = 1

#plt.plot(n,impulse_signal)
impulse_filtered = np.zeros_like(impulse_signal)
for i in range(len(S)):
    vd = np.kron(VT[i,:],I).reshape(63*4032)
    svd_filt_Vt = signal.lfilter(vd, 1.0, impulse_signal)
    svd_filt_US = signal.lfilter(US[:,i], 1.0, svd_filt_Vt)

    impulse_filtered+=svd_filt_US
filtered_signal = signal.lfilter(taps,1,impulse_signal)
#plt.plot(impulse_filtered)
#%%
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

err = compare_spectrums(impulse_filtered,filtered_signal,fs,"Filters Impulse Response")
print(err)


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


# %%
# Make the Error function x rank decomposition
error = []
for j in range(len(S),1,-6):
    impulse_filtered = np.zeros_like(impulse_signal)
    for i in range(j):
        vd = np.kron(VT[i,:],I).reshape(63*4032)
        svd_filt_Vt = signal.lfilter(vd, 1.0, impulse_signal)
        svd_filt_US = signal.lfilter(US[:,i], 1.0, svd_filt_Vt)

        impulse_filtered+=svd_filt_US
    error.append(compare_spectrums(impulse_filtered,filtered_signal,fs,f"Filters Impulse Response S = {j}"))
# %%
plt.plot(error)
# %%
#From 10 to 2 for faster analisys
error = []
for j in range(10,1,-1):
    impulse_filtered = np.zeros_like(impulse_signal)
    for i in range(j):
        vd = np.kron(VT[i,:],I).reshape(63*4032)
        svd_filt_Vt = signal.lfilter(vd, 1.0, impulse_signal)
        svd_filt_US = signal.lfilter(US[:,i], 1.0, svd_filt_Vt)

        impulse_filtered+=svd_filt_US
    error.append(compare_spectrums(impulse_filtered,filtered_signal,fs,f"Filters Impulse Response S = {j}"))

# %%
plt.plot(error)
# %%
#Calculate Compression Num_coef = R^2 + RC
num_coef_tot = US.shape[0]**2 + US.shape[0]*VT.shape[0]
print(num_coef_tot)
FIR_num_coef = US.shape[0]*VT.shape[0]
SVD_num_coef = np.zeros(US.shape[0])
for i in range(US.shape[0]-1,0,-1):
    SVD_num_coef[i] = i**2 + i*VT.shape[0]
# %%
FIR_num_coef_plot = np.zeros_like(SVD_num_coef)
for f in range(len(FIR_num_coef_plot)):
    FIR_num_coef_plot[f] = FIR_num_coef

plt.plot(SVD_num_coef)
plt.plot(FIR_num_coef_plot)
plt.plot(5,SVD_num_coef[5],'or')
print(SVD_num_coef[2])
# %%
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
    #print(f'{S.shape = },{W_p.shape = }, {R_compressed = } , {VT.shape[0] = }, {num_coefs = }')
    return num_coefs , R_compressed
    

#%%

l2 = int(np.log2(taps.shape[0]))
print(l2)
r_tests = []
for i in range(l2):
    r_tests.append(2**i)
print(r_tests)

n_coefs = []
R_comps = []

n_coefs.append(100000)
n_coefs.append(100000)
R_comps.append(100000)
R_comps.append(100000)
for R in range(2,1025):
    n_coef , R_comp = calc_num_svd_coefs(taps,R,0.05)
    n_coefs.append(n_coef)
    R_comps.append(R_comp)

# %%
plt.xscale('log')
plt.plot(n_coefs[2:])
# %%
R_chosen = np.argmin(n_coefs)
S_decomp = R_comps[R_chosen]
n_coefs[58]

# %%
FIR_ch = format_W(taps,R_chosen)
filtered_signal = signal.lfilter(taps,1.0,impulse_signal)
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
    svd_filt_Vt = signal.lfilter(vd, 1.0, impulse_signal)
    svd_filt_US = signal.lfilter(US[:,i], 1.0, svd_filt_Vt)
    
    impulse_filtered+=svd_filt_US

compare_spectrums(impulse_filtered,filtered_signal,fs,f"Filters Impulse Response N branches = {S_decomp}")
# %%

# %%
# Download coefficients from Viga
def analyze_FIR_compression(file,fs,threshold,taps,notFILE):

    if (notFILE):
        wsec_impulse = taps
    else:
        wsec_impulse = np.loadtxt(file, delimiter=',')
    
    n_coefs = []
    R_comps = []
    n_coefs.append(100000)
    n_coefs.append(100000)
    R_comps.append(100000)
    R_comps.append(100000)

    for R in range(2,501):
        n_coef , R_comp = calc_num_svd_coefs(wsec_impulse,R,threshold)
        n_coefs.append(n_coef)
        R_comps.append(R_comp)
    plt.plot(n_coefs[2:])
    plt.show()
    fs = fs
    R_chosen = np.argmin(n_coefs)
    num_coef = np.min(n_coefs)
    S_decomp = R_comps[R_chosen]

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
        svd_filt_Vt = signal.lfilter(vd, 1.0, impulse_signal)
        svd_filt_US = signal.lfilter(US[:,i], 1.0, svd_filt_Vt)

        impulse_filtered+=svd_filt_US
    error = compare_spectrums_phase(impulse_filtered,filtered_signal,fs,f"Filters Impulse Response N branches = {S_decomp} N coefs = {num_coef}")
    print(error)
    plt.plot(S)
# %%
analyze_FIR_compression('ActVibModules/wsecimpulse.txt',400,0.1,0,False)
# %%
analyze_FIR_compression('ActVibModules/wfbkimpulse.txt',400,0.01,0,False)

# %%
analyze_FIR_compression(None,800,0.1,taps,True)
# %%
