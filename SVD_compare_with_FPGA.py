
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
X = np.random.rand(1,8)
H = np.random.rand(8,5)
H2 = np.random.rand(5,2)
Y = X @ H
Y2 = Y @ H2
U , S , VT = np.linalg.svd(H)
US = np.zeros_like(U)

for i in range(len(S)):
    US[:,i] = U[:,i]*S[i]
US = US[:,:len(S)]

print(f"{X.shape = } | {U.shape =} | {US.shape = } |{VT.shape = } |{Y.shape = } |{H.shape = } |")
#%%
Y_test = np.zeros_like(Y)
data_df = []
for i in range(len(S)):
    data_info = []
    data_info.append(i)
    for j in range(len(S)):
        if i == 0 :
            print('Alo',f'{1} * {US[j,i]} = {1 * US[j,i]}')
    USX = X @ US[:,i]
    data_info.append(USX[0])
    temp = np.outer(USX , VT[i,:])
    for t in temp[0,:]:
        data_info.append(t)
    Y_test += temp
    for y in Y_test[0,:]:
        data_info.append(y)
    data_df.append(data_info)
print(Y_test - Y)

df = pd.DataFrame(data_df)
df.columns = ["i" , "ya_i" , "yi_i0" , "yi_i1" , "yi_i2" , "yi_i3" , "yi_i4" , "Y[0]","Y[1]","Y[2]","Y[3]","Y[4]"]
# %%
df
# %%

def float_to_fixed_32(number: float,decimal_bits = 28) -> np.int32:
    """
    Returns: 32-bit signed integer (usable as uint32 in hardware).
    """
    scaled = number * (2**decimal_bits) # Multiply by 2^16
    fixed = int(np.round(scaled))  # Round to nearest integer
    

    # Handle overflow (optional)
    fixed = np.int32(fixed)  # Force 32-bit signed integer
    return fixed

def to_32bit_hex(value):
    """Convert a signed integer to 32-bit hex string (SystemVerilog compatible)."""
    return f"{value & 0xFFFFFFFF:08X}"  # Mask to 32 bits, uppercase, zero-padded

# Original arrays

VT_bit = []
UT_bit = []
for i in range (VT.shape[0]-1,-1,-1) :
    VT_bit.append(np.array(list(map(float_to_fixed_32,VT[i,:]))).tolist())
    UT_bit.append(np.array(list(map(float_to_fixed_32,US[:,i]))).tolist())

print(VT_bit)
# Print VT_i in hex
print("logic signed [31:0] VT [cycle_times-1:0][num_outputs-1:0] = '{")
for val in VT_bit:
    print("{",end='')
    for v in val:
        print(f"32'h{to_32bit_hex(v)}", end=", ")  # SystemVerilog-style format
    print("},\n",end='')

# Print UT_i in hex
print("logic signed [31:0] UT [cycle_times-1:0][num_inputs-1:0] = '{")
for val in UT_bit:
    print("{",end='')
    for v in val:
        print(f"32'h{to_32bit_hex(v)}", end=", ")
    print("},\n",end='')
# %%
print("X (Hex):")
X_bit = np.array(list(map(float_to_fixed_32,X[0,:]))).tolist()
for val in X_bit :
    print(f"32'h{to_32bit_hex(val)}", end=", ")
# %%
print(X)
# %%
df
# %%
df['ya_i'][0]/(2**15)
    # %%
df['ya_i'][4]/(2**14)

# %%
print(0.00699005275964737 * 2**4)
print(-0.0323758870363235 * 2**4)
print(0.00275652855634689 * 2**4)
print(0.00114447996020317 * 2**4)
print(-0.307097282260656  * 2**4)
df
# %%
results = [0.00953686237335205,0.0171590447425842,-0.0112476944923401,0.0138716697692871,-0.0200411677360535]
r_n = []
for r in results:
    r_n.append(r * 2**8)
print(r_n)
    # %%
# %%
U2 , S2 , VT2 = np.linalg.svd(H2)
US2 = np.zeros_like(U2)

for i in range(len(S2)):
    US2[:,i] = U2[:,i]*S2[i]
US2 = US2[:,:len(S2)]
# %%
VT2_bit = []
UT2_bit = []
for i in range (VT2.shape[0]-1,-1,-1) :
    VT2_bit.append(np.array(list(map(float_to_fixed_32,VT2[i,:]))).tolist())
    UT2_bit.append(np.array(list(map(float_to_fixed_32,US2[:,i]))).tolist())

# Print VT_i in hex
print("logic signed [31:0] VT2 [cycle_times-1:0][num_outputs-1:0] = '{")
for val in VT2_bit:
    print("{",end='')
    for v in val:
        print(f"32'h{to_32bit_hex(v)}", end=", ")  # SystemVerilog-style format
    print("},\n",end='')

# Print UT_i in hex
print("logic signed [31:0] UT2 [cycle_times-1:0][num_inputs-1:0] = '{")
for val in UT2_bit:
    print("{",end='')
    for v in val:
        print(f"32'h{to_32bit_hex(v)}", end=", ")
    print("},\n",end='')
# %%
Y2
# %%
Y
# %%
