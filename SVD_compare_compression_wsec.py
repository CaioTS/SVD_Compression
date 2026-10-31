#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

import sys
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
print(current_file_dir)
# Change to that directory
os.chdir(current_file_dir)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(os.listdir('.'))  # This will show all files in current folder

from functions_SVD_compare_wsec import analyze_FIR_compression , get_wsec_filter
# %%
fs = 416

N = np.arange(500,8000,500)
coefs = np.zeros_like(N)
mems = np.zeros_like(N)
ops = np.zeros_like(N)
for n in N: 
    
    wsec = get_wsec_filter(n,fs)
    mems[N == n],coefs[N == n], ops[N == n] = analyze_FIR_compression(None,fs,0.01,wsec,True,0)
    print(n,mems[N == n] ,coefs[N == n],ops[N == n])
# %%
compression_coefs = 100 * (coefs/(N))
compression_mems  = 100 * (mems/(2*N))
compression_ops   = 100 * (ops/(2*N -1))
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Create the figure
fig = go.Figure()

# Add all three traces to the same plot
fig.add_trace(go.Scatter(
    x=N, 
    y=compression_coefs, 
    mode='lines+markers', 
    name='Coefficients Compression',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=N, 
    y=compression_mems, 
    mode='lines+markers', 
    name='Memory Compression',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=N, 
    y=compression_ops, 
    mode='lines+markers', 
    name='Operations Compression',
    line=dict(width=2)
))

# Update layout
fig.update_layout(
    title='Compression Metrics Comparison',
    xaxis_title='Filter Size',
    yaxis_title='Compression Ratio (%)',
    yaxis=dict(range=[0, 100]),
    height=600,
    width=900,
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.2)',
        borderwidth=1
    )
)

# Show the figure
fig.show()

# Create subplots with 1 row and 3 columns
fig = make_subplots(
    rows=1, 
    cols=3,
    subplot_titles=(
        'Coefficients (coefs)', 
        'Memory (mems)', 
        'Operations (ops)'
    ),
    horizontal_spacing=0.1
)

# Add traces to each subplot
fig.add_trace(
    go.Scatter(x=N, y=coefs, mode='lines+markers', name='coefs', line=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=N, y=mems, mode='lines+markers', name='mems', line=dict(color='red')),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=N, y=ops, mode='lines+markers', name='ops', line=dict(color='green')),
    row=1, col=3
)

# Update layout
fig.update_layout(
    title_text='Performance Metrics Analysis',
    height=500,
    width=1200,
    showlegend=True
)

# Update axes labels
fig.update_xaxes(title_text='Filter Size', row=1, col=1)
fig.update_xaxes(title_text='Filter Size', row=1, col=2)
fig.update_xaxes(title_text='Filter Size', row=1, col=3)

fig.update_yaxes(title_text='Number of Coefficients', row=1, col=1)
fig.update_yaxes(title_text='Memory Usage', row=1, col=2)
fig.update_yaxes(title_text='Number of Operations', row=1, col=3)

# Show the figure
fig.show()

# %%
"""
Caso eu tenha uma limitação de memória de 4000: 

meu filtro original seria de 2000 coeficientes, com a SVD, conseguiria
trabalhar com um filtro equivalente ao original de 3000 coeficientes.

Resultando em um maior número de amostras, melhorando o passo do sistema
de controle.


"""
# %%
