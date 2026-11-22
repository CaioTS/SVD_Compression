# %%
"""
Joint SVD compression of the secondary and feedback paths.

Both paths are responses of the same beam to the same actuator, measured at
different points. They share the poles (modes) and differ only in the modal
residues (mode shapes at the sensors). Therefore, after stacking each impulse
response into an L x K matrix, both matrices share the same column and row
subspaces, and a single SVD basis can be used for both filters:

    X_s = U M_s V^T,   X_f = U M_f V^T        (shared U, V; small r x r cores)

Going one step further, the cores can be simultaneously block-diagonalized,
which splits the filters into modal branches shared by both paths:

    X_s = A B^T,       X_f = A G B^T          (G: block diagonal, per mode)

For proportional damping G is diagonal, with one real gain per mode equal to
the ratio of the mode shapes at the reference and error sensors.

The first `headlen` samples are implemented as a short direct FIR, since they
are not part of the sum of exponentials (feedthrough / finite-difference
transient), and the stacked matrix is built without zero padding. With this,
the tail has rank exactly 2M.
"""
import os
import numpy as np
import scipy.linalg as sla
from scipy.signal import lfilter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from CantileverBeam import CantileverBeam

os.makedirs("figs", exist_ok=True)

def save_fig(fig, path, show=True):
    """Same template as SVD_FPGA/python/methodology.py: removes the title (the
    dissertation uses the LaTeX caption), writes the figure and shows it.
    Subplot titles (panel labels) are kept. PDF export needs kaleido
    (e.g. `uv run --with kaleido python Joint_SVD_Paths.py`)."""
    fig.update_layout(title_text=None, margin=dict(t=40))
    fig.write_image(path)
    if show:
        fig.show()

# %% General Definitions (same configuration as SVD_FPGA/python/methodology.py):

fs = 832.0 # Sampling frequency in Hertz
nmodes = 3
dampingfactors = [0.01] * nmodes
noisestd = 0.0 # Measurement noise in the impulse responses (0.0 for exact rank)

nsettle = 2751 # FIR length for 5% settling at fs = 832 Hz (eq. n-coef)
headlen = 2 # Direct FIR taps for the non-modal initial samples
rank = 2 * nmodes # Number of SVD branches (2 per mode)

# %% Beam definitions:

npoints = 100 # Number of points in the beam (same as methodology.py)
beamlength = 0.58 # Length of the beam in meters
beamwidth = 0.05 # Width of the beam in meters
beamthickness = 0.006 # Thickness of the beam in meters

# Positions of sensors and forces:
controlpos = 60 # Position of the control force (34.8 cm)
errorpos = 95 # Position of the error acceleration measurement (55.1 cm)
referencepos = 75 # Position of the reference acceleration measurement (43.5 cm)


def impulse_responses(beam, nsamples):
    """Secondary (actuator -> error) and feedback (actuator -> reference) paths."""
    beam.reset()
    hs = np.zeros(nsamples)
    hf = np.zeros(nsamples)
    for n in range(nsamples):
        beam.setforce(controlpos, 1.0 if n == 0 else 0.0)
        beam.update()
        hs[n] = beam.getaccel(errorpos)
        hf[n] = beam.getaccel(referencepos)
    return hs, hf


# Stacked matrix dimensions. The tail is extended with real samples up to L*K
# (no zero padding), so that every column is a segment of the same exponentials.
# Same rule as choose_RC(mode="extend") in SVD_FPGA/python/aux_functions.py:
# smallest L*K >= ntail with L >= K and L/K <= 1.5 (ties: smallest L + K).
def choose_LK(n, max_ratio=1.5):
    best = None
    for k in range(int(np.ceil(np.sqrt(n / max_ratio))), int(np.ceil(np.sqrt(n))) + 1):
        l = int(np.ceil(n / k))
        if l < k or l / k > max_ratio:
            continue
        key = (l * k - n, l + k)
        if best is None or key < best[0]:
            best = (key, l, k)
    return best[1], best[2]

ntail = nsettle - headlen
L, K = choose_LK(ntail) # 55 x 50 at fs = 832 Hz
nfir = headlen + L * K

cbeam = CantileverBeam(npoints=npoints, width=beamwidth, thickness=beamthickness,
                        length=beamlength, Tsampling=1.0/fs, nmodes=nmodes,
                        damp=dampingfactors, noisestd=noisestd)
print("Natural frequencies are:\n",
      ",\n".join(cbeam.freqsHz.astype(str).tolist()),
      " (all in Hz).")
hs, hf = impulse_responses(cbeam, nfir)
if noisestd > 0:
    cbeam.noisestd = 0.0
    hs_clean, hf_clean = impulse_responses(cbeam, nfir)
else:
    hs_clean, hf_clean = hs, hf
print(f"FIR length: {nfir} = {headlen} (head) + {L} x {K} (tail)")


# %% Helper functions:

def stack(h):
    """Stacks h in blocks of L samples: X[l, m] = h[m*L + l]."""
    return h.reshape(K, L).T

def unstack(X):
    return X.T.ravel()

def truncated_svd(X, r):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :r], s[:r], Vt[:r].T

def relerr(h, happrox):
    return np.linalg.norm(h - happrox) / np.linalg.norm(h)

def upsample(c, factor):
    """Coefficients of c(z^factor)."""
    out = np.zeros((len(c) - 1) * factor + 1)
    out[::factor] = c
    return out


# %% Individual SVD (one compression per path, as in the PDF):

Xs = stack(hs[headlen:])
Xf = stack(hf[headlen:])

Us, ss, Vs = truncated_svd(Xs, rank)
Uf, sf, Vf = truncated_svd(Xf, rank)
hs_ind = np.concatenate([hs[:headlen], unstack((Us * ss) @ Vs.T)])
hf_ind = np.concatenate([hf[:headlen], unstack((Uf * sf) @ Vf.T)])


# %% Joint SVD: shared U and V, one small core per path.

# Each path is normalized so that both have the same weight in the shared basis.
Xs_n = Xs / np.linalg.norm(Xs)
Xf_n = Xf / np.linalg.norm(Xf)
U = np.linalg.svd(np.hstack([Xs_n, Xf_n]), full_matrices=False)[0][:, :rank]
V = np.linalg.svd(np.hstack([Xs_n.T, Xf_n.T]), full_matrices=False)[0][:, :rank]
Ms = U.T @ Xs @ V
Mf = U.T @ Xf @ V

hs_joint = np.concatenate([hs[:headlen], unstack(U @ Ms @ V.T)])
hf_joint = np.concatenate([hf[:headlen], unstack(U @ Mf @ V.T)])


# %% Modal form: simultaneous block-diagonalization of the two cores.

# Ms = P Cs Q^T and Mf = P Cf Q^T with Cs, Cf diagonal per pole, so
# T = Ms^-1 Mf is diagonalized by the modal basis. Its eigenvalues are the
# ratios between the residues of the feedback and secondary paths.
T = np.linalg.solve(Ms, Mf)
evals, evecs = sla.eig(T)
order = np.argsort(evals.real)
G, W = sla.cdf2rdf(evals[order], evecs[:, order]) # Real block-diagonal form

A = U @ Ms @ W # L x r, absorbs the secondary path gains
B = V @ np.linalg.inv(W).T # K x r
# X_s = A B^T  and  X_f = A G B^T

# Removes round-off entries of G (they would count as extra branches):
G[np.abs(G) < 1e-9 * np.abs(G).max()] = 0.0

# Groups branches belonging to the same mode (blocks of G):
offdiag = np.abs(G - np.diag(np.diag(G))) > 0
groups = []
for i in range(rank):
    if groups and (offdiag[i, groups[-1][-1]] or
                   np.isclose(G[i, i], G[groups[-1][-1], groups[-1][-1]], rtol=1e-4)):
        groups[-1].append(i)
    else:
        groups.append([i])

proportional = not offdiag.any()
print("\nModal branches (blocks of G):", groups)
print("G is diagonal (proportional damping):", proportional)

# Identifies which beam mode each group represents and compares the gains with
# the mode shape ratio at the sensors.
freqgrid = np.fft.rfftfreq(1 << 14, 1/fs)
for grp in groups:
    gk = np.zeros(L * K)
    for i in grp:
        gk += unstack(np.outer(A[:, i], B[:, i]))
    fpeak = freqgrid[np.argmax(np.abs(np.fft.rfft(gk, 1 << 14)))]
    k = np.argmin(np.abs(np.abs(cbeam.freqsHz - fs*np.round(cbeam.freqsHz/fs)) - fpeak))
    ratio = cbeam.vmod[referencepos, k] / cbeam.vmod[errorpos, k]
    print(f"  branches {grp}: peak at {fpeak:6.2f} Hz (mode {k+1}, {cbeam.freqsHz[k]:.2f} Hz)"
          f" | gain(s) {np.round(np.diag(G)[grp], 4)} | phi_ref/phi_err = {ratio:.4f}")


# %% Streaming implementations and verification against direct convolution:

class LowRankFIR:
    """
        h = head + z^-P * sum_i a_i(z) * sum_j C[i,j] b_j(z^L)
        Implemented as: sparse b-bank on the input -> core C -> a-bank.
    """
    def __init__(self, head, A, C, B):
        self.head, self.A, self.C, self.B = head, A, C, B

    def filter(self, x):
        P = len(self.head)
        y = lfilter(self.head, 1.0, x)
        xd = np.concatenate([np.zeros(P), x[:len(x)-P]])
        w = np.array([lfilter(upsample(self.B[:, j], L), 1.0, xd) for j in range(self.B.shape[1])])
        z = self.C @ w
        for i in range(self.A.shape[1]):
            y += lfilter(self.A[:, i], 1.0, z[i])
        return y


class SharedModalPair:
    """
        Both paths fed by the SAME input (e.g. online identification with
        auxiliary noise, or modified FxLMS with S(z) applied to y(n)).
        The modal bank is computed once; each path only weights its outputs.
    """
    def __init__(self, heads, A, G, B, groups):
        self.heads, self.A, self.G, self.B, self.groups = heads, A, G, B, groups
        # Pairs (i, j) of a_i(z) b_j(z^L) needed by any of the paths:
        self.pairs = [(i, j) for grp in groups for i in grp for j in grp
                      if i == j or abs(G[i, j]) > 0]

    def filter(self, x):
        P = len(self.heads[0])
        xd = np.concatenate([np.zeros(P), x[:len(x)-P]])
        w = {j: lfilter(upsample(self.B[:, j], L), 1.0, xd) for j in range(self.B.shape[1])}
        c = {(i, j): lfilter(self.A[:, i], 1.0, w[j]) for (i, j) in self.pairs}
        ys = lfilter(self.heads[0], 1.0, x) + sum(c[(i, i)] for i in range(self.A.shape[1]))
        yf = lfilter(self.heads[1], 1.0, x) + sum(self.G[i, j] * c[(i, j)] for (i, j) in self.pairs)
        return ys, yf

    def macs(self):
        return self.B.size + len(self.pairs) * L + len(self.pairs) + sum(len(h) for h in self.heads)


rng = np.random.default_rng(0)
x1 = rng.standard_normal(4 * nfir) # Reference signal (input of S)
x2 = rng.standard_normal(4 * nfir) # Control signal (input of F)

# Different inputs (feedforward control with feedback neutralization):
# coefficients A, B are shared; each path keeps its own (block-diagonal) core.
filt_s = LowRankFIR(hs[:headlen], A, np.eye(rank), B)
filt_f = LowRankFIR(hf[:headlen], A, G, B)
# The structured filters must match a direct convolution with the jointly
# compressed impulse responses (exact re-parametrization of U Ms V^T, U Mf V^T).
err_s = relerr(np.convolve(hs_joint, x1)[:len(x1)], filt_s.filter(x1))
err_f = relerr(np.convolve(hf_joint, x2)[:len(x2)], filt_f.filter(x2))

# Same input: the modal bank is shared, each path costs only its gains.
pair = SharedModalPair([hs[:headlen], hf[:headlen]], A, G, B, groups)
ys, yf = pair.filter(x1)
err_s_same = relerr(np.convolve(hs_joint, x1)[:len(x1)], ys)
err_f_same = relerr(np.convolve(hf_joint, x1)[:len(x1)], yf)


# %% Summary:

nG = int(np.count_nonzero(np.abs(G) > 0))
cost = {
    "FIR (two full filters)":
        (2 * nfir, 2 * nfir),
    "Individual SVD (one basis per path)":
        (2 * (rank * (L + K) + headlen), 2 * (rank * (L + K) + headlen)),
    "Joint SVD, shared U,V + r x r cores":
        (rank * (L + K) + 2 * rank**2 + 2 * headlen, 2 * (rank * (L + K) + rank**2 + headlen)),
    "Joint modal, different inputs":
        (rank * (L + K) + nG + 2 * headlen, 2 * (rank * (L + K) + headlen) + nG),
    "Joint modal, same input":
        (rank * (L + K) + nG + 2 * headlen, pair.macs()),
}
print(f"\nRank {rank}, L = {L}, K = {K}")
print(f"Relative L2 error vs. measured FIR:")
print(f"  individual SVD : S {relerr(hs, hs_ind):.1e}   F {relerr(hf, hf_ind):.1e}")
print(f"  joint SVD      : S {relerr(hs, hs_joint):.1e}   F {relerr(hf, hf_joint):.1e}")
if noisestd > 0:
    print(f"Relative L2 error vs. noiseless response:")
    print(f"  measured FIR   : S {relerr(hs_clean, hs):.1e}   F {relerr(hf_clean, hf):.1e}")
    print(f"  individual SVD : S {relerr(hs_clean, hs_ind):.1e}   F {relerr(hf_clean, hf_ind):.1e}")
    print(f"  joint SVD      : S {relerr(hs_clean, hs_joint):.1e}   F {relerr(hf_clean, hf_joint):.1e}")
print(f"Streaming check (vs. np.convolve with the joint compressed responses):")
print(f"  different inputs: S {err_s:.1e}   F {err_f:.1e}")
print(f"  same input      : S {err_s_same:.1e}   F {err_f_same:.1e}")
print(f"\n{'Implementation':38s} {'Coefficients':>12s} {'MACs/sample':>12s}")
for name, (ncoef, nmac) in cost.items():
    print(f"{name:38s} {ncoef:12d} {nmac:12d}")

# Same metric as the dissertation (eq. recursos-fir and recursos-svd), for the
# two paths together, each path filtering its own input signal. The head adds
# d coefficients, 2d memory elements and 2d operations per path.
def res_branches(B, extra_coefs=0, extra_ops=0):
    """(N_C, N_M, N_O) of one FIR-SVD path: B branches, head, extra core."""
    nc = B * (L + K) + headlen
    nm = L * K + B * (L + K) + K + B * L + 2 * headlen
    no = 2 * B * (L + K) + (B - 1) + 2 * headlen
    return nc + extra_coefs, nm + extra_coefs, no + extra_ops

nshared = rank * (L + K)
fir1 = (nsettle, 2 * nsettle, 2 * nsettle - 1)
ind1 = res_branches(rank)
core_ops = rank**2 + rank * (rank - 1) # r x r core: r^2 mult., r(r-1) sums
joint_core = [res_branches(rank, rank**2, core_ops)[i] * 2 - (nshared if i < 2 else 0) for i in range(3)]
joint_modal = [ind1[i] * 2 + nG - (nshared if i < 2 else 0) for i in range(3)]
resources = {
    "FIR": [2 * v for v in fir1],
    "Individual FIR-SVD": [2 * v for v in ind1],
    "Joint SVD (shared U, V + cores)": joint_core,
    "Joint modal (shared A, B + G)": joint_modal,
}
print(f"\nResources, dissertation metric (two paths together):")
print(f"{'Implementation':34s} {'N_C':>7s} {'N_M':>7s} {'N_O':>7s}")
for name, (nc, nm, no) in resources.items():
    print(f"{name:34s} {nc:7d} {nm:7d} {no:7d}")
print("Subspace check: max principal angle U_s x U_f = "
      f"{np.degrees(sla.subspace_angles(Us, Uf).max()):.1e} deg, "
      f"V_s x V_f = {np.degrees(sla.subspace_angles(Vs, Vf).max()):.1e} deg")


# %% Singular values: individual vs. joint, with and without the head split.

path_pt = {"Secondary": "Caminho secundário", "Feedback": "Caminho de feedback"}
pathcolors = {"Secondary": "#1f77b4", "Feedback": "#ff7f0e", "Joint": "#2ca02c"}

def normsv(X):
    s = np.linalg.svd(X, compute_uv=False)
    return s / s[0]

def stack_padded(h):
    # Previous stacking (choose_RC "pad"): no head split, zero padding up to L*K.
    Kp = int(np.sqrt(nsettle))
    Lp = int(np.ceil(nsettle / Kp))
    x = np.zeros(Lp * Kp)
    x[:nsettle] = h[:nsettle]
    return x.reshape(Kp, Lp).T

nsv = 12
idx = np.arange(1, nsv + 1)
Xs_p, Xf_p = stack_padded(hs), stack_padded(hf)
svsets = {
    "padded": {
        "Secondary": normsv(Xs_p),
        "Feedback": normsv(Xf_p),
        "Joint": normsv(np.hstack([Xs_p / np.linalg.norm(Xs_p), Xf_p / np.linalg.norm(Xf_p)])),
    },
    "head": {
        "Secondary": normsv(Xs),
        "Feedback": normsv(Xf),
        "Joint": normsv(np.hstack([Xs_n, Xf_n])),
    },
}
svnames = {"Secondary": "Secundário", "Feedback": "<i>Feedback</i>", "Joint": "Conjunta [X<sub>s</sub> X<sub>f</sub>]"}
fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                    subplot_titles=["Resposta completa, com zeros",
                                    f"Cabeça (d = {headlen}) + cauda, sem zeros"])
for col, key in enumerate(["padded", "head"], 1):
    for name, s in svsets[key].items():
        fig.add_trace(go.Scatter(x=idx, y=np.maximum(s[:nsv], 1e-17), mode="lines+markers",
                                 name=svnames[name], legendgroup=name, showlegend=(col == 1),
                                 line=dict(color=pathcolors[name], width=3 if name == "Joint" else 1.5,
                                           dash="dash" if name == "Joint" else "solid"),
                                 marker=dict(color=pathcolors[name])), row=1, col=col)
    fig.add_vline(x=rank + 0.5, line=dict(color="gray", dash="dot"), row=1, col=col)
fig.update_yaxes(type="log", range=[-17, 0.5])
fig.update_yaxes(title_text="σ<sub>k</sub> / σ<sub>1</sub>", row=1, col=1)
fig.update_xaxes(title_text="Índice k")
fig.update_layout(template="plotly_white", width=1000, height=450)
save_fig(fig, "figs/Joint_S_values.pdf")


# %% Frequency responses of both paths and errors of the joint compression:

nfft = 1 << 15
freqs = np.fft.rfftfreq(nfft, 1/fs)
# The relative complex error 20*log10(|Ha - H| / |H|) includes magnitude and
# phase errors and keeps the curves distinguishable even when they are tiny.
# Bins where the error is exactly zero are floored at 1e-16 (-320 dB).
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("Magnitude",
                                    "Erro relativo |H<sub>c</sub> − H| / |H|",
                                    "Erro de fase"))
labpt = {"individual": "SVD individual", "joint": "SVD conjunta"}
for name, h, hind, hjoint in [("Secondary", hs, hs_ind, hs_joint), ("Feedback", hf, hf_ind, hf_joint)]:
    H = np.fft.rfft(h, nfft)
    fig.add_trace(go.Scatter(x=freqs, y=20*np.log10(np.abs(H)), name=f"{path_pt[name]} (FIR)",
                             line=dict(color=pathcolors[name], width=6), opacity=0.35), row=1, col=1)
    for lab, ha, dash in [("individual", hind, "dash"), ("joint", hjoint, "dot")]:
        Ha = np.fft.rfft(ha, nfft)
        style = dict(color=pathcolors[name], dash=dash, width=1.5)
        fig.add_trace(go.Scatter(x=freqs, y=20*np.log10(np.abs(Ha)), name=f"{path_pt[name]}, {labpt[lab]}",
                                 legendgroup=f"{name}{lab}", line=style, showlegend=False),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=freqs, y=20*np.log10(np.maximum(np.abs(Ha - H) / np.abs(H), 1e-16)),
                                 name=f"{path_pt[name]}, {labpt[lab]}", legendgroup=f"{name}{lab}",
                                 line=style), row=2, col=1)
        fig.add_trace(go.Scatter(x=freqs, y=np.degrees(np.angle(Ha / H)), name=f"{path_pt[name]}, {labpt[lab]}",
                                 legendgroup=f"{name}{lab}", line=style, showlegend=False),
                      row=3, col=1)
fig.update_yaxes(title_text="dB", row=1, col=1)
fig.update_yaxes(title_text="dB", row=2, col=1)
fig.update_yaxes(title_text="Graus", exponentformat="power", row=3, col=1)
fig.update_yaxes(range=[-330, -225], row=2, col=1) # floor at 1e-16 (-320 dB)
fig.update_xaxes(title_text="Frequência (Hz)", row=3, col=1)
fig.update_layout(template="plotly_white", width=900, height=850)
save_fig(fig, "figs/Joint_freq_error.pdf")


# %% Shared modal branches and the gain applied by each path:

H = np.fft.rfft(hs, nfft)
fig = go.Figure()
fig.add_trace(go.Scatter(x=freqs, y=20*np.log10(np.abs(np.fft.rfft(hs[headlen:], nfft))),
                         name="Cauda do caminho secundário", line=dict(color="lightgray", width=5)))
for grp in groups:
    gk = np.zeros(L * K)
    for i in grp:
        gk += unstack(np.outer(A[:, i], B[:, i]))
    Gk = np.fft.rfft(gk, nfft)
    fpk = freqgrid[np.argmax(np.abs(np.fft.rfft(gk, 1 << 14)))]
    fig.add_trace(go.Scatter(x=freqs, y=20*np.log10(np.abs(Gk)), line=dict(width=1.5),
                             name=f"Ramos {', '.join(str(i+1) for i in grp)} ({fpk:.1f} Hz), "
                                  f"ganho do <i>feedback</i> = {np.diag(G)[grp[0]]:.3f}".replace(".", ",")))
fig.update_layout(xaxis_title="Frequência (Hz)", yaxis_title="Magnitude (dB)",
                  template="plotly_white", width=800, height=450,
                  legend=dict(x=0.99, y=0.99, xanchor="right", bgcolor="rgba(255,255,255,0.8)"))
save_fig(fig, "figs/Joint_modal_branches.pdf")

# %%
