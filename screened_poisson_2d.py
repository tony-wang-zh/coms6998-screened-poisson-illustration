import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Screened Poisson 2D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0a0a0f;
    color: #e8e4dc;
  }

  .stApp { background: #0a0a0f; }

  section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e30;
  }

  h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    letter-spacing: -0.03em;
  }

  .plot-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6b6b8a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }

  .stButton>button {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    border-radius: 2px;
    transition: all 0.15s;
  }

  div[data-testid="stSlider"] label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #9090b0;
  }

  .metric-box {
    background: #12121e;
    border: 1px solid #1e1e30;
    border-radius: 4px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #6b6b8a;
    margin-bottom: 8px;
  }

  .section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a4a6a;
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e1e30;
  }

  .status-ready  { color: #4ade80; }
  .status-idle   { color: #6b6b8a; }
  .status-solved { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ──────────────────────────────────────────────────────
for k, v in [
    ("points", None), ("normals", None), ("locked", False),
    ("chi", None), ("boundary_pts", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Curve generators ────────────────────────────────────────────────────────
def generate_circle(n, noise):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 + noise * np.random.randn(n)
    pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
    # outward normals
    nrm = np.column_stack([np.cos(t), np.sin(t)])
    return pts, nrm

def generate_sine(n, noise):
    x = np.linspace(-np.pi, np.pi, n)
    y = np.sin(x) + noise * np.random.randn(n)
    pts = np.column_stack([x, y])
    # normal = (-dy/dx, 1) normalised
    dydx = np.cos(x)
    raw = np.column_stack([-dydx, np.ones(n)])
    nrm = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return pts, nrm

# ─── Screened Poisson solver (2D FD on a regular grid) ───────────────────────
def solve_screened_poisson(pts, nrms, alpha, bc_type, grid_res=80):
    """
    Solve:   Δχ − α·χ = −α·χ₀    (screened, Poisson-like)
    where χ₀ is the indicator splat from the oriented points.

    Boundary condition on the grid boundary:
      Dirichlet : χ = 0
      Neumann   : ∂χ/∂n = 0  (zero-flux, via ghost cells)

    Returns grid X, Y, Chi, and the extracted 0.5-contour.
    """
    pad = 0.3
    xmin, xmax = pts[:, 0].min() - pad, pts[:, 0].max() + pad
    ymin, ymax = pts[:, 1].min() - pad, pts[:, 1].max() + pad

    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    h_x = xs[1] - xs[0]
    h_y = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys)           # shape (grid_res, grid_res)
    N = grid_res * grid_res

    def idx(i, j):
        return i * grid_res + j          # row-major: i=row(y), j=col(x)

    # ── Build χ₀: splatted indicator from oriented points ──────────────────
    # We approximate the vector field V whose divergence gives the RHS.
    # For the screened version the RHS is simply –α * indicator.
    # Indicator: each sample point contributes a Gaussian blob of its
    # inside/outside sign (inside = 1).  For the circle we use the sign
    # of (normal · (grid_pt – sample_pt)).
    tree = cKDTree(pts)
    sigma = 2.0 * max(h_x, h_y)         # spread

    chi0 = np.zeros(N)
    gpts = np.column_stack([X.ravel(), Y.ravel()])

    dist, idx_nn = tree.query(gpts, k=min(8, len(pts)))
    for k in range(dist.shape[1]):
        nn_pts  = pts[idx_nn[:, k]]
        nn_nrms = nrms[idx_nn[:, k]]
        diff    = gpts - nn_pts          # (N,2)
        proj    = (diff * nn_nrms).sum(axis=1)   # sign: inside < 0
        w       = np.exp(-dist[:, k]**2 / (2 * sigma**2))
        chi0   += w * (proj < 0).astype(float)
    # normalise to [0,1]
    chi0 /= (chi0.max() + 1e-12)

    # ── Build sparse FD Laplacian ──────────────────────────────────────────
    A = lil_matrix((N, N))
    b = np.zeros(N)

    for i in range(grid_res):
        for j in range(grid_res):
            n_ij = idx(i, j)
            on_border = (i == 0 or i == grid_res-1 or
                         j == 0 or j == grid_res-1)

            if on_border:
                if bc_type == "Dirichlet":
                    A[n_ij, n_ij] = 1.0
                    b[n_ij] = 0.0
                else:  # Neumann via ghost: duplicate interior neighbour
                    A[n_ij, n_ij] = -(2/h_x**2 + 2/h_y**2) - alpha
                    # neighbour clamping
                    ip = min(i+1, grid_res-1); im = max(i-1, 0)
                    jp = min(j+1, grid_res-1); jm = max(j-1, 0)
                    A[n_ij, idx(ip, j)] += 1/h_y**2
                    A[n_ij, idx(im, j)] += 1/h_y**2
                    A[n_ij, idx(i, jp)] += 1/h_x**2
                    A[n_ij, idx(i, jm)] += 1/h_x**2
                    b[n_ij] = -alpha * chi0[n_ij]
            else:
                A[n_ij, n_ij]          = -(2/h_x**2 + 2/h_y**2) - alpha
                A[n_ij, idx(i+1, j)]  =  1/h_y**2
                A[n_ij, idx(i-1, j)]  =  1/h_y**2
                A[n_ij, idx(i, j+1)]  =  1/h_x**2
                A[n_ij, idx(i, j-1)]  =  1/h_x**2
                b[n_ij] = -alpha * chi0[n_ij]

    chi_vec = spsolve(A.tocsr(), b)
    Chi = chi_vec.reshape(grid_res, grid_res)

    # normalise for display
    lo, hi = Chi.min(), Chi.max()
    if hi > lo:
        Chi_norm = (Chi - lo) / (hi - lo)
    else:
        Chi_norm = Chi

    return X, Y, Chi_norm, xs, ys

# ─── Plotting helpers ─────────────────────────────────────────────────────────
DARK_BG   = "#0a0a0f"
GRID_COL  = "#1a1a28"
PT_COL    = "#f97316"
NRM_COL   = "#38bdf8"
CURVE_COL = "#f8fafc"
CMAP      = "plasma"

def fig_preview(pts, nrms, curve):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="#3a3a5a")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5)

    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=PT_COL, zorder=3, alpha=0.85)
    scale = 0.12 if curve == "Circle" else 0.25
    ax.quiver(pts[:, 0], pts[:, 1],
              nrms[:, 0] * scale, nrms[:, 1] * scale,
              color=NRM_COL, scale=1, scale_units='xy',
              width=0.003, headwidth=4, headlength=5, alpha=0.8, zorder=4)
    ax.set_aspect('equal')
    ax.set_title("Input Points + Normals", color="#9090b0",
                 fontsize=9, fontfamily="monospace", pad=8)
    fig.tight_layout()
    return fig

def fig_reconstruction(pts, nrms, X, Y, Chi):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="#3a3a5a")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, zorder=0)

    ax.contourf(X, Y, Chi, levels=60, cmap=CMAP, alpha=0.55, zorder=1)
    ax.contour(X, Y, Chi, levels=[0.5], colors=[CURVE_COL],
               linewidths=1.8, zorder=3)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=PT_COL,
               zorder=4, alpha=0.6, label="samples")
    ax.set_aspect('equal')
    ax.set_title("Reconstructed Boundary  (χ = 0.5)", color="#9090b0",
                 fontsize=9, fontfamily="monospace", pad=8)
    fig.tight_layout()
    return fig

def fig_chi_field(X, Y, Chi):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="#3a3a5a")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

    cf = ax.contourf(X, Y, Chi, levels=80, cmap=CMAP)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#6b6b8a", labelsize=7)
    cbar.ax.yaxis.label.set_color("#6b6b8a")
    cbar.outline.set_edgecolor(GRID_COL)
    ax.contour(X, Y, Chi, levels=12, colors='white',
               linewidths=0.35, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title("χ  field", color="#9090b0",
                 fontsize=9, fontfamily="monospace", pad=8)
    fig.tight_layout()
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Screened Poisson 2D")
    st.markdown('<div class="section-title">01 — Input Curve</div>',
                unsafe_allow_html=True)

    curve = st.selectbox("Curve type", ["Circle", "Sine wave"],
                         disabled=st.session_state.locked)
    n_pts = st.slider("Point density", 20, 300, 80, 10,
                      disabled=st.session_state.locked)
    noise = st.slider("Noise σ", 0.0, 0.4, 0.02, 0.005,
                      disabled=st.session_state.locked)

    st.markdown('<div class="section-title">02 — Solver</div>',
                unsafe_allow_html=True)

    alpha = st.slider("α  (screening weight)", 0.01, 20.0, 2.0, 0.01,
                      disabled=not st.session_state.locked)
    bc    = st.selectbox("Boundary condition",
                         ["Dirichlet", "Neumann"],
                         disabled=not st.session_state.locked)

    st.markdown('<div class="section-title">03 — Actions</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    save_btn  = col_a.button("💾 Save & Solve",
                             use_container_width=True,
                             disabled=st.session_state.locked)
    clear_btn = col_b.button("✕  Clear",
                             use_container_width=True,
                             disabled=not st.session_state.locked)

    # status
    if st.session_state.locked:
        st.markdown('<p class="status-solved">● solved</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-idle">○ idle — tweak & save</p>',
                    unsafe_allow_html=True)

# ─── Live preview generation (always fresh while unlocked) ───────────────────
if not st.session_state.locked:
    if curve == "Circle":
        pts, nrms = generate_circle(n_pts, noise)
    else:
        pts, nrms = generate_sine(n_pts, noise)
else:
    pts   = st.session_state.points
    nrms  = st.session_state.normals

# ─── Button actions ───────────────────────────────────────────────────────────
if save_btn and not st.session_state.locked:
    st.session_state.points  = pts
    st.session_state.normals = nrms
    st.session_state.locked  = True
    st.rerun()

if clear_btn and st.session_state.locked:
    st.session_state.points  = None
    st.session_state.normals = None
    st.session_state.locked  = False
    st.session_state.chi     = None
    st.rerun()

# ─── Main layout ─────────────────────────────────────────────────────────────
st.markdown("# Screened Poisson  Surface Reconstruction — 2D")
st.markdown(
    '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;'
    'color:#4a4a6a;margin-top:-12px;">'
    'Kazhdan & Hoppe 2013 · finite-difference 2-D demo</p>',
    unsafe_allow_html=True,
)

if not st.session_state.locked:
    # ── Preview only ──────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="plot-label">live preview</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_preview(pts, nrms, curve), use_container_width=True)
    with col2:
        st.markdown('<div class="plot-label">awaiting solve…</div>',
                    unsafe_allow_html=True)
        fig_ph, ax_ph = plt.subplots(figsize=(5, 5))
        fig_ph.patch.set_facecolor(DARK_BG)
        ax_ph.set_facecolor(DARK_BG)
        for sp in ax_ph.spines.values():
            sp.set_edgecolor(GRID_COL)
        ax_ph.text(0.5, 0.5, "Press  Save & Solve",
                   transform=ax_ph.transAxes,
                   ha='center', va='center',
                   color="#3a3a5a", fontsize=12,
                   fontfamily='monospace')
        ax_ph.set_xticks([]); ax_ph.set_yticks([])
        fig_ph.tight_layout()
        st.pyplot(fig_ph, use_container_width=True)

else:
    # ── Solve & display ───────────────────────────────────────────────────
    with st.spinner("Running screened Poisson solver…"):
        X, Y, Chi, xs, ys = solve_screened_poisson(
            pts, nrms, alpha=alpha, bc_type=bc, grid_res=90
        )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown('<div class="plot-label">input samples</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_preview(pts, nrms, curve), use_container_width=True)

    with col2:
        st.markdown('<div class="plot-label">reconstructed boundary</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_reconstruction(pts, nrms, X, Y, Chi),
                  use_container_width=True)

    with col3:
        st.markdown('<div class="plot-label">χ  scalar field</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_chi_field(X, Y, Chi), use_container_width=True)

    # ── Info strip ────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-box">samples<br>'
        f'<span style="color:#f97316;font-size:1.1rem">{len(pts)}</span></div>',
        unsafe_allow_html=True)
    c2.markdown(
        f'<div class="metric-box">α  screening<br>'
        f'<span style="color:#60a5fa;font-size:1.1rem">{alpha:.2f}</span></div>',
        unsafe_allow_html=True)
    c3.markdown(
        f'<div class="metric-box">boundary cond.<br>'
        f'<span style="color:#a78bfa;font-size:1.1rem">{bc}</span></div>',
        unsafe_allow_html=True)
    c4.markdown(
        f'<div class="metric-box">grid<br>'
        f'<span style="color:#4ade80;font-size:1.1rem">90 × 90</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
        'color:#3a3a5a;margin-top:4px;">'
        'Δχ − α·χ = −α·χ₀ &nbsp;|&nbsp; '
        'χ₀ built by nearest-neighbour Gaussian splatting of oriented samples &nbsp;|&nbsp; '
        'boundary extracted at χ = 0.5'
        '</p>',
        unsafe_allow_html=True,
    )
