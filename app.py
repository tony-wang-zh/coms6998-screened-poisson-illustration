import streamlit as st
import matplotlib.pyplot as plt
from geometry import get_circle, get_sine_wave, add_noise
from solver import solve_screened_poisson

st.set_page_config(layout="wide")
st.title("Interactive Screened Poisson Reconstruction (2D)")

# --- 1. SESSION STATE INITIALIZATION ---
# This ensures the points don't disappear when you move a slider
if 'points' not in st.session_state:
    st.session_state.points = None
    st.session_state.normals = None

# --- 2. STEP 1: GENERATE DATA ---
st.sidebar.header("Step 1: Data Generation")
# shape_type = st.sidebar.selectbox("Base Shape", ["Circle", "Sine Wave"])
shape_type = "Circle"
density = st.sidebar.slider("Point Density", 50, 500, 250)
noise_lvl = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.0)
noise_lvl = noise_lvl / 100.0 # this can't handle much noise apparently 


p, n = get_sine_wave(density)
    
# Store in session state
points, normals = add_noise(p, n, noise_lvl)

col1, col2 = st.sidebar.columns(2)
if col1.button("Save"):
    if shape_type == "Circle":
        p, n = get_circle(density)
    else:
        p, n = get_sine_wave(density)
    
    # Store in session state
    st.session_state.points, st.session_state.normals = points, normals
    plt.close('all')

if col2.button("Clear Data"):
    st.session_state.points = None
    st.session_state.normals = None
    st.rerun()

    


# --- 3. STEP 2: RECONSTRUCTION PARAMETERS ---
st.sidebar.header("Step 2: Solver Params")
alpha = st.sidebar.slider("Screening Weight (Alpha)", 0.0, 10.0, 0.0)
res = st.sidebar.slider("resolution of grid", 0, 128, 32)
bc = st.sidebar.radio("Boundary Condition", ["dirichlet", "neumann"])


# --- 4. VISUALIZATION AND SOLVING ---
if st.session_state.points is not None:
    # Perform the solve using current slider values but stored points
    chi = solve_screened_poisson(
        res, 
        st.session_state.points, 
        st.session_state.normals, 
        alpha, 
        bc
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Point Cloud
    ax[0].scatter(st.session_state.points[:, 0], st.session_state.points[:, 1], 
                  s=10, c='red', label="Frozen Points")
    ax[0].quiver(st.session_state.points[:, 0], st.session_state.points[:, 1], 
                 st.session_state.normals[:, 0], st.session_state.normals[:, 1], alpha=0.2)
    # ax[1].contour(chi, levels=[0], extent=[-1, 1, -1, 1], colors='black', linewidths=2)
    ax[0].set_title(f"Input: {len(st.session_state.points)} Points")
    ax[0].set_xlim(-1.2, 1.2)
    ax[0].set_ylim(-1.2, 1.2)
    ax[0].set_aspect('equal')

    # Right: Reconstruction
    im = ax[1].imshow(chi, extent=[-1, 1, -1, 1], origin='lower', cmap='RdBu_r')
    # The "Surface" is the zero-crossing. Because of the screening term, 
    # we look for the 0.5 contour (midway between inside 1 and outside 0)
    ax[1].contour(chi, levels=[0], extent=[-1, 1, -1, 1], colors='black', linewidths=2)
    ax[1].set_title(f"Reconstruction (Alpha={alpha})")
    ax[1].set_xlim(-1.2, 1.2)
    ax[1].set_ylim(-1.2, 1.2)
    plt.colorbar(im, ax=ax[1])

    st.pyplot(fig)
    
    if st.button("Save Comparison Image"):
        plt.savefig(f"recon_a{alpha}_res{res}.png")
        st.success(f"Saved as recon_a{alpha}_res{res}.png")

else:
    st.info("Click 'Save' in the sidebar and then start reconstruction")