import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

def solve_screened_poisson(grid_res, points, normals, alpha, bc_type='dirichlet'):
    """
    Solves (Laplacian - alpha*I) Chi = Div(V)
    """
    N = grid_res
    h = 2.0 / (N - 1)  # Grid spacing assuming domain [-1, 1]
    size = N * N
    
    # 1. Create the Vector Field V on the grid
    vx = np.zeros((N, N))
    vy = np.zeros((N, N))
    
    # Map points to grid indices
    px = ((points[:, 0] + 1) / 2 * (N - 1)).astype(int)
    py = ((points[:, 1] + 1) / 2 * (N - 1)).astype(int)
    
    # Simple splatting of normals into the grid
    for i in range(len(points)):
        if 0 <= px[i] < N and 0 <= py[i] < N:
            vx[py[i], px[i]] = normals[i, 0]
            vy[py[i], px[i]] = normals[i, 1]

    # 2. Compute Divergence of V: div(V) = dVx/dx + dVy/dy
    dvx = np.gradient(vx, axis=1) / h
    dvy = np.gradient(vy, axis=0) / h
    div_v = dvx + dvy

    # 3. Construct Finite Difference Laplacian Matrix (L)
    # Using 5-point stencil
    main_diag = -4 * np.ones(size)
    side_diag = np.ones(size - 1)
    # Correct for row boundaries in the 1D flattened array
    side_diag[np.arange(1, size) % N == 0] = 0 
    up_down_diag = np.ones(size - N)
    
    from scipy.sparse import diags
    L = diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
              [0, -1, 1, -N, N], shape=(size, size), format='csr') / (h**2)

    # 4. Add Screening Term: (L - alpha * I)
    # The paper uses alpha to weight the interpolation constraint
    A = L - alpha * eye(size, format='csr')

    # 5. Setup RHS (Target values)
    b = div_v.flatten()
    
    # Add point value constraints to RHS if alpha > 0
    # For a characteristic function, points are target value ~1.0
    for i in range(len(points)):
        idx = py[i] * N + px[i]
        if 0 <= idx < size:
            b[idx] -= alpha * 1.0 

    # 6. Apply Boundary Conditions
    if bc_type == 'dirichlet':
        # Force edges to zero
        mask = np.ones((N, N), dtype=bool)
        mask[1:-1, 1:-1] = False
        edge_indices = np.where(mask.flatten())[0]
        for idx in edge_indices:
            A.data[A.indptr[idx]:A.indptr[idx+1]] = 0
            A[idx, idx] = 1
            b[idx] = 0
            
    # Solve linear system
    chi = spsolve(A, b)
    return chi.reshape((N, N))