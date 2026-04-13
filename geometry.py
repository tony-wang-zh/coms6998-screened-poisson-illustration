import numpy as np

def get_circle(n_points=200):
    t = np.linspace(0, 2*np.pi, n_points)
    points = np.vstack([np.cos(t), np.sin(t)]).T
    # Normals for a circle point outward
    normals = np.vstack([np.cos(t), np.sin(t)]).T
    return points, normals

def get_sine_wave(n_points=200):
    x = np.linspace(-1, 1, n_points)
    y = 0.5 * np.sin(np.pi * x)
    points = np.vstack([x, y]).T
    # Approximate normals: perpendicular to tangent (-dy, dx)
    dx = np.gradient(x)
    dy = np.gradient(y)
    mag = np.sqrt(dx**2 + dy**2)
    normals = np.vstack([-dy/mag, dx/mag]).T
    return points, normals

def add_noise(points, normals, noise_level=0):
    noise = np.random.normal(0, noise_level, points.shape)
    return points + noise, normals