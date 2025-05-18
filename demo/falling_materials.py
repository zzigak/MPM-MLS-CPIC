import taichi as ti
import os
from taichi.tools.video import VideoManager
import math

ti.init(arch=ti.gpu)

# --- Recording Setup ---
record_output = True
output_directory = "falling_materials_simulation"
video_filename = "falling_materials"
# --- End Recording Setup ---

# Material IDs
JELLO = 0
METAL = 1
SNOW = 2
LIQUID = 3
HONEY = 4
GROUND = 5
N_MATERIALS = 6

# Material properties
material_rho = ti.field(dtype=float, shape=N_MATERIALS)
material_rho[JELLO] = 2.0       # Jello density
material_rho[METAL] = 10.0      # Metal density
material_rho[SNOW] = 3.0        # Snow density
material_rho[LIQUID] = 1.0      # Liquid density
material_rho[HONEY] = 1.4       # Honey density
material_rho[GROUND] = 20.0     # Ground density

quality = 1
n_particles, n_grid = 20000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol = (dx * 0.5)**2

material_mass = ti.field(dtype=float, shape=N_MATERIALS)
for i in range(N_MATERIALS):
    material_mass[i] = p_vol * material_rho[i]

# Material parameters
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# Honey properties
mu_honey = 600.0    # High viscosity for honey
lambda_honey = 500.0
honey_yield_strength = 0.007

# Fields for simulation
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())

@ti.kernel
def substep():
    # Clear grid
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    # P2G (Particle to Grid)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        # Deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]

        # Material-specific behaviors
        mat_type = material[p]
        stress = ti.Matrix.zero(float, 2, 2)  # Initialize stress

        if mat_type == JELLO:
            # Jello: Soft elastic with some plasticity
            h = 0.3
            mu, la = mu_0 * h, lambda_0 * h

            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(2)):
                J *= sig[d, d]

            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                     ti.Matrix.identity(float, 2) * la * J * (J - 1)

        elif mat_type == METAL:
            # Metal: Stiff elastic
            h = 5.0
            mu, la = mu_0 * h, lambda_0 * h

            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(2)):
                J *= sig[d, d]

            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                     ti.Matrix.identity(float, 2) * la * J * (J - 1)

        elif mat_type == SNOW:
            # Snow: Plasticity with hardening
            h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
            mu, la = mu_0 * h, lambda_0 * h

            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plastic constraint
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig

            # Reconstruct elastic deformation gradient
            F[p] = U @ sig @ V.transpose()

            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                     ti.Matrix.identity(float, 2) * la * J * (J - 1)

        elif mat_type == LIQUID:
            # Liquid: Zero shear stress
            mu = 0.0
            la = lambda_0

            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(2)):
                J *= sig[d, d]

            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)

            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                     ti.Matrix.identity(float, 2) * la * J * (J - 1)

        elif mat_type == HONEY:
            # Honey: High viscosity with yield strength
            h = 1.0  # No hardening for honey
            mu, la = mu_honey * h, lambda_honey * h

            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(2)):
                J *= sig[d, d]

            # Stress calculation for viscous fluid
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                     ti.Matrix.identity(float, 2) * la * J * (J - 1)

            # Apply yield criterion (honey flows when stress exceeds yield strength)
            if stress.norm() > honey_yield_strength:
                F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)

        # Compute momentum contribution
        mass = material_mass[material[p]]
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + mass * C[p]

        # Scatter to grid (P2G)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * mass

    # Grid operations
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            # Momentum to velocity
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]

            # Apply gravity
            grid_v[i, j] += dt * gravity[None]

            # Apply material-specific grid operations
            # For honey - add additional damping

            # Boundary conditions - containing box
            if i < 3 and grid_v[i, j][0] < 0: grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

    # G2P (Grid to Particle)
    for p in x:
        if material[p] == GROUND:
            continue  # Skip ground particles - they don't move

        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)

        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p], C[p] = new_v, new_C

        # Apply additional damping for honey
        if material[p] == HONEY:
            v[p] *= ti.exp(-dt * 5.0)

        # Update position
        x[p] += dt * v[p]

@ti.kernel
def reset():
    # Set gravity
    gravity[None] = [0, -9.8]

    # Initialize material cubes
    cube_size = 0.15
    bar_height = 0.08  # Height of the honey bar
    bar_width = 0.8    # Width of the honey bar
    spacing = 0.05     # Space between cubes

    # Calculate horizontal positions for the cubes
    total_width = 2 * cube_size + spacing
    start_x = (1.0 - total_width) / 2  # Center the group of cubes

    # Honey bar at the top (wide horizontal bar)
    honey_particles = int(n_particles * 0.25)  # Allocate 25% particles for the bar
    for i in range(honey_particles):
        material[i] = HONEY
        x[i] = [
            0.1 + ti.random() * bar_width,  # Span most of the screen width
            0.9 - ti.random() * bar_height  # Thin bar at top
        ]
        v[i] = [0, -0.5]  # Slightly slower initial velocity
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)

    # Adjust remaining particles count for other materials
    remaining_particles = n_particles - honey_particles
    cube_particles = remaining_particles // 4  # Divide among 4 other materials

    # First row (jello and metal)
    row1_y = 0.6
    # Jello (left)
    cube_x = start_x
    for i in range(honey_particles, honey_particles + cube_particles):
        material[i] = JELLO
        x[i] = [
            cube_x + ti.random() * cube_size,
            row1_y + ti.random() * cube_size
        ]
        v[i] = [0, -1]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)

    # Metal (right)
    cube_x += cube_size + spacing
    for i in range(honey_particles + cube_particles,
                   honey_particles + 2 * cube_particles):
        material[i] = METAL
        x[i] = [
            cube_x + ti.random() * cube_size,
            row1_y + ti.random() * cube_size
        ]
        v[i] = [0, -1]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)

    # Second row (snow and water)
    row2_y = 0.35
    # Snow (left)
    cube_x = start_x
    for i in range(honey_particles + 2 * cube_particles,
                   honey_particles + 3 * cube_particles):
        material[i] = SNOW
        x[i] = [
            cube_x + ti.random() * cube_size,
            row2_y + ti.random() * cube_size
        ]
        v[i] = [0, -1]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)

    # Water (right)
    cube_x += cube_size + spacing
    for i in range(honey_particles + 3 * cube_particles, n_particles):
        material[i] = LIQUID
        x[i] = [
            cube_x + ti.random() * cube_size,
            row2_y + ti.random() * cube_size
        ]
        v[i] = [0, -1]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)

# GUI setup
gui = ti.GUI("Falling Materials", res=512, background_color=0x112233)
palette_colors = [
    0xFFD700,  # JELLO - Gold
    0x8A8A8A,  # METAL - Gray
    0xFFFFFF,  # SNOW - White
    0x1E90FF,  # LIQUID - Blue
    0xFFA500,  # HONEY - Orange
    0x704214   # GROUND - Brown
]

# Video recording setup
video_manager = None
if record_output:
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    video_manager = VideoManager(output_dir=output_directory, framerate=30, automatic_build=False)
    print(f"Recording enabled. Video will be saved in '{output_directory}/{video_filename}.mp4'")

# Initialize simulation
reset()

# Main simulation loop
total_frames_to_run = 900 if record_output else 20000

for frame in range(total_frames_to_run):
    if gui.running:
        # Handle input
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "r":
                reset()
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break

        # Run simulation steps
        for s in range(int(2e-3 // dt)):
            substep()

        # Render particles
        gui.circles(x.to_numpy(), radius=1.5, palette=palette_colors, palette_indices=material)

        # Record video frame if enabled
        if record_output and video_manager:
            video_manager.write_frame(gui.get_image())

        gui.show()
    else:
        break

# Finalize video recording
if record_output and video_manager:
    print("Making video...")
    video_manager.make_video(gif=False, mp4=True, video_filename=video_filename)
    print(f"Video saved as {output_directory}/{video_filename}.mp4")
else:
    print("Simulation finished.")
