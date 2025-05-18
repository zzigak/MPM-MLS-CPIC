
import taichi as ti
import os
from taichi.tools.video import VideoManager

ti.init(arch=ti.gpu)

# --- Recording Setup ---
record_output = True
output_directory = "simulation_video_output_jello_focus"
video_filename = "jello_metal_blocks_close_start"
# --- End Recording Setup ---


# Material IDs
LIQUID = 0
JELLY = 1
METAL_RED = 2
METAL_GREEN = 3
METAL_BLUE = 4

N_MATERIALS = 5

material_rho = ti.field(dtype=float, shape=N_MATERIALS)
material_rho[LIQUID] = 2.0
material_rho[JELLY] = 2.0       # Jello density
material_rho[METAL_RED] = 4.0   # Red block density
material_rho[METAL_GREEN] = 15.0 # Green block density
material_rho[METAL_BLUE] = 30.0 # Blue block density

quality = 1
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol = (dx * 0.5)**2

material_mass = ti.field(dtype=float, shape=N_MATERIALS)
for i in range(N_MATERIALS):
    material_mass[i] = p_vol * material_rho[i]

E, nu = 5e3, 0.2
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

x = ti.Vector.field(2, dtype=float, shape=n_particles)
v = ti.Vector.field(2, dtype=float, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
material = ti.field(dtype=int, shape=n_particles)
Jp = ti.field(dtype=float, shape=n_particles)

grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        mass = material_mass[material[p]]
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == JELLY: h = 0.3
        elif material[p] == METAL_RED or material[p] == METAL_GREEN or material[p] == METAL_BLUE: h = 5.0
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == LIQUID: mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig_d = sig[d, d]
            Jp[p] *= sig[d, d] / new_sig_d
            sig[d, d] = new_sig_d
            J *= new_sig_d
        if material[p] == LIQUID: F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0: grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
    for p in x:
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
        x[p] += dt * v[p]

@ti.kernel
def init_sphere_throw():
    # Particle distribution: 70% Jello, 30% Sphere
    n_jello_particles = int(n_particles * 0.70)
    n_sphere_particles = n_particles - n_jello_particles

    # Jello block parameters (at the bottom)
    jello_y_min = 0.025
    jello_y_height = 0.20 # Slightly shorter jello to give sphere more room to impact
    jello_x_min = 0.1
    jello_x_width = 0.8

    # Sphere parameters
    sphere_center_x = 0.5
    sphere_center_y = 0.85 # Start near the top
    sphere_radius = 0.08  # Radius of the sphere in simulation units
    initial_sphere_velocity = ti.Vector([0.0, -8.0]) # Initial downward velocity

    particle_idx = 0
    # Initialize Jello Particles
    for i in range(n_jello_particles):
        x[particle_idx] = [
            ti.random() * jello_x_width + jello_x_min,
            ti.random() * jello_y_height + jello_y_min,
        ]
        material[particle_idx] = JELLY
        v[particle_idx] = [0.0, 0.0]
        F[particle_idx] = ti.Matrix([[1, 0], [0, 1]])
        Jp[particle_idx] = 1.0
        C[particle_idx] = ti.Matrix.zero(float, 2, 2)
        particle_idx += 1

    # Initialize Sphere Particles
    # Attempt to distribute particles somewhat evenly within the sphere
    for i in range(n_sphere_particles):
        # Generate random angle and radius for 2D circular distribution
        # For a more uniform distribution, use sqrt(random) for radius
        r = sphere_radius * ti.sqrt(ti.random())
        theta = 2 * math.pi * ti.random()
        offset = ti.Vector([r * ti.cos(theta), r * ti.sin(theta)])
        x[particle_idx] = ti.Vector([sphere_center_x, sphere_center_y]) + offset

        material[particle_idx] = METAL_BLUE # Heavy metal
        v[particle_idx] = initial_sphere_velocity
        F[particle_idx] = ti.Matrix([[1, 0], [0, 1]])
        Jp[particle_idx] = 1.0
        C[particle_idx] = ti.Matrix.zero(float, 2, 2)
        particle_idx += 1


@ti.kernel
def reset():
    # MODIFIED: Increased jello particle proportion
    n_jello_particles = int(n_particles * 0.75) # More particles for jello
    n_metal_particles_remaining = n_particles - n_jello_particles
    n_metal_particles_each = n_metal_particles_remaining // 3

    # Ensure at least one particle per metal block if remaining particles are very few
    if n_metal_particles_each == 0 and n_metal_particles_remaining > 0:
        n_metal_particles_each = 1


    jello_end_idx = n_jello_particles
    metal_red_end_idx = jello_end_idx + n_metal_particles_each
    metal_green_end_idx = metal_red_end_idx + n_metal_particles_each
    # Blue block gets from metal_green_end_idx to n_particles

    jello_y_min = 0.025
    jello_y_height = 0.25
    jello_x_min = 0.1
    jello_x_width = 0.8

    block_size = 0.05
    jello_top_y = jello_y_min + jello_y_height
    # MODIFIED: Blocks start just above the jello
    block_initial_y_center = jello_top_y + block_size * 0.5 + 0.02 # Small gap above jello

    for i in range(n_particles):
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)
        v[i] = [0.0, 0.0]
        if i < jello_end_idx:
            material[i] = JELLY
            x[i] = [ti.random() * jello_x_width + jello_x_min, ti.random() * jello_y_height + jello_y_min]
        elif i < metal_red_end_idx:
            material[i] = METAL_RED
            center = ti.Vector([0.3, block_initial_y_center])
            x[i] = center + ti.Vector([(ti.random() - 0.5) * block_size, (ti.random() - 0.5) * block_size])
        elif i < metal_green_end_idx:
            material[i] = METAL_GREEN
            center = ti.Vector([0.5, block_initial_y_center])
            x[i] = center + ti.Vector([(ti.random() - 0.5) * block_size, (ti.random() - 0.5) * block_size])
        else: # Blue block (check particle index range)
             if i < n_particles: # Ensure we don't go out of bounds if n_metal_particles_each was rounded
                material[i] = METAL_BLUE
                center = ti.Vector([0.7, block_initial_y_center])
                x[i] = center + ti.Vector([(ti.random() - 0.5) * block_size, (ti.random() - 0.5) * block_size])


print("[Hint] Use WSAD/arrow keys to control gravity. Use L/R mouse to attract/repel. Press R to reset.")
gui = ti.GUI("Taichi MLS-MPM-128: Jello and Metal Blocks", res=512, background_color=0xDDDDDD)

palette_colors = [
    0x068587,
    0xC71585,
    0xFF0000,
    0x008200,
    0x0000FF
]

video_manager = None
if record_output:
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    video_manager = VideoManager(output_dir=output_directory, framerate=30, automatic_build=False)
    print(f"Recording enabled. Video will be saved in '{output_directory}/{video_filename}.mp4'")

reset()
gravity[None] = [0, -1]

total_frames_to_run = 300 if record_output else 20000 # e.g., 10 seconds at 30fps for recording

for frame in range(total_frames_to_run):
    if gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "r":
                reset()
                gravity[None] = [0, -1]
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break

        current_gravity_x = gravity[None][0]
        current_gravity_y = gravity[None][1]
        if gui.is_pressed(ti.GUI.LEFT, "a"): current_gravity_x = -1.0
        elif gui.is_pressed(ti.GUI.RIGHT, "d"): current_gravity_x = 1.0
        else: current_gravity_x = 0.0
        if gui.is_pressed(ti.GUI.UP, "w"): current_gravity_y = 1.0
        elif gui.is_pressed(ti.GUI.DOWN, "s"): current_gravity_y = -1.0
        else: current_gravity_y = -1.0
        gravity[None] = [current_gravity_x, current_gravity_y]

        mouse = gui.get_cursor_pos()
        attractor_pos[None] = [mouse[0], mouse[1]]
        attractor_strength[None] = 0.0
        if gui.is_pressed(ti.GUI.LMB): attractor_strength[None] = 1.0
        if gui.is_pressed(ti.GUI.RMB): attractor_strength[None] = -1.0

        for s in range(int(2e-3 // dt)):
            substep()

        gui.circles(x.to_numpy(), radius=1.5, palette=palette_colors, palette_indices=material)
        # gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15) # Optional: mouse cursor

        if record_output and video_manager:
            img = gui.get_image()
            video_manager.write_frame(img)

        gui.show()
    else:
        break

if record_output and video_manager:
    print("Making video...")
    video_manager.make_video(gif=False, mp4=True, video_filename=video_filename)
    print(f"Video saved as {output_directory}/{video_filename}.mp4")
else:
    print("Simulation finished.")
# import taichi as ti
# import os
# from taichi.tools.video import VideoManager
# import math # For pi, cos, sin

# ti.init(arch=ti.gpu)

# # --- Recording Setup ---
# record_output = True
# # MODIFIED: New output directory and filename for the new scene
# output_directory = "simulation_video_output_sphere_impact"
# video_filename = "jello_metal_sphere_impact"
# # --- End Recording Setup ---


# # Material IDs
# LIQUID = 0  # Kept for structure, but not used in this scene
# JELLY = 1
# METAL_SPHERE = 2 # New material for the sphere

# # MODIFIED: Number of materials
# N_MATERIALS = 3

# material_rho = ti.field(dtype=float, shape=N_MATERIALS)
# material_rho[LIQUID] = 2.0       # Not used, but defined
# material_rho[JELLY] = 2.0        # Jello density
# material_rho[METAL_SPHERE] = 5.0 # Metal sphere density

# quality = 1
# n_particles, n_grid = 9000 * quality**2, 128 * quality
# dx, inv_dx = 1 / n_grid, float(n_grid)
# dt = 1e-4 / quality
# p_vol = (dx * 0.5)**2

# material_mass = ti.field(dtype=float, shape=N_MATERIALS)
# for i in range(N_MATERIALS):
#     material_mass[i] = p_vol * material_rho[i]

# E, nu = 5e3, 0.2 # Base elastic properties
# mu_0 = E / (2 * (1 + nu))
# lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# x = ti.Vector.field(2, dtype=float, shape=n_particles)
# v = ti.Vector.field(2, dtype=float, shape=n_particles)
# C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
# F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
# material = ti.field(dtype=int, shape=n_particles)
# Jp = ti.field(dtype=float, shape=n_particles) # Plastic volume change tracker

# grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
# grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
# gravity = ti.Vector.field(2, dtype=float, shape=())
# attractor_strength = ti.field(dtype=float, shape=())
# attractor_pos = ti.Vector.field(2, dtype=float, shape=())

# @ti.kernel
# def substep():
#     for i, j in grid_m:
#         grid_v[i, j] = [0, 0]
#         grid_m[i, j] = 0
#     for p in x:
#         mass = material_mass[material[p]]
#         base = (x[p] * inv_dx - 0.5).cast(int)
#         fx = x[p] * inv_dx - base.cast(float)
#         w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

#         # Deformation gradient update
#         F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]

#         # Hardening factor (h) based on material type
#         # Jp is initialized to 1.0. The line `Jp[p] *= sig[d,d]/new_sig_d` with new_sig_d=sig[d,d] means Jp stays 1.0.
#         # So, default h = ti.exp(10 * (1.0 - Jp[p])) = ti.exp(0) = 1.0.
#         h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p])))) # Default h based on Jp (effectively 1.0 here)

#         if material[p] == JELLY:
#             h = 0.3  # Makes jello softer
#         elif material[p] == METAL_SPHERE:
#             h = 5.0  # Makes metal sphere stiffer (E_eff = 5.0 * E)

#         mu, la = mu_0 * h, lambda_0 * h
#         if material[p] == LIQUID: # Not used in this scene
#             mu = 0.0

#         U, sig, V = ti.svd(F[p])
#         J = 1.0 # Determinant of F

#         # This loop as written effectively keeps Jp[p] at its initial value (1.0)
#         # and calculates J = sig[0,0] * sig[1,1]
#         for d in ti.static(range(2)):
#             new_sig_d = sig[d, d]
#             # Jp[p] *= sig[d, d] / new_sig_d # This is Jp[p] *= 1.0 if new_sig_d is sig[d,d]
#             # sig[d, d] = new_sig_d # This does nothing new if new_sig_d is sig[d,d]
#             J *= new_sig_d # Accumulate J = product of singular values

#         if material[p] == LIQUID:
#             F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)

#         # Stress calculation (First Piola-Kirchhoff stress based on corotated elasticity)
#         stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
#                  ti.Matrix.identity(float, 2) * la * J * (J - 1)

#         stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
#         affine = stress + mass * C[p]

#         # P2G: Scatter particle data to grid
#         for i_offset, j_offset in ti.static(ti.ndrange(3, 3)):
#             offset = ti.Vector([i_offset, j_offset])
#             dpos = (offset.cast(float) - fx) * dx
#             weight = w[i_offset][0] * w[j_offset][1]
#             grid_v[base + offset] += weight * (mass * v[p] + affine @ dpos)
#             grid_m[base + offset] += weight * mass

#     # Grid operations
#     for i, j in grid_m:
#         if grid_m[i, j] > 0:
#             grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j] # Normalize velocity
#             grid_v[i, j] += dt * gravity[None] * 30 # Apply gravity

#             # Attractor force
#             dist = attractor_pos[None] - dx * ti.Vector([i, j])
#             grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100

#             # Boundary conditions
#             if i < 3 and grid_v[i, j][0] < 0: grid_v[i, j][0] = 0
#             if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
#             if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
#             if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0 # Ground and ceiling

#     # G2P: Gather grid data back to particles
#     for p in x:
#         base = (x[p] * inv_dx - 0.5).cast(int)
#         fx = x[p] * inv_dx - base.cast(float)
#         w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
#         new_v = ti.Vector.zero(float, 2)
#         new_C = ti.Matrix.zero(float, 2, 2)
#         for i_offset, j_offset in ti.static(ti.ndrange(3, 3)):
#             dpos = ti.Vector([i_offset, j_offset]).cast(float) - fx
#             g_v = grid_v[base + ti.Vector([i_offset, j_offset])]
#             weight = w[i_offset][0] * w[j_offset][1]
#             new_v += weight * g_v
#             new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
#         v[p], C[p] = new_v, new_C
#         x[p] += dt * v[p] # Advect particles

# @ti.kernel
# def reset():
#     # Jello parameters
#     jello_y_min = 0.025
#     jello_y_height = 0.25
#     jello_x_min = 0.1
#     jello_x_width = 0.8

#     # Metal sphere parameters
#     sphere_center = ti.Vector([0.5, 0.7]) # x, y position
#     sphere_radius = 0.05
#     initial_sphere_vy = -2.5 # Initial downward velocity for the sphere

#     # Particle distribution
#     n_jello_particles = int(n_particles * 0.85) # ~70% particles for jello

#     for i in range(n_particles):
#         # Common initialization for all particles
#         F[i] = ti.Matrix([[1, 0], [0, 1]])
#         Jp[i] = 1.0
#         C[i] = ti.Matrix.zero(float, 2, 2)

#         if i < n_jello_particles:
#             material[i] = JELLY
#             x[i] = [
#                 jello_x_min + ti.random(dtype=float) * jello_x_width,
#                 jello_y_min + ti.random(dtype=float) * jello_y_height
#             ]
#             v[i] = [0.0, 0.0] # Jello starts at rest
#         else: # Sphere particles
#             material[i] = METAL_SPHERE
#             # Generate random point within the sphere (polar coordinates for better distribution)
#             rand_r = sphere_radius * ti.sqrt(ti.random(dtype=float)) # sqrt for uniform area sampling
#             rand_angle = ti.random(dtype=float) * 2.0 * math.pi

#             offset_x = rand_r * ti.cos(rand_angle)
#             offset_y = rand_r * ti.sin(rand_angle)

#             x[i] = sphere_center + ti.Vector([offset_x, offset_y])
#             v[i] = [0.0, initial_sphere_vy] # Sphere has initial downward velocity


# # MODIFIED: GUI Title and Palette
# gui = ti.GUI("Taichi MLS-MPM-128: Metal Sphere on Jello", res=512, background_color=0xDDDDDD)

# palette_colors = [
#     0x068587,  # LIQUID (index 0, not actively used)
#     0xC71585,  # JELLY (index 1, e.g., magenta/pink)
#     0x707070   # METAL_SPHERE (index 2, e.g., grey)
# ]

# video_manager = None
# if record_output:
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#     video_manager = VideoManager(output_dir=output_directory, framerate=30, automatic_build=False)
#     print(f"Recording enabled. Video will be saved in '{output_directory}/{video_filename}.mp4'")

# reset()
# gravity[None] = [0, -1] # Standard downward gravity, strength scaled in substep

# total_frames_to_run = 300 if record_output else 20000 # Approx 10 seconds of video if recording

# for frame in range(total_frames_to_run):
#     if gui.running:
#         # Handle user input for reset and exit
#         if gui.get_event(ti.GUI.PRESS):
#             if gui.event.key == "r":
#                 reset()
#                 gravity[None] = [0, -1] # Reset gravity to default on scene reset
#             elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
#                 break

#         # Handle gravity controls
#         current_gravity_x = gravity[None][0]
#         current_gravity_y = gravity[None][1]

#         if gui.is_pressed(ti.GUI.LEFT, "a"): current_gravity_x = -1.0
#         elif gui.is_pressed(ti.GUI.RIGHT, "d"): current_gravity_x = 1.0
#         else: current_gravity_x = 0.0

#         if gui.is_pressed(ti.GUI.UP, "w"): current_gravity_y = 1.0
#         elif gui.is_pressed(ti.GUI.DOWN, "s"): current_gravity_y = -1.0
#         else: current_gravity_y = -1.0 # Default to downward gravity if no up/down key

#         gravity[None] = [current_gravity_x, current_gravity_y]

#         # Handle mouse attractor
#         mouse = gui.get_cursor_pos()
#         attractor_pos[None] = [mouse[0], mouse[1]]
#         attractor_strength[None] = 0.0
#         if gui.is_pressed(ti.GUI.LMB): attractor_strength[None] = 1.0
#         if gui.is_pressed(ti.GUI.RMB): attractor_strength[None] = -1.0

#         # Run simulation substeps
#         for s in range(int(2e-3 // dt)): # Number of substeps for stability
#             substep()

#         # Render particles
#         gui.circles(x.to_numpy(), radius=1.5, palette=palette_colors, palette_indices=material)
#         # gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15) # Optional: visualize mouse attractor position

#         # Write frame to video if recording
#         if record_output and video_manager:
#             img = gui.get_image()
#             video_manager.write_frame(img)

#         gui.show()
#     else: # gui.running is False
#         break

# # Finalize video if recording
# if record_output and video_manager:
#     print("Making video...")
#     video_manager.make_video(gif=False, mp4=True, video_filename=video_filename)
#     print(f"Video saved as {output_directory}/{video_filename}.mp4")
# else:
#     print("Simulation finished.")
