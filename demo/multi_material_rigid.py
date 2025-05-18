import taichi as ti
import numpy as np
import math
import os
from taichi.tools.video import VideoManager

# Use ti.f32 for consistency
ti.init(arch=ti.gpu, default_fp=ti.f32) # Try to run on GPU

dim = 2
grid_res = 128
num_particles = 10000 # Increased for more objects
dt = 1e-4
dx = 1 / grid_res
inv_dx = float(grid_res)

# Material IDs
LIQUID = 0
JELLO = 1
METAL = 2
SNOW = 3
HONEY = 4
N_MATERIALS = 5

# Material properties
material_rho = ti.field(dtype=ti.f32, shape=N_MATERIALS)
material_rho[LIQUID] = 1.0
material_rho[JELLO] = 1.2
material_rho[METAL] = 40.0
material_rho[SNOW] = 0.8
material_rho[HONEY] = 1.4

# Material-specific parameters
honey_mu = 600.0
honey_lambda = 500.0
honey_yield = 0.007

# Elastic parameters
E = 5e3 # youngs-modulus
nu = 0.2 # poisson's ratio
mu_0, lamb_0 = E / (2*(1+nu)), E*nu / ((1+nu) * (1-2*nu)) # Lame params

# Particle volume and mass
p_vol = (dx * 0.5)**dim
material_mass = ti.field(dtype=ti.f32, shape=N_MATERIALS)
for i in range(N_MATERIALS):
    material_mass[i] = p_vol * material_rho[i]

# define particle properties
x = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles) # Position
v = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles) # Velocity
C = ti.Matrix.field(dim,dim, dtype=ti.f32, shape=num_particles) # affine velocity matrix
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=num_particles) # deformation gradient
p_material = ti.field(dtype=ti.int32, shape=num_particles) # Material ID
Jp = ti.field(dtype=ti.f32, shape=num_particles) # Plasticity for snow

# grid properties
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(grid_res, ) * dim) # velocity
grid_m = ti.field(dtype=ti.f32, shape=(grid_res, ) * dim) # mass

# gravity vector
g_val = -9.8
g = ti.Vector([0.0, g_val])

substeps = 50

# Rigid fan properties
N_FANS = 2
rigid_center = ti.Vector.field(dim, dtype=ti.f32, shape=N_FANS)
rigid_radius = ti.field(dtype=ti.f32, shape=N_FANS)
omega = ti.field(dtype=ti.f32, shape=N_FANS) # Angular velocity
theta = ti.field(dtype=ti.f32, shape=N_FANS) # Current angle

@ti.kernel
def sub_step():
    # clear grid
    for I in ti.grouped(grid_m):
        grid_v[I].fill(0.0)
        grid_m[I] = 0.0

    # P2G
    for p in x:
        # update F = (I + dt * C) * F
        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]

        base_coord = (x[p] * inv_dx - 0.5).cast(int)
        fracXY = x[p] * inv_dx - base_coord.cast(float)
        w = [0.5 * (1.5 - fracXY) ** 2, 0.75 - (fracXY - 1.0) **2, 0.5 * (fracXY - 0.5) **2]

        mat = p_material[p]
        h = 1.0 # Default hardening/softening factor
        mu = mu_0
        lamb = lamb_0

        # Material-specific parameters
        if mat == LIQUID:
            mu = 0.0
        elif mat == JELLO:
            h = 0.15
        elif mat == METAL:
            h = 2.0
        elif mat == SNOW:
            h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        elif mat == HONEY:
            mu = honey_mu
            lamb = honey_lambda

        # Adjust parameters based on hardening
        if mat != HONEY: # Honey uses its own mu/lambda directly
            mu = mu_0 * h
            lamb = lamb_0 * h

        # SVD to get rotation R (part of polar decomposition)
        U, SIG, V = ti.svd(F[p])
        R = U @ V.transpose()

        # Calculate determinant J (volume change ratio)
        J = 1.0
        for d in ti.static(range(dim)):
            J *= SIG[d, d]

        cauchy_stress = ti.Matrix.zero(ti.f32, dim, dim)

        if mat == LIQUID:
            # Reset deformation gradient for fluid
            F[p] = ti.Matrix.identity(ti.f32, dim) * (J**(1.0/dim))
            cauchy_stress = lamb * J * (J - 1.0) * ti.Matrix.identity(ti.f32, dim)
        elif mat == HONEY:
            # Honey behavior - highly viscous fluid
            stress = 2 * mu * (F[p] - R) @ F[p].transpose() + \
                    ti.Matrix.identity(ti.f32, dim) * lamb * J * (J - 1)

            # Apply yield criterion
            if stress.norm() > honey_yield:
                F[p] = ti.Matrix.identity(ti.f32, dim) * ti.sqrt(J) # Simplified reset, could be more sophisticated
            cauchy_stress = stress
        elif mat == SNOW:
            # Snow plasticity
            # Jp update from SIG seems problematic, let's simplify:
            J_new = 1.0
            for d in ti.static(range(dim)):
                SIG[d, d] = min(max(SIG[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                J_new *= SIG[d,d]
            Jp[p] = J_new # This might be what was intended for Jp update
            F[p] = U @ SIG @ V.transpose()
            # Or simpler, update Jp based on total volume change
            # Jp[p] = J
            cauchy_stress = 2 * mu * (F[p] - R) @ F[p].transpose() + \
                           ti.Matrix.identity(ti.f32, dim) * lamb * J * (J - 1)
        else: # JELLO or METAL
            cauchy_stress = 2 * mu * (F[p] - R) @ F[p].transpose() + \
                           ti.Matrix.identity(ti.f32, dim) * lamb * J * (J - 1)

        # Calculate the stress contribution term
        stress_term = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * cauchy_stress
        Q = stress_term + material_mass[mat] * C[p]

        for offs in ti.static(ti.ndrange(*([3] * dim))):
            grid_offset = ti.Vector(offs)
            x_i = grid_offset * dx # position of grid node in grid space units from base_coord
            x_p = fracXY * dx    # position of particle in grid space units from base_coord

            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offs[d]][d]

            idx = base_coord + grid_offset
            grid_v[idx] += weight * (material_mass[mat] * v[p] + Q @ (x_i-x_p)) # APIC P2G
            grid_m[idx] += weight * material_mass[mat]

    # Grid operations
    for I in ti.grouped(grid_m):
        if grid_m[I] > 1e-10: # if grid node has mass
            grid_v[I] /= grid_m[I]
            grid_v[I] += dt * g

            # Boundary conditions
            for d in ti.static(range(dim)):
                if I[d] < 3 and grid_v[I][d] < 0:
                    grid_v[I][d] = 0
                if I[d] >= grid_res - 3 and grid_v[I][d] > 0:
                    grid_v[I][d] = 0
        else:
            grid_v[I].fill(0.0)

        # Rigid fan interaction
        if ti.static(dim == 2):
            for fan_idx in ti.static(range(N_FANS)):
                pos = I.cast(float) * dx # grid node position
                rel = pos - rigid_center[fan_idx]
                for k in ti.static(range(5)): # 5 blades for the fan
                    φ_fan = theta[fan_idx] + 2 * math.pi * k / 5
                    dir_k = ti.Vector([ti.cos(φ_fan), ti.sin(φ_fan)]) # direction of the k-th blade
                    proj = rel.dot(dir_k) # projection of relative position onto blade direction

                    if 0 <= proj <= rigid_radius[fan_idx]: # if along the blade
                        perp = rel - dir_k * proj # perpendicular component
                        d = perp.norm()
                        if d < dx: # if close enough to the blade's line
                            n = perp / (d + 1e-6) # normal pointing away from blade line
                            v_r = ti.Vector([-omega[fan_idx] * rel.y, omega[fan_idx] * rel.x]) # fan's surface velocity
                            v_rel = grid_v[I] - v_r # relative velocity of grid node to fan
                            vn = v_rel.dot(n) # normal component of relative velocity
                            if vn < 0: # if moving towards the blade
                                vt = v_rel - vn * n # tangential component of relative velocity
                                mu_f = 0.1 # friction coefficient
                                vt_norm = vt.norm() + 1e-6
                                vt_dir = vt / vt_norm
                                vt_new_norm = max(vt_norm + mu_f * vn, 0) # apply friction (Coulomb)
                                vt_new = vt_new_norm * vt_dir
                                grid_v[I] = v_r + vt_new # update grid velocity

    # G2P
    for p in x:
        base_coord = (x[p] * inv_dx - 0.5).cast(int)
        fracXY = x[p] * inv_dx - base_coord.cast(float)
        w = [0.5 * (1.5 - fracXY)**2,
             0.75 - (fracXY - 1.0)**2,
             0.5 * (fracXY - 0.5)**2]

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)

        for offs in ti.static(ti.ndrange(*([3]*dim))):
            grid_offset = ti.Vector(offs)
            idx = base_coord + grid_offset
            dpos_grid_units = grid_offset.cast(float) - fracXY # distance from particle to grid node in grid units

            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offs[d]][d]

            g_v = grid_v[idx]
            new_v += weight * g_v
            new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos_grid_units) # APIC C update

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]

        # Rigid fan collision for particles (simple position correction)
        if ti.static(dim == 2):
            for fan_idx in ti.static(range(N_FANS)):
                rel_p = x[p] - rigid_center[fan_idx]
                for k in ti.static(range(5)): # 5 blades
                    φ_fan = theta[fan_idx] + 2 * math.pi * k / 5
                    dir_k = ti.Vector([ti.cos(φ_fan), ti.sin(φ_fan)])
                    proj = rel_p.dot(dir_k)

                    if 0 <= proj <= rigid_radius[fan_idx]: # Along the blade
                        perp = rel_p - dir_k * proj
                        d = perp.norm()
                        if d < dx * 0.5: # Particle radius collision (using dx*0.5 as proxy for particle size)
                            n = perp / (d + 1e-6)
                            # Push particle out
                            x[p] = rigid_center[fan_idx] + dir_k * proj + n * (dx * 0.5)
                            # Reflect velocity (simple)
                            vn = v[p].dot(n)
                            if vn < 0:
                                v[p] -= (1.0 + 0.0) * vn * n # Elastic collision with normal

    if ti.static(dim == 2):
        for fan_idx in ti.static(range(N_FANS)):
            theta[fan_idx] += omega[fan_idx] * dt
@ti.kernel
def init():
    # Initialize fan properties
    rigid_center[0] = ti.Vector([0.25, 0.12])
    rigid_radius[0] = 0.1
    omega[0] = 15.0  # Clockwise for left fan
    theta[0] = 0.0

    rigid_center[1] = ti.Vector([0.75, 0.12])
    rigid_radius[1] = 0.1
    omega[1] = -15.0 # Counter-clockwise for right fan
    theta[1] = 0.0

    # Particle distribution parameters
    current_particle_idx = 0

    # Bottom fluids
    fluid_height = 0.1
    num_liquid_particles = int(num_particles * 0.20)
    num_honey_particles = int(num_particles * 0.20)

    # Liquid (left half)
    for i in range(num_liquid_particles):
        p = current_particle_idx + i
        p_material[p] = LIQUID
        x[p] = [
            0.05 + ti.random() * 0.40, # x in [0.05, 0.45]
            0.02 + ti.random() * fluid_height # y in [0.02, 0.02 + fluid_height]
        ]
        v[p] = [0, 0]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += num_liquid_particles

    # Honey (right half)
    for i in range(num_honey_particles):
        p = current_particle_idx + i
        p_material[p] = HONEY
        x[p] = [
            0.55 + ti.random() * 0.40, # x in [0.55, 0.95]
            0.02 + ti.random() * fluid_height # y in [0.02, 0.02 + fluid_height]
        ]
        v[p] = [0, 0]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += num_honey_particles

    # Dropping 10 Smaller Cubes in a pyramid shape
    cube_size = 0.06  # Smaller cube size
    particles_per_small_cube = int((num_particles * 0.60) / 10)  # 60% of particles divided among 10 cubes

    # Calculate the center position for the pyramid
    center_x = 0.5
    base_y = 0.5  # Base height above the liquid

    # Parameters for positioning
    horizontal_spacing = cube_size * 1.5  # Space between cubes horizontally
    vertical_spacing = cube_size * 1.2    # Space between layers vertically

    # Create base layer (4 cubes)
    # Cube 0: Base layer, leftmost
    material_id = JELLO
    cube_center_x = center_x - 1.5 * horizontal_spacing
    cube_center_y = base_y + 0.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Cube 1: Base layer, left-center
    material_id = SNOW
    cube_center_x = center_x - 0.5 * horizontal_spacing
    cube_center_y = base_y + 0.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Cube 2: Base layer, right-center
    material_id = SNOW
    cube_center_x = center_x + 0.5 * horizontal_spacing
    cube_center_y = base_y + 0.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Cube 3: Base layer, rightmost
    material_id = JELLO
    cube_center_x = center_x + 1.5 * horizontal_spacing
    cube_center_y = base_y + 0.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Create second layer (3 cubes)
    # Cube 4: Second layer, left
    material_id = METAL
    cube_center_x = center_x - 1.0 * horizontal_spacing
    cube_center_y = base_y + 1.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Cube 5: Second layer, center
    material_id = SNOW
    cube_center_x = center_x + 0.0 * horizontal_spacing
    cube_center_y = base_y + 1.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Cube 6: Second layer, right
    material_id = JELLO
    cube_center_x = center_x + 1.0 * horizontal_spacing
    cube_center_y = base_y + 1.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Create third layer (2 cubes)
    # Cube 7: Third layer, left
    material_id = METAL
    cube_center_x = center_x - 0.5 * horizontal_spacing
    cube_center_y = base_y + 2.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Cube 8: Third layer, right
    material_id = METAL
    cube_center_x = center_x + 0.5 * horizontal_spacing
    cube_center_y = base_y + 2.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube

    # Create top layer (1 cube)
    # Cube 9: Top layer, center
    material_id = JELLO
    cube_center_x = center_x + 0.0 * horizontal_spacing
    cube_center_y = base_y + 3.0 * vertical_spacing
    for i in range(particles_per_small_cube):
        p = current_particle_idx + i
        p_material[p] = material_id
        x[p] = [
            cube_center_x - cube_size/2 + ti.random() * cube_size,
            cube_center_y - cube_size/2 + ti.random() * cube_size
        ]
        v[p] = [0, -0.2]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        C[p].fill(0.0)
        Jp[p] = 1.0
    current_particle_idx += particles_per_small_cube
# Visualization setup
#
palette_colors = [
    0x4169E1,  # LIQUID (index 0)
    0xFF69B4,  # JELLO (index 1)
    0x676767,  # METAL (index 2)
    0xFFFFFF,  # SNOW (index 3)
    0xFFA500   # HONEY (index 4)
]

RECORD = True # Set to True to record a video

if __name__ == '__main__':
    init()

    output_dir = './mpm_multi_material_fan_video'
    if RECORD:
        os.makedirs(output_dir, exist_ok=True)
        video_manager = VideoManager(output_dir=output_dir,
                                   framerate=30, # Lowered framerate for faster video generation
                                   automatic_build=False) # Set to False if ffmpeg issues, then build manually

    if dim == 3:
        # 3D visualization code would go here
        print("3D visualization not implemented in this example.")
        pass
    else:
        gui = ti.GUI("MLS-MPM 2D (Multi-Material & Fans)", (512, 512), background_color=0x111111)
        frame = 0
        max_frames = 300 # Record for 10 seconds at 30fps

        while gui.running and not gui.get_event(gui.ESCAPE):
            if RECORD and frame >= max_frames:
                break

            p_material_np = p_material.to_numpy()
            print(f"--- Frame {frame} Debug Info ---")
            print(f"  p_material field shape: {p_material_np.shape}")
            min_val = np.min(p_material_np)
            max_val = np.max(p_material_np)
            print(f"  Min value in p_material: {min_val}")
            print(f"  Max value in p_material: {max_val}")
            print(f"  Length of palette_colors: {len(palette_colors)}")

            unique_ids, counts = np.unique(p_material_np, return_counts=True)
            print(f"  Unique material IDs and counts: {dict(zip(unique_ids, counts))}")

            # Check for values that would cause the assertion to fail
            # (i.e., values >= len(palette_colors))
            if max_val >= len(palette_colors):
                print(f"  ERROR DETECTED: Max material ID ({max_val}) is >= len(palette_colors) ({len(palette_colors)})")
                problematic_indices = np.where(p_material_np >= len(palette_colors))[0]
                print(f"  Found {len(problematic_indices)} particles with out-of-bounds material IDs.")
                print(f"  First 10 particle indices with problematic material IDs: {problematic_indices[:10]}")
                if len(problematic_indices) > 0:
                    example_particle_index = problematic_indices[0]
                    print(f"    Example: Particle at original index p={example_particle_index} has material_id={p_material_np[example_particle_index]}")
            else:
                print("  Max material ID is within palette bounds.")

            for s in range(substeps):
                sub_step()

            gui.circles(x.to_numpy(),
                       radius=1.0, # Adjust for particle size visibility
                       palette=palette_colors,
                       palette_indices=p_material)

            # Draw rigid fans
            for i in range(N_FANS):
                center_np = rigid_center[i].to_numpy()
                current_theta = theta[i]
                current_radius = rigid_radius[i]
                for k in range(5): # 5 blades
                    φ = current_theta + 2 * math.pi * k / 5
                    p0_vec = ti.Vector([center_np[0], center_np[1]])
                    p1_vec = p0_vec + ti.Vector([ti.cos(φ), ti.sin(φ)]) * current_radius
                    gui.line(p0_vec.to_numpy(), p1_vec.to_numpy(), radius=2, color=0xFF0000) # Red fans

            if RECORD:
                video_manager.write_frame(gui.get_image())
                if frame % 30 == 0: # Print progress
                    print(f"Recording frame {frame}/{max_frames}")


            gui.show()
            frame += 1

        if RECORD:
            print("Building video...")
            video_manager.make_video(gif=False, mp4=True) # Create mp4
            print(f"Video saved to {output_dir}")
