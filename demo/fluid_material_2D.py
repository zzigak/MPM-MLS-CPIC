import taichi as ti
import numpy as np
import math
import os

ti.init(arch=ti.gpu) # Try to run on GPU

dim = 2
grid_res = 128
num_particles = 9000 # Use a fixed number for consistency if dim changes logic
dt = 1e-4 # Reference uses 1e-4 / quality, let's try this
dx = 1 / grid_res
inv_dx = float(grid_res)

# material properties
rho = 1
p_vol = (dx * 0.5)**dim
p_m = p_vol * rho
E = 5e3 # youngs-modulus
nu = 0.2 # poisson's ratio
# Use consistent naming with reference if preferred (la vs lamb)
mu_0, lamb_0 = E / (2*(1+nu)), E*nu / ((1+nu) * (1-2*nu)) # Lame params
# For pure fluid simulation, we can assume material = 0
# plastic = 0 # Not used in fluid model directly

# define particle properties
x = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles) # Position
v = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles) # Velocity
C = ti.Matrix.field(dim,dim , dtype=ti.f32, shape=num_particles) # affine velocity matrix
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=num_particles) # deformation gradient
# J = ti.field(dtype=ti.f32, shape=num_particles) # We calculate J on the fly, don't need to store if only for fluid
# Jp = ti.field(dtype=ti.f32, shape=num_particles) # Only needed for plasticity (like snow)

# stores momentum and mass
grid = ti.Vector.field(dim + 1, dtype=ti.f32, shape=(grid_res, ) * dim)

# gravity vector
if dim == 3:
    # Note: 3D might need different dt or substeps
    g = ti.Vector([0.0, -9.8, 0.0], dt=ti.f32)
    dt = 2e-4 # Maybe revert dt for 3D if 1e-4 is too small
else:
    g = ti.Vector([0.0, -9.8], dt=ti.f32)

# Reference uses 50 substeps per frame with dt=1e-4
substeps = 50 # Adjust as needed based on dt

@ti.kernel
def sub_step():
    # clear grid
    for I in ti.grouped(grid):
        grid[I] = ti.Vector.zero(ti.f32, dim+1)

    # P2G
    for p in x:
        # Update F based on C from *last* G2P step (Standard MLS-MPM)
        # F_new = (I + dt * C) * F_old
        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]

        # P2G setup (same as before)
        base_coord = (x[p] * inv_dx - 0.5).cast(int)
        fracXY = x[p] * inv_dx - base_coord.cast(float)
        w = [0.5 * (1.5 - fracXY) ** 2, 0.75 - (fracXY - 1) **2, 0.5 * (fracXY - 0.5) **2]

        # Fluid constitutive model
        mu = 0.0 # For fluid, shear modulus is zero
        lamb = lamb_0 # Use base Lame parameter for pressure

        U, sig, V = ti.svd(F[p])

        # Calculate determinant J (volume change ratio)
        J = 1.0
        for d in ti.static(range(dim)):
             J *= sig[d, d]

        # --- Fluid specific modification ---
        # Reset deformation gradient to isotropic scaling for fluid
        # This prevents accumulation of shear stresses which fluids don't have
        # It effectively keeps only the volume change information (J)
        F[p] = ti.Matrix.identity(ti.f32, dim) * ti.sqrt(J) # Or J**(1/dim) for 3D ? Reference uses sqrt(J) for 2D. sqrt(J) is fine for 2D. For 3D use J**(1.0/dim)
        # F[p] = ti.Matrix.identity(ti.f32, dim) * (J**(1.0/dim)) # More general form


        # Calculate Cauchy stress
        # Stress = 2*mu*(F - R)*F.T + lambda*J*(J-1)*I
        # Since mu = 0 for fluid, the first term vanishes.
        # R = U @ V.transpose() # Rotation - not strictly needed if mu=0, but good for general case
        # stress = 2 * mu * (F[p] - R) @ F[p].transpose() + \
        #          lamb * J * (J - 1) * ti.Matrix.identity(ti.f32, dim)
        # Simplified for mu=0:
        stress = lamb * J * (J - 1) * ti.Matrix.identity(ti.f32, dim)

        # Calculate the stress contribution term for the grid momentum update
        # Original: stress_term = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * cauchy_stress
        # Reference: stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        # Let's follow reference structure:
        stress_term = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress

        # Combine stress and APIC terms into the matrix Q used for scattering
        # Original: Q = stress_term + p_m * C[p]
        # Reference: affine = stress + p_mass * C[p]  (Same calculation)
        Q = stress_term + p_m * C[p] # C[p] is from the *previous* G2P step

        # P2G Scatter (same as before)
        for offs in ti.static(ti.ndrange(*([3] * dim))):
            grid_offset = ti.Vector(offs)
            # dpos = (grid_offset.cast(float) - fracXY) * dx # dpos calculation in reference
            dpos = (ti.Vector(offs).cast(float) - fracXY) * dx # More direct translation of ref

            # Recompute x_i and x_p if needed, or use dpos directly like reference
            # x_i = grid_offset * dx  <- This is incorrect, should be relative to base_coord
            # x_p = fracXY * dx     <- This is incorrect, should be position within the cell
            # Correct relative position calculation:
            # node_pos = (base_coord + grid_offset) * dx
            # particle_pos = x[p]
            # dpos = node_pos - particle_pos # But reference uses dpos = (offset - fx)*dx

            # Let's use the reference way: weight * (p_mass * v[p] + affine @ dpos)
            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offs[d]][d]

            idx = base_coord + grid_offset
            grid[idx][:dim] += weight * (p_m * v[p] + Q @ dpos) # Use dpos from reference
            grid[idx][dim]  += weight * p_m

    # Grid operations (Apply gravity and boundary conditions)
    for I in ti.grouped(grid):
        if grid[I][dim] > 1e-10: # Use epsilon to avoid division by zero
            # Momentum to velocity
            grid[I][:dim] /= grid[I][dim]
            # Apply gravity
            grid[I][:dim] += dt * g

            # Boundary conditions (same as before, seems correct)
            for d in ti.static(range(dim)):
                if I[d] < 3 and grid[I][d] < 0:
                    grid[I][d] = 0
                if I[d] >= grid_res - 3 and grid[I][d] > 0: # Use >= grid_res - 3
                    grid[I][d] = 0
        else:
            grid[I][:dim] = ti.Vector.zero(ti.f32, dim) # Ensure velocity is zero if mass is near zero


    # G2P
    for p in x:
        # G2P setup (same as before)
        base_coord = (x[p] * inv_dx - 0.5).cast(int)
        fracXY     = x[p] * inv_dx - base_coord.cast(float)
        w = [0.5 * (1.5 - fracXY)**2,
             0.75 - (fracXY - 1.0)**2,
             0.5 * (fracXY - 0.5)**2]

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)

        # G2P gather (same as before, seems correct)
        for offs in ti.static(ti.ndrange(*([3]*dim))):
            grid_offset = ti.Vector(offs)
            idx = base_coord + grid_offset

            # node_rel_pos = (grid_offset.cast(ti.f32) - fracXY) * dx # Your original
            dpos = (grid_offset.cast(ti.f32) - fracXY) # Reference uses this * inv_dx in C calc, let's check
            # Reference C calc: new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            # where dpos = ti.Vector([i, j]).cast(float) - fx
            # So dpos is in grid units, not world units. Let's match reference.
            dpos_grid_units = grid_offset.cast(float) - fracXY


            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offs[d]][d]

            g_v = grid[idx][:dim]

            new_v += weight * g_v
            # Match reference calculation for C: 4 * inv_dx * weight * g_v @ dpos.T
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos_grid_units) # Use dpos in grid units

        v[p] = new_v
        C[p] = new_C # Store C for next step's F update
        x[p] += dt * v[p] # Advection

    # DO NOT update F[p] here anymore. It's done at the start of P2G.


@ti.kernel
def init():
    # Initialize particles in a specific region, e.g., a block
    for i in range(num_particles):
        # Place particles in a smaller block, e.g., bottom left quadrant
        x[i] = [ti.random() * 0.4 + 0.1 for _ in range(dim)]
        if dim == 2:
             x[i][1] = ti.random() * 0.4 + 0.1 # Example initial block
        elif dim == 3:
             x[i][1] = ti.random() * 0.4 + 0.1
             x[i][2] = ti.random() * 0.4 + 0.1

        v[i] = ti.Vector.zero(ti.f32, dim)
        F[i] = ti.Matrix.identity(ti.f32, dim)
        # J[i] = 1.0 # Not needed if calculated on the fly
        C[i] = ti.Matrix.zero(ti.f32, dim, dim)

# --- Main simulation loop (rest of your code) ---
# (Keep your existing main loop structure for 2D GUI or 3D window/video)
if __name__ == '__main__':
    init()
    if dim == 3:
        # --- set up 3D window, scene, camera ----------------
        window = ti.ui.Window("MLS-MPM 3D", (512, 512), vsync=True)
        canvas = window.get_canvas()
        scene  = window.get_scene()
        camera = ti.ui.Camera()
        canvas.set_background_color((0.9, 0.9, 0.9))

        # Optional: Video recording setup (if needed)
        record_video = False # Set to True to enable recording
        if record_video:
            home = os.path.expanduser("~")
            out_dir = os.path.join(home, "mls_mpm_frames_fluid")
            from taichi.tools.video import VideoManager # Make sure this import is available
            video_manager = VideoManager(
                output_dir=out_dir,
                video_filename="mpm_sim_fluid.mp4",
                width=512, height=512,
                framerate=24, # Corresponds to how many sim frames make a video frame
                automatic_build=False
            )
            os.makedirs(video_manager.directory, exist_ok=True)
            print(f"Video will be saved to {out_dir}/mpm_sim_fluid.mp4")


        angle  = 0.0
        radius = 1.5
        # --- main loop -------------------------------------
        while window.running and not window.is_pressed(ti.ui.ESCAPE):
            for s in range(substeps):
                sub_step()

            # rotate camera around scene
            angle += 0.01
            cam_x = 0.5 + radius * math.sin(angle)
            cam_z = 0.5 + radius * math.cos(angle)
            camera.position(cam_x, 0.5, cam_z)
            camera.lookat(0.5, 0.5, 0.5)

            scene.set_camera(camera)
            scene.point_light((cam_x, 1.0, cam_z), (1.0, 1.0, 1.0))
            # Adjust particle radius for visibility
            scene.particles(x, radius=0.005, color=(0.2, 0.4, 0.8)) # Smaller radius might be needed
            canvas.set_background_color((0.1, 0.1, 0.15)) # Darker background?
            canvas.scene(scene)


            if record_video:
                # capture this frame into the video
                img = window.get_image_buffer_as_numpy()
                video_manager.write_frame(img)

            window.show()

        # once the window closes, assemble the MP4
        if record_video:
            video_manager.make_video(mp4=True)
            print("Video saved.")


    else: # dim == 2
        gui = ti.GUI("MLS-MPM 2D Fluid", (512, 512), background_color=0x111111)
        while gui.running and not gui.get_event(gui.ESCAPE):
            for s in range(substeps):
                sub_step()
            # Use a smaller radius for better fluid look
            gui.circles(x.to_numpy(), radius=1.0, color=0x068587)
            gui.show()
