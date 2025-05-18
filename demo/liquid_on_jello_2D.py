import taichi as ti
import numpy as np
import math
import os

# Use ti.f32 for consistency
ti.init(arch=ti.gpu, default_fp=ti.f32) # Try to run on GPU

dim = 2
grid_res = 128
num_particles = 4000 # Increased to accommodate two blocks clearly
dt = 1e-4
dx = 1 / grid_res
inv_dx = float(grid_res)

# material properties
rho = 1.0
p_vol = (dx * 0.5)**dim
p_m = p_vol * rho

E = 5e3 # youngs-modulus
nu = 0.2 # poisson's ratio

mu_0, lamb_0 = E / (2*(1+nu)), E*nu / ((1+nu) * (1-2*nu)) # Lame params


# define particle properties
x = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles) # Position
v = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles) # Velocity
C = ti.Matrix.field(dim,dim , dtype=ti.f32, shape=num_particles) # affine velocity matrix
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=num_particles) # deformation gradient
p_material = ti.field(dtype=ti.int32, shape=num_particles) # Material ID per particle (0: liquid, 1: jello)
if dim == 3:
    p_color = ti.Vector.field(3, dtype=ti.f32, shape=num_particles) # Color field for 3D visualization


# stores momentum and mass
grid = ti.Vector.field(dim + 1, dtype=ti.f32, shape=(grid_res, ) * dim)

# gravity vector
g_val = -9.8
if dim == 3:
    g = ti.Vector([0.0, g_val, 0.0])
else:
    g = ti.Vector([0.0, g_val])

substeps = 50


@ti.kernel
def sub_step():
    # clear grid
    for I in ti.grouped(grid):
        grid[I].fill(0.0)

    # P2G
    for p in x:
        # update F = (I + dt * C) * F
        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]

        # get coordinates of the lower-left grid node around the particle
        base_coord = (x[p] * inv_dx - 0.5).cast(int)

        # fracXY.x is how far (in grid‑units) you are inside the cell
        fracXY = x[p] * inv_dx - base_coord.cast(float)

        # interpolation kernel (quadratic, can change to qubic)
        w = [0.5 * (1.5 - fracXY) ** 2, 0.75 - (fracXY - 1.0) **2, 0.5 * (fracXY - 0.5) **2]


        h = 1.0 # Default hardening/softening factor
        mu = mu_0
        lamb = lamb_0
        is_fluid = False

        if p_material[p] == 0: # liquid
            mu = 0.0
            is_fluid = True
        elif p_material[p] == 1: # jello
            h = 0.3 # make jello softer
            mu = mu_0 * h
            lamb = lamb_0 * h
            is_fluid = False
        # Add other material types here if needed

        # SVD to get rotation R (part of polar decomposition)
        U, SIG, V = ti.svd(F[p])
        R = U @ V.transpose()

        # Calculate determinant J (volume change ratio)
        J = 1.0
        for d in ti.static(range(dim)):
            J *= SIG[d, d]


        cauchy_stress = ti.Matrix.zero(ti.f32, dim, dim)

        if is_fluid:
             # liqud
            F[p] = ti.Matrix.identity(ti.f32, dim) * (J**(1.0/dim)) # Reset deformation gradient for fluid
            cauchy_stress = lamb * J * (J - 1.0) * ti.Matrix.identity(ti.f32, dim) # Pressure term only
        else: # jello (or other elastic materials)
            cauchy_stress = 2.0 * mu * (F[p] - R) @ F[p].transpose() + \
                            lamb * J * (J - 1.0) * ti.Matrix.identity(ti.f32, dim)

        # Calculate the stress contribution term for the grid momentum update
        stress_term = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * cauchy_stress

        # Combine stress and APIC terms into the matrix Q used for scattering
        Q = stress_term + p_m * C[p]


        for offs in ti.static(ti.ndrange(*([3] * dim))):
            grid_offset = ti.Vector(offs)
            x_i = grid_offset * dx
            x_p = fracXY * dx

            # combine per-axis weights
            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offs[d]][d]

            idx = base_coord + grid_offset

            grid[idx][0:dim] += weight * (p_m * v[p]+ Q @ (x_i-x_p))
            grid[idx][dim]   += weight * p_m


    # forall grid nodes, apply gravity, handle collisions with boundaries
    for I in ti.grouped(grid):
        # grid mass > 0
        if grid[I][dim] > 1e-10:
            # mv / m = v
            grid[I][:dim] /= grid[I][dim]
            grid[I][:dim] += dt * g

            for d in ti.static(range(dim)):
                # if we’re within 3 cells of the lower face and pointing outwards
                if I[d] < 3 and grid[I][d] < 0:
                    grid[I][d] = 0
                # if within 3 cells of the upper face and pointing outwards
                if I[d] >= grid_res - 3 and grid[I][d] > 0:
                    grid[I][d] = 0
        else:
            grid[I][:dim].fill(0.0)

    # G2P
    for p in x:
        # recompute base_coord & fracXY as above
        base_coord = (x[p] * inv_dx - 0.5).cast(int)
        fracXY     = x[p] * inv_dx - base_coord.cast(float)
        w = [0.5 * (1.5 - fracXY)**2,
             0.75 - (fracXY - 1.0)**2,
             0.5 * (fracXY - 0.5)**2]

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)

        # gather from 3×3 (2D) or 3x3x3 (3D) neighborhood
        for offs in ti.static(ti.ndrange(*([3]*dim))):
            grid_offset = ti.Vector(offs)
            idx = base_coord + grid_offset
            dpos_grid_units = grid_offset.cast(float) - fracXY

            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offs[d]][d]

            # gather velocity vector
            g_v = grid[idx][:dim]

            # accumulate
            new_v += weight * g_v
            new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos_grid_units)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]


# @ti.kernel
# def init():
#     num_particles_per_block = num_particles // 2
#     for i in range(num_particles):
#         block_id = i // num_particles_per_block
#         p_material[i] = block_id # 0 for liquid, 1 for jello

#         if block_id == 0: # Liquid block on the left/bottom
#              if ti.static(dim == 2): # Use ti.static here
#                  x[i] = [ti.random() * 0.3 + 0.1, ti.random() * 0.4 + 0.1]
#              elif ti.static(dim == 3): # Use ti.static here
#                  x[i] = [ti.random() * 0.3 + 0.1, ti.random() * 0.4 + 0.1, ti.random()*0.3 + 0.1]
#         else: # Jello block on the right/top
#              if ti.static(dim == 2): # Use ti.static here
#                  x[i] = [ti.random() * 0.3 + 0.6, ti.random() * 0.4 + 0.1]
#              elif ti.static(dim == 3): # Use ti.static here
#                  x[i] = [ti.random() * 0.3 + 0.6, ti.random() * 0.4 + 0.1, ti.random()*0.3 + 0.6]

#         v[i].fill(0.0)
#         F[i] = ti.Matrix.identity(ti.f32, dim)
#         C[i].fill(0.0)


@ti.kernel
def init():
    num_particles_per_block = num_particles // 2
    for i in range(num_particles):

        if i < num_particles_per_block: # Liquid Block (Material 0) - Top, Larger Volume
            p_material[i] = 0
            if ti.static(dim == 2):
                 # x in [0.3, 0.7], y in [0.5, 0.9] (Width 0.4 x 0.4)
                 x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.4 + 0.5]
            elif ti.static(dim == 3):
                 # x in [0.3, 0.7], y in [0.5, 0.9], z in [0.3, 0.7] (Width 0.4 x 0.4 x 0.4)
                 x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.4 + 0.5, ti.random() * 0.4 + 0.3]
        else: # Jello Block (Material 1) - Bottom, Smaller Volume (denser)
             p_material[i] = 1
             if ti.static(dim == 2):
                 # x in [0.4, 0.6], y in [0.1, 0.3] (Width 0.2 x 0.2) -> 1/4th the area
                 x[i] = [ti.random() * 0.2 + 0.4, ti.random() * 0.2 + 0.1]
             elif ti.static(dim == 3):
                 # x in [0.4, 0.6], y in [0.1, 0.3], z in [0.4, 0.6] (Width 0.2 x 0.2 x 0.2) -> 1/8th the volume
                 x[i] = [ti.random() * 0.2 + 0.4, ti.random() * 0.2 + 0.1, ti.random() * 0.2 + 0.4]

        v[i].fill(0.0)
        F[i] = ti.Matrix.identity(ti.f32, dim)
        C[i].fill(0.0)


@ti.kernel
def init_2():
    # Initializes liquid at the top and 10 small jello cubes below it
    num_liquid_particles = num_particles // 2
    num_jello_particles = num_particles - num_liquid_particles
    num_jello_cubes = 10
    particles_per_jello_cube = num_jello_particles // num_jello_cubes

    if particles_per_jello_cube == 0:
        print("Warning: Not enough particles to distribute among 10 jello cubes. Increase num_particles.")
        particles_per_jello_cube = 1

    # Define layout parameters for jello cubes
    jello_cube_width = 0.08 # Width of each small jello cube
    jello_grid_cols = 5
    jello_grid_rows = 2 if dim == 2 else 2 # Use 2 rows for both 2D and 3D layout logic below
    # Calculate horizontal spacing and base for jello cubes
    total_jello_width_x = jello_grid_cols * jello_cube_width
    jello_spacing_x = (0.8 - total_jello_width_x) / (jello_grid_cols - 1) if jello_grid_cols > 1 else 0
    jello_base_x = 0.1 + (1.0 - 0.1 - 0.1 - total_jello_width_x - (jello_grid_cols - 1) * jello_spacing_x) / 2.0 + jello_cube_width / 2.0


    # Define vertical position for jello cubes (below the liquid)
    jello_base_y = 0.2 # Base y-center for the *lower* row of jello cubes
    jello_spacing_y = 0.15 # Vertical spacing between the centers of the two rows

    # For 3D layout (z-dimension) - Adjusting to fit 2 rows in z as well
    total_jello_width_z = jello_grid_rows * jello_cube_width
    jello_spacing_z = (0.8 - total_jello_width_z) / (jello_grid_rows - 1) if jello_grid_rows > 1 else 0
    jello_base_z = 0.1 + (1.0 - 0.1 - 0.1 - total_jello_width_z - (jello_grid_rows - 1) * jello_spacing_z) / 2.0 + jello_cube_width / 2.0


    for i in range(num_particles):
        if i < num_liquid_particles: # Liquid Block (Material 0) - Top, Large Volume
            p_material[i] = 0
            if ti.static(dim == 2):
                # x in [0.1, 0.9], y in [0.6, 0.9] (Width 0.8 x 0.3)
                x[i] = [ti.random() * 0.8 + 0.1, ti.random() * 0.3 + 0.6]
            elif ti.static(dim == 3):
                # x in [0.1, 0.9], y in [0.6, 0.9], z in [0.1, 0.9] (Width 0.8 x 0.3 x 0.8)
                x[i] = [ti.random() * 0.8 + 0.1, ti.random() * 0.3 + 0.6, ti.random() * 0.8 + 0.1]
        else: # Jello Blocks (Material 1) - Middle/Upper-Middle, 10 Small Cubes
             p_material[i] = 1

             # Determine which cube this particle belongs to
             particle_index_in_jello = i - num_liquid_particles
             cube_index = particle_index_in_jello // particles_per_jello_cube
             if cube_index >= num_jello_cubes:
                 cube_index = num_jello_cubes - 1 # Clamp index

             # Calculate grid position (col, row) of the cube
             col = cube_index % jello_grid_cols
             row = cube_index // jello_grid_cols # Determines y-level (2D) or z-level (3D)

             # Calculate center of the cube
             center_x = jello_base_x + col * (jello_cube_width + jello_spacing_x)

             if ti.static(dim == 2):
                 # In 2D, use 'row' to adjust y position for two rows
                 center_y = jello_base_y + row * jello_spacing_y # Lower row (row=0) at base_y, upper row (row=1) higher
                 # Random position within the cube
                 x[i] = [center_x + (ti.random() - 0.5) * jello_cube_width,
                         center_y + (ti.random() - 0.5) * jello_cube_width]
             elif ti.static(dim == 3):
                 # In 3D, use 'row' to determine z position, keep y consistent for rows
                 center_y = jello_base_y + 0.5 * jello_spacing_y # Place all cube centers around this average height
                 center_z = jello_base_z + row * (jello_cube_width + jello_spacing_z) # Spread cubes along z
                 # Random position within the cube
                 x[i] = [center_x + (ti.random() - 0.5) * jello_cube_width,
                         center_y + (ti.random() - 0.5) * jello_cube_width,
                         center_z + (ti.random() - 0.5) * jello_cube_width]


        # Initialize other particle properties
        v[i].fill(0.0)
        F[i] = ti.Matrix.identity(ti.f32, dim)
        C[i].fill(0.0)


@ti.kernel
def set_particle_colors():
    # Example colors: Blue for liquid (0), Red for Jello (1)
    color_liquid = ti.Vector([0.2, 0.4, 0.8])
    color_jello = ti.Vector([0.8, 0.2, 0.2])
    for i in range(num_particles):
        if p_material[i] == 0:
            p_color[i] = color_liquid
        elif p_material[i] == 1:
            p_color[i] = color_jello
        # Add more colors if more materials exist


if __name__ == '__main__':
    init_2()
    if dim == 3:
        set_particle_colors() # Set initial colors for 3D
        window = ti.ui.Window("MLS-MPM 3D General", (512, 512), vsync=True)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        canvas.set_background_color((0.1, 0.1, 0.15))

        angle = 0.0
        radius = 1.5

        while window.running and not window.is_pressed(ti.ui.ESCAPE):
            for s in range(substeps):
                sub_step()

            # rotate camera
            angle += 0.01
            cam_x = 0.5 + radius * math.sin(angle)
            cam_z = 0.5 + radius * math.cos(angle)
            camera.position(cam_x, 0.5, cam_z)
            camera.lookat(0.5, 0.5, 0.5)

            scene.set_camera(camera)
            scene.point_light((cam_x, 1.0, cam_z), (1.0, 1.0, 1.0))

            set_particle_colors() # Update colors if needed (though they don't change here)
            scene.particles(x, radius=0.005, per_vertex_color=p_color) # Use per-particle colors

            canvas.scene(scene)
            window.show()
    else: # dim == 2
        gui = ti.GUI(f"MLS-MPM 2D (Liquid & Jello)", (512, 512), background_color=0x111111)
        # Define the color palette: [liquid_color, jello_color]
        palette = [0x068587, 0xEB553B]
        while gui.running and not gui.get_event(gui.ESCAPE):
            for s in range(substeps):
                sub_step()

            # Pass the p_material field directly to palette_indices
            gui.circles(x.to_numpy(),
                        radius=1.0,
                        palette=palette,
                        palette_indices=p_material)
            gui.show()
