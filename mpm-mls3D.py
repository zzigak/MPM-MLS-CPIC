import taichi as ti
import numpy as np
import math
from skimage import measure
from taichi.tools.video import VideoManager
import os
import trimesh
from scipy.ndimage import gaussian_filter

ti.init(arch=ti.gpu) # Try to run on GPU



class MeshWrapper:
    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
    def __hash__(self):
        return hash(self._arr.tobytes())
    def __iter__(self):
        return iter(self._arr)
    def __getitem__(self, i):
        return self._arr[i]
    def __len__(self):
        return self._arr.shape[0]


dim = 3

grid_res = 200
num_particles = 20000 if dim == 2 else 100000

dt = 1e-4 if dim == 2 else 2e-4
dx = 1 / grid_res
inv_dx = float(grid_res)

# material properties
rho = 1
p_vol = (dx * 1.0)**3.0
p_m = p_vol * rho

# Material densities
material_rho = ti.field(dtype=float, shape=4)
material_rho[0] = 2.0  # liquid
material_rho[1] = 2.5  # jello
material_rho[2] = 4.0  # snow
material_rho[3] = 20.0  # metal

# Material masses
material_mass = ti.field(dtype=float, shape=4)
for i in range(4):
    material_mass[i] = p_vol * material_rho[i]

E = 1e4  # youngs-modulus
nu = 0.2 # poisson's ratio

mu_0, lamb_0 = E / (2*(1+nu)), E*nu / ((1+nu) * (1-2*nu)) # Lame params

# visualization settings
enable_mesh_generation = False
MESH_TYPE = 1  # 0: no mesh, 1: bunny, 2: rectangle
MATERIAL_TYPE = 2  # 0: liquid, 1: jello, 2: snow, 3: metal
use_liquid = True

plastic = 0 # bool

Jp = ti.field(dtype=ti.f32, shape=num_particles)  # plastic deformation

# Metal ball parameters
metal_ball_radius = 0.1
metal_ball_center = ti.Vector([0.5, 0.8, 0.5])  # Start high above the snow
metal_ball_velocity = ti.Vector([0.0, -5.0, 0.0])  # Initial downward velocity

# define particle properties
# TODO: potentially combine into a struct

x = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles) # Position
v = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles) # Velocity
C = ti.Matrix.field(dim,dim , dtype=ti.f32, shape=num_particles) # affine velocity matrix
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=num_particles) # deformation gradient
J = ti.field(dtype=ti.f32, shape=num_particles) # determinant of the deformation
p_material = ti.field(dtype=ti.int32, shape=num_particles) # Material ID per particle (0: liquid, 1: jello, 2: snow)

# stores momentum and mass
grid = ti.Vector.field(dim + 1, dtype=ti.f32, shape=(grid_res, ) * dim)

# gravity vector
if dim == 3:
    g = ti.Vector([0.0, -9.8, 0.0], dt=ti.f32)
else:
    g = ti.Vector([0.0, -9.8], dt=ti.f32)

substeps = 50

mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(100000,))
mesh_faces = ti.Vector.field(3, dtype=ti.i32, shape=(100000,))
mesh_vertex_count = ti.field(dtype=ti.i32, shape=())
mesh_face_count = ti.field(dtype=ti.i32, shape=())

p_color = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)  # Color field for particles

@ti.kernel
def init_mesh_collision(vertices: ti.types.ndarray(), faces: ti.types.ndarray()):
    mesh_vertex_count[None] = vertices.shape[0]
    mesh_face_count[None] = faces.shape[0]

    for i in range(vertices.shape[0]):
        mesh_vertices[i] = ti.Vector([vertices[i, 0], vertices[i, 1], vertices[i, 2]])

    for i in range(faces.shape[0]):
        mesh_faces[i] = ti.Vector([faces[i, 0], faces[i, 1], faces[i, 2]])

@ti.func
def compute_closest_point_on_triangle(p, a, b, c):
    # Compute closest point on triangle to point p
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = ab.dot(ap)
    d2 = ac.dot(ap)

    result = a
    should_return = False

    if d1 <= 0.0 and d2 <= 0.0:
        result = a
        should_return = True

    if not should_return:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)

        if d3 >= 0.0 and d4 <= d3:
            result = b
            should_return = True

    if not should_return:
        cp = p - c
        d5 = ab.dot(cp)
        d6 = ac.dot(cp)

        if d6 >= 0.0 and d5 <= d6:
            result = c
            should_return = True

    if not should_return:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            result = a + v * ab
            should_return = True

    if not should_return:
        cp = p - c
        d5 = ab.dot(cp)
        d6 = ac.dot(cp)
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            v = d2 / (d2 - d6)
            result = a + v * ac
            should_return = True

    if not should_return:
        bp = p - b
        cp = p - c
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        d5 = ab.dot(cp)
        d6 = ac.dot(cp)
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            result = b + v * (c - b)
            should_return = True

    if not should_return:
        bp = p - b
        cp = p - c
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        d5 = ab.dot(cp)
        d6 = ac.dot(cp)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        result = a + v * ab + w * ac

    return result

@ti.func
def check_mesh_collision(p_pos):
    min_dist = float('inf')
    closest_point = ti.Vector([0.0, 0.0, 0.0])
    collision_normal = ti.Vector([0.0, 0.0, 0.0])
    has_collision = False

    for f in range(mesh_face_count[None]):
        a = mesh_vertices[mesh_faces[f][0]]
        b = mesh_vertices[mesh_faces[f][1]]
        c = mesh_vertices[mesh_faces[f][2]]

        closest = compute_closest_point_on_triangle(p_pos, a, b, c)
        dist = (p_pos - closest).norm()

        if dist < min_dist:
            min_dist = dist
            closest_point = closest
            # Compute normal of the triangle
            edge1 = b - a
            edge2 = c - a
            normal = edge1.cross(edge2).normalized()
            collision_normal = normal

    if min_dist < 0.01:  # Collision threshold
        has_collision = True

    return has_collision, closest_point, collision_normal

@ti.kernel
def sub_step():
    # clear grid
    for I in ti.grouped(grid):
        grid[I] = ti.Vector.zero(ti.f32, dim+1)

    # P2G
    for p in x:
        # get coordinates of the lower-left grid node around the particle
        base_coord = (x[p] * (1/dx) - 0.5).cast(int)

        # fracXY.x is how far (in grid‑units) you are inside the cell
        fracXY = x[p] * (1/dx) - base_coord.cast(float)

        # interpolation kernel (quadratic, can change to qubic)
        w = [0.5 * (1.5 - fracXY) ** 2, 0.75 - (fracXY -1) **2, 0.5 * (fracXY -0.5) **2]

        # (F is updated later in G2P)
        current_F = F[p]

        # SVD to get rotation R (part of polar decomposition)
        U, SIG, V = ti.svd(current_F)
        R = U @ V.transpose()

        # Calculate determinant J (volume change ratio)
        J_ = 1.0
        for d in ti.static(range(dim)):
             J_ *= SIG[d, d]

        # Material-specific stress calculation
        h = 1.0  # Default hardening/softening factor
        mu = mu_0
        lamb = lamb_0
        is_fluid = False

        if p_material[p] == 0:  # liquid
            mu = 0.0
            is_fluid = True
        elif p_material[p] == 1:  # jello
            h = 0.3  # make jello softer
            mu = mu_0 * h
            lamb = lamb_0 * h
            is_fluid = False
        elif p_material[p] == 2:  # snow
            # Snow gets harder when compressed
            h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
            mu = mu_0 * h
            lamb = lamb_0 * h
            is_fluid = False

            # Apply snow plasticity
            for d in ti.static(range(dim)):
                new_sig = SIG[d, d]
                new_sig = ti.min(ti.max(new_sig, 1 - 2.5e-2), 1 + 4.5e-3)
                Jp[p] *= SIG[d, d] / new_sig
                SIG[d, d] = new_sig
                J_ *= new_sig

            # Reconstruct elastic deformation gradient after plasticity
            current_F = U @ SIG @ V.transpose()

        elif p_material[p] == 3:  # metal
            h = 20.0
            mu = mu_0 * h
            lamb = lamb_0 * h
            is_fluid = False

        cauchy_stress = ti.Matrix.zero(ti.f32, dim, dim)

        if is_fluid:
            # liquid
            F[p] = ti.Matrix.identity(ti.f32, dim) * (J_**(1.0/dim))  # Reset deformation gradient for fluid
            cauchy_stress = lamb * J_ * (J_ - 1.0) * ti.Matrix.identity(ti.f32, dim)  # Pressure term only
        else:  # jello or snow
            cauchy_stress = 2.0 * mu * (current_F - R) @ current_F.transpose() + \
                          lamb * J_ * (J_ - 1.0) * ti.Matrix.identity(ti.f32, dim)

        # Calculate the stress contribution term for the grid momentum update
        stress_term = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * cauchy_stress

        # Get material mass
        mass = material_mass[p_material[p]]

        # Combine stress and APIC terms into the matrix Q used for scattering
        Q = stress_term + mass * C[p]

        for offs in ti.static(ti.ndrange(*([3] * dim))):
            grid_offset = ti.Vector(offs)
            x_i = grid_offset * dx
            x_p = fracXY * dx

            # combine per-axis weights
            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offs[d]][d]

            idx = base_coord + grid_offset

            grid[idx][0:dim] += weight * (mass * v[p] + Q @ (x_i-x_p))
            grid[idx][dim]   += weight * mass

    # forall grid nodes, apply gravity, handle collisions with boundaries
    for I in ti.grouped(grid):
        if grid[I][dim] > 0: # grid mass > 0

            grid[I][0:dim] /= grid[I][dim] # mv / m = v
            grid[I][0:dim] += dt * g


            for d in ti.static(range(dim)):
                # if we're within 3 cells of the lower face and pointing outwards
                if I[d] < 3 and grid[I][d] < 0:
                    grid[I][d] = 0
                # if within 3 cells of the upper face and pointing outwards
                if I[d] > grid_res - 3 and grid[I][d] > 0:
                    grid[I][d] = 0

                # Add tighter horizontal boundaries
                if d != 1:
                    # Left boundary (x or z)
                    if I[d] < grid_res * 0.2 and grid[I][d] < 0:
                        grid[I][d] = 0
                    # Right boundary (x or z)
                    if I[d] > grid_res * 0.8 and grid[I][d] > 0:
                        grid[I][d] = 0

            #if i > grid_res - 3 and grid[i,j].x > 0: grid[i,j].x = 0
            #if j < 3 and grid[i,j].y < 0: grid[i,j].y = 0
            #if j > grid_res - 3 and grid[i,j].y > 0: grid[i,j].y = 0


    for p in x:
        # recompute base_coord & fracXY as above
        base_coord = (x[p] * inv_dx - 0.5).cast(int)
        fracXY     = x[p] * inv_dx - base_coord.cast(ti.f32)

        w = [0.5 * (1.5 - fracXY)**2,
            0.75 - (fracXY - 1.0)**2,
            0.5 * (fracXY - 0.5)**2]

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)

        # gather from 3×3 neighborhood

        for offs in ti.static(ti.ndrange(*([3]*dim))):
            grid_offset = ti.Vector(offs)

            node_rel_pos = (grid_offset.cast(ti.f32) - fracXY) * dx

            # combine per-axis weights
            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offs[d]][d]

            idx = base_coord + grid_offset

            # gather velocity vector
            gv = grid[idx][0:dim]

            # accumulate
            new_v += weight * gv
            new_C += 4 * inv_dx * weight * gv.outer_product(node_rel_pos)


        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]

        # After updating position, check for mesh collisions
        if ti.static(dim == 3):
            has_collision, closest_point, normal = check_mesh_collision(x[p])
            if has_collision:
                # Move particle away from surface
                x[p] = closest_point + normal * 0.01
                # Reflect velocity
                v[p] = v[p] - 2.0 * v[p].dot(normal) * normal
                # Add a small outward velocity to prevent sticking
                v[p] += normal * 0.05
                # Damping
                v[p] *= 1



@ti.kernel
def reset_state():
    # TODO: if we want to allow reset of state
    for i in range(num_particles):
        x[i] = [ti.random() * 0.2 + 0.3 for _ in range(dim)]
        v[i] = ti.Vector.zero(ti.f32, dim)
        F[i] = ti.Matrix.identity(ti.f32, dim)
        J[i] = 1.0
        Jp[i] = 1.0  # Initialize plastic deformation
        C[i] = ti.Matrix.zero(ti.f32, dim, dim)
        p_material[i] = 0 if use_liquid else 1



@ti.kernel
def init():
    # First initialize all particles as snow
    for i in range(num_particles):
        if MATERIAL_TYPE == 2:  # Snow
            # Place snow particles in a layer starting from the ground
            x[i] = [ti.random() * 0.2 + 0.4,  # x position
                    ti.random() * 0.2 + 0.0,  # y position (layer from 0.0 to 0.2)
                    ti.random() * 0.2 + 0.4]  # z position
            v[i] = ti.Vector([0.0, 0.0, 0.0])  # No initial velocity
            p_material[i] = 2  # Snow material
        else:
            # Original initialization for other materials
            x[i] = [ti.random() * 0.2 + 0.4,  # x position
                    ti.random() * 0.2 + 0.7,  # y position
                    ti.random() * 0.2 + 0.4]  # z position
            v[i] = ti.Vector([0.0, -5.0, 0.0])  # Initial downward velocity
            p_material[i] = MATERIAL_TYPE

        F[i] = ti.Matrix.identity(ti.f32, dim)
        J[i] = 1.0
        Jp[i] = 1.0  # Initialize plastic deformation
        C[i] = ti.Matrix.zero(ti.f32, dim, dim)

    # If we're simulating snow, add the metal ball
    if MATERIAL_TYPE == 2:
        # Calculate how many particles we need for the metal ball
        # Using a rough estimate of volume ratio
        ball_volume = 4.0/3.0 * 3.14159 * metal_ball_radius**3
        snow_layer_volume = 0.2 * 0.2 * 0.2  # Approximate snow layer volume
        num_ball_particles = int(num_particles * (ball_volume / snow_layer_volume))

        # Initialize metal ball particles
        for i in range(num_ball_particles):
            # Generate random points in a sphere
            theta = ti.random() * 2.0 * 3.14159
            phi = ti.acos(2.0 * ti.random() - 1.0)
            r = metal_ball_radius * ti.pow(ti.random(), 1.0/3.0)  # Cube root for uniform distribution

            # Convert to Cartesian coordinates
            x[i] = metal_ball_center + ti.Vector([
                r * ti.sin(phi) * ti.cos(theta),
                r * ti.sin(phi) * ti.sin(theta),
                r * ti.cos(phi)
            ])
            v[i] = metal_ball_velocity
            p_material[i] = 3  # Metal material
            F[i] = ti.Matrix.identity(ti.f32, dim)
            J[i] = 1.0
            Jp[i] = 1.0
            C[i] = ti.Matrix.zero(ti.f32, dim, dim)

@ti.kernel
def update_particle_colors():
    for p in range(num_particles):
        if p_material[p] == 3:  # Metal
            p_color[p] = ti.Vector([1.0, 0.2, 0.2])  # Red
        else:
            p_color[p] = ti.Vector([0.2, 0.8, 1.0])  # Blue

def compute_density_cell_centers_3d(grid_np, dx):
    # average the mass at the 8 corners of each voxel,
    # then divide by voxel volume dx^3 to get density
    m = grid_np[..., dim]
    c000 = m[:-1, :-1, :-1]; c100 = m[1:, :-1, :-1]
    c010 = m[:-1, 1:, :-1]; c001 = m[:-1, :-1, 1]
    c110 = m[1:, 1:, :-1];  c101 = m[1:, :-1, 1]
    c011 = m[:-1, 1:, 1];   c111 = m[1:, 1:, 1]
    cell_mass = (c000 + c100 + c010 + c001 +
                 c110 + c101 + c011 + c111) * 0.125
    return cell_mass / (dx**3)

def save_bunny(vertices, faces, out_dir='meshes'):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_out_dir = os.path.join(script_dir, out_dir)
        os.makedirs(full_out_dir, exist_ok=True)
        path = os.path.join(full_out_dir, 'bunny.obj')
    except PermissionError as e:
        print(f"Permission error creating directory: {e}")
        home_dir = os.path.expanduser("~")
        full_out_dir = os.path.join(home_dir, "Documents", out_dir)
        os.makedirs(full_out_dir, exist_ok=True)
        path = os.path.join(full_out_dir, 'bunny.obj')
        print(f"Using alternative directory: {full_out_dir}")

    with open(path, 'w') as f:
        f.write('o bunny\n')
        # Write bunny vertices
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        # Write bunny faces
        for face in faces:
            i, j, k = face + 1
            f.write(f'f {i} {j} {k}\n')

def save_particles(verts, faces, normals, frame, out_dir='meshes'):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_out_dir = os.path.join(script_dir, out_dir)
        os.makedirs(full_out_dir, exist_ok=True)
        path = os.path.join(full_out_dir, f'particles_{frame:04d}.obj')
    except PermissionError as e:
        print(f"Permission error creating directory: {e}")
        home_dir = os.path.expanduser("~")
        full_out_dir = os.path.join(home_dir, "Documents", out_dir)
        os.makedirs(full_out_dir, exist_ok=True)
        path = os.path.join(full_out_dir, f'particles_{frame:04d}.obj')
        print(f"Using alternative directory: {full_out_dir}")

    with open(path, 'w') as f:
        f.write('o particles\n')
        # Generate UV coordinates based on vertex positions
        uvs = []
        for v in verts:
            x, y, z = v
            r = math.sqrt(x*x + y*y + z*z)
            if r == 0:
                u, v = 0, 0
            else:
                u = 0.5 + math.atan2(z, x) / (2 * math.pi)
                v = 0.5 - math.asin(y / r) / math.pi
            uvs.append([u, v])

        # Write particle vertices
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        # Write particle texture coordinates
        for uv in uvs:
            f.write(f'vt {uv[0]:.6f} {uv[1]:.6f}\n')
        # Write particle normals
        for n in normals:
            f.write(f'vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n')
        # Write particle faces
        for tri in faces:
            i, j, k = tri + 1
            f.write(f'f {i}/{i}/{i} {j}/{j}/{j} {k}/{k}/{k}\n')

def load_obj_file(file_path):
    mesh = trimesh.load(file_path)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces

def create_rectangle_mesh(width=0.1, height=1.0, depth=0.1, position=(0.5, 0.5, 0.5)):
    # Create vertices for a vertical rectangle
    vertices = np.array([
        # Front face
        [position[0] - width/2, position[1] - height/2, position[2] - depth/2],  # bottom left
        [position[0] + width/2, position[1] - height/2, position[2] - depth/2],  # bottom right
        [position[0] + width/2, position[1] + height/2, position[2] - depth/2],  # top right
        [position[0] - width/2, position[1] + height/2, position[2] - depth/2],  # top left
        # Back face
        [position[0] - width/2, position[1] - height/2, position[2] + depth/2],  # bottom left
        [position[0] + width/2, position[1] - height/2, position[2] + depth/2],  # bottom right
        [position[0] + width/2, position[1] + height/2, position[2] + depth/2],  # top right
        [position[0] - width/2, position[1] + height/2, position[2] + depth/2],  # top left
    ], dtype=np.float32)

    # Create faces (triangles)
    faces = np.array([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Back face
        [4, 6, 5], [4, 7, 6],
        # Left face
        [0, 3, 7], [0, 7, 4],
        # Right face
        [1, 5, 6], [1, 6, 2],
        # Top face
        [3, 2, 6], [3, 6, 7],
        # Bottom face
        [0, 4, 5], [0, 5, 1]
    ], dtype=np.int32)

    return vertices, faces

@ti.func
def is_in_metal_ball(pos):
    return (pos - metal_ball_center).norm() < metal_ball_radius

if __name__ == '__main__':
    init()
    if dim == 3:
        record = False

        # Initialize mesh-related variables
        vertices = None
        faces = None
        v_field = None
        i_field = None

        if MESH_TYPE == 1:  # Bunny
            # Load and process bunny
            obj_path = "bunny.obj"
            vertices, faces = load_obj_file(obj_path)
            vertices = (vertices - vertices.min(axis=0)) / (vertices.max(axis=0) - vertices.min(axis=0))
            scale = 0.3
            position = np.array([0.3, 0.0, 0.25])
            vertices = vertices * scale
            vertices = vertices + position
            save_bunny(vertices, faces, out_dir='meshes/bunny')
        elif MESH_TYPE == 2:  # Rectangle
            # Create rectangle mesh
            vertices, faces = create_rectangle_mesh(
                width=0.01,
                height=0.3,
                depth=0.3,
                position=(0.4, 0.3, 0.4)
            )
            save_bunny(vertices, faces, out_dir='meshes/rectangle')

        # Initialize mesh collision only if a mesh is selected
        if MESH_TYPE > 0:
            init_mesh_collision(vertices, faces)

            # Convert to Taichi fields for rendering
            v_field = ti.Vector.field(3, ti.f32, shape=vertices.shape[0])
            i_field = ti.field(ti.i32, shape=faces.shape[0] * 3)

            # Copy data to Taichi fields
            v_field.from_numpy(vertices)
            i_field.from_numpy(faces.reshape(-1))

        # start Taichi video recorder
        if record:
            vm = VideoManager(output_dir='.', framerate=30, automatic_build=False)

        window = ti.ui.Window("MLS-MPM 3D", (512, 512), vsync=True)
        canvas = window.get_canvas()
        scene  = window.get_scene()
        camera = ti.ui.Camera()

        # Precompute worst-case mesh sizes for marching cubes
        max_voxels       = (grid_res - 1) ** 3
        max_triangles    = max_voxels * 5            # ≤5 tris/voxel
        max_indices_flat = max_triangles * 3         # 3 verts/tri
        max_verts_est    = max_indices_flat         # every vert unique

        # Allocate mesh fields for marching cubes
        v_field_mc = ti.Vector.field(3, ti.f32, shape=(max_verts_est,))
        n_field_mc = ti.Vector.field(3, ti.f32, shape=(max_verts_est,))
        i_field_mc = ti.field(ti.i32, shape=(max_indices_flat,))

        angle  = 0.0
        radius = 1.5
        frame = 0

        while window.running:
            for s in range(substeps):
                sub_step()

            # rotate camera
            angle += 0.01
            cam_x  = 0.5 + radius * math.sin(angle)
            cam_z  = 0.5 + radius * math.cos(angle)
            camera.position(cam_x, 0.5, cam_z)
            camera.lookat(0.5, 0.5, 0.5)

            scene.set_camera(camera)
            scene.point_light((cam_x, 1.0, cam_z), (1.0, 1.0, 1.0))

            # Draw the OBJ mesh only if a mesh is selected
            if MESH_TYPE > 0 and v_field is not None and i_field is not None:
                scene.mesh(
                    v_field,
                    i_field,
                    vertex_count=vertices.shape[0],
                    index_count=faces.shape[0] * 3,
                    color=(0.8, 0.8, 0.8),
                    two_sided=True
                )

            if enable_mesh_generation:
                # marching-cubes isosurface
                grid_np = grid.to_numpy()
                dens    = compute_density_cell_centers_3d(grid_np, dx)
                # Add smoothing to reduce blobbiness
                dens = gaussian_filter(dens, sigma=0.5)

                # Skip marching cubes if density is too low
                if ti.static(enable_mesh_generation):
                    if np.max(dens) < 0.01:
                        continue

                    verts, faces_mc, normals, _ = measure.marching_cubes(dens, level=0.05)
                    verts = (verts + 0.5) / grid_res  # normalize

                    # Save only the particles for this frame
                    save_particles(verts.astype(np.float32),
                                 faces_mc.astype(np.int32),
                                 normals.astype(np.float32),
                                 frame)

                    # flatten
                    Nv       = verts.shape[0]
                    Mt       = faces_mc.shape[0]
                    idx_flat = faces_mc.reshape(-1).astype(np.int32)
                    for i in range(Nv):
                        v_field_mc[i] = verts[i].astype(np.float32)
                        n_field_mc[i] = normals[i].astype(np.float32)
                    for j in range(idx_flat.shape[0]):
                        i_field_mc[j] = int(idx_flat[j])

                    # draw marching cubes mesh
                    scene.mesh(
                        v_field_mc,
                        i_field_mc,
                        normals=n_field_mc,
                        vertex_count=Nv,
                        index_count=Mt * 3,
                        color=(0.8, 0.8, 0.8),
                        two_sided=True
                    )

            frame += 1

            # draw particles on top
            update_particle_colors()
            scene.particles(x, radius=0.001, per_vertex_color=p_color)

            # render and optionally record
            canvas.scene(scene)
            window.show()
            if record:
                img = window.get_image_buffer_as_numpy()
                vm.write_frame(img)

        # finalize video if recording
        if record:
            vm.make_video(mp4=True)

    else:
        gui = ti.GUI("MLS-MPM 2D", (512, 512), background_color=0x111111)
        while gui.running and not gui.get_event(gui.ESCAPE):
            for s in range(substeps):
                sub_step()
            gui.circles(x.to_numpy(), radius=1, color=0x068587)
            gui.show()
