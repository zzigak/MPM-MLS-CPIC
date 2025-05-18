import taichi as ti 
import numpy as np 


ti.init(arch=ti.gpu)



grid_res = 128
num_particles = 9000

dt = 1e-4
dx = 1 / grid_res
inv_dx = float(grid_res)

rho = 1 
p_vol = (dx * 1.0)**2.0
p_m = p_vol * rho

E = 0.1e4 # youngs-modulus
nu = 0.2 # poisson's ratio

mu_0, lamb_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame

plastic = 0 # bool


# define particle properties
# TODO: potentially combine into a struct

x = ti.Vector.field(2, dtype=ti.f32, shape=num_particles) # Position
v = ti.Vector.field(2, dtype=ti.f32, shape=num_particles) # Velocity
C = ti.Matrix.field(2,2, dtype=ti.f32, shape=num_particles) # affine velocity matrix
F = ti.Matrix.field(2,2, dtype=ti.f32, shape=num_particles) # deformation gradient
J = ti.field(dtype=ti.f32, shape=num_particles) # determinant of the deformation

grid = ti.Vector.field(3, dtype=ti.f32, shape=(grid_res, grid_res))
#gravity = ti.Vector.field(2, dtype=ti.f32, shape=())
g = ti.Vector([0,-9.8])

substeps = 50

# rigid body definitions
rigid_center = ti.Vector.field(2, dtype=ti.f32, shape=())
rigid_radius = ti.field(dtype=ti.f32, shape=())
omega        = ti.field(dtype=ti.f32, shape=())
theta        = ti.field(dtype=ti.f32, shape=())


@ti.kernel
def sub_step():
# TODO: implement P2G, G update, G2P, advection

    # clear grid
    for i,j in grid:
        grid[i,j] = [0.0,0.0,0.0]

    
    # P2G
    for p in x:
        # update parameters, stress, etc.

        # get coordinates of the lower-left grid node around the particle
        base_coord = (x[p] * (1/dx) - 0.5).cast(int)
        
        # fracXY.x is how far (in grid‑units) you are inside the cell
        fracXY = x[p] * (1/dx) - base_coord.cast(float)

        # interpolation kernel (quadratic, can change to qubic)
        w = [0.5 * (1.5 - fracXY) ** 2, 0.75 - (fracXY -1) **2, 0.5 * (fracXY -0.5) **2]

        # volume change
        J_ = ti.math.determinant(F[p])

        # TODO: polar decompose F 
        U, SIG, V = ti.svd(F[p])
        R = U @ V.transpose()
        S = V @ SIG @ V.transpose()

        # PK1 = \partial_{F}\psi(F) = 2 \mu (F-R) + \lambda (J-1) JF^{-T}
        PK = 2 * mu_0 * (F[p]-R)  + lamb_0 * (J_ - 1) * J_ * F[p].inverse().transpose()
       
        stress = - dt * p_vol * (4 / (dx**2))  * PK @ F[p].transpose() 

        apic_mom = C[p] * p_m

        Q = stress + apic_mom # see equation (29) in the paper


        # transfer to grid
        for i, j in ti.static(ti.ndrange(3,3)): 
            # average over neighborhood of grid nodes
            grid_offset = ti.Vector([i,j])
            x_i = grid_offset * dx
            x_p = fracXY * dx

            weight = w[i][0] * w[j][1]

            mom = p_m * v[p] # interal term

            stf = Q @ (x_i - x_p) # i think this is the force term

            grid[base_coord + grid_offset] += weight * ti.Vector([mom.x + stf.x, mom.y + stf.y, p_m])


     
    # forall grid nodes, apply gravity, handle collisions with boundaries 
    for i,j in grid:
        if grid[i,j][2] > 0: # grid mass > 0
            
            # compute velocity
            grid[i,j].xy /= grid[i,j][2]
            grid[i,j].xy += dt * g

            # collision with spinning fan blades
            pos = ti.Vector([i, j]) * dx
            rel = pos - rigid_center[None]
            for k in ti.static(range(5)):
                φ = theta[None] + 2 * 3.1415926 * k / 5
                dir_k = ti.Vector([ti.cos(φ), ti.sin(φ)])
                proj = rel.dot(dir_k)
                if 0 <= proj <= rigid_radius[None]:
                    perp = rel - dir_k * proj
                    d = perp.norm()
                    if d < dx:
                        n = perp / (d + 1e-6)
                        v_rigid = ti.Vector([-omega[None] * rel.y, omega[None] * rel.x])
                        v_rel = grid[i,j].xy - v_rigid
                        vn = v_rel.dot(n)
                        if vn < 0:
                            vt = v_rel - vn * n
                            mu_f = 0.1
                            vt_norm = vt.norm() + 1e-6
                            vt_dir = vt / vt_norm
                            vt_new = max(vt_norm + mu_f * vn, 0) * vt_dir
                            grid[i,j].xy = v_rigid + vt_new

            # wall bounces
            if i < 3 and grid[i,j].x < 0: grid[i,j].x = 0
            if i > grid_res - 3 and grid[i,j].x > 0: grid[i,j].x = 0
            if j < 3 and grid[i,j].y < 0: grid[i,j].y = 0
            if j > grid_res - 3 and grid[i,j].y > 0: grid[i,j].y = 0
            

    for p in x:
        # recompute base_coord & fracXY as above
        base_coord = (x[p] * inv_dx - 0.5).cast(int)
        fracXY     = x[p] * inv_dx - base_coord.cast(ti.f32)

        w = [
            0.5 * (1.5 - fracXY)**2,
            0.75 - (fracXY - 1.0)**2,
            0.5 * (fracXY - 0.5)**2
        ]

        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)

        # gather from 3×3 neighborhood
        for I, J in ti.static(ti.ndrange(3, 3)):
            node_rel_pos = (ti.Vector([I, J]).cast(ti.f32) - fracXY) * dx

            weight = w[I].x * w[J].y

            idx = base_coord + ti.Vector([I, J])
            gv  = ti.Vector([grid[idx].x, grid[idx].y])

            new_v += weight * gv

            # accumulate affine C
            new_C += 4 * inv_dx * weight * gv.outer_product(node_rel_pos)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]

        F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]

        # penetration correction against fan blades
        rel_p = x[p] - rigid_center[None]
        for k in ti.static(range(5)):
            φ = theta[None] + 2 * 3.1415926 * k / 5
            dir_k = ti.Vector([ti.cos(φ), ti.sin(φ)])
            proj = rel_p.dot(dir_k)
            if 0 <= proj <= rigid_radius[None]:
                perp = rel_p - dir_k * proj
                d = perp.norm()
                if d < dx:
                    n = perp / (d + 1e-6)
                    x[p] = rigid_center[None] + dir_k * proj + n * dx
                    vn = v[p].dot(n)
                    if vn < 0:
                        v[p] -= vn * n

    # rigid body angle update
    theta[None] += omega[None] * dt


@ti.kernel
def reset_state():
    # TODO: if we want to allow reset of state 
    for i in range(num_particles):
        x[i] = [
            ti.random() * 0.2 + 0.3,  
            ti.random() * 0.2 + 0.3,  
        ]
        v[i] = [0.0, 0.0]
        F[i] = ti.Matrix([[1.0, 0.0],
                          [0.0, 1.0]])
        J[i] = 1.0
        C[i] = ti.Matrix.zero(ti.f32, 2, 2) 



@ti.kernel
def init():
    for i in range(num_particles):
        x[i] = [
            ti.random(),
            ti.random() * 0.5 + 0.3,
        ]
        v[i] = [0.0, 0.0]
        F[i] = ti.Matrix([[1.0, 0.0],
                          [0.0, 1.0]])
        J[i] = 1.0
        C[i] = ti.Matrix.zero(ti.f32, 2, 2) 


def compute_density_cell_centers(grid_np, dx):
    m = grid_np[..., 2]
    c00 = m[:-1, :-1]; c10 = m[1:, :-1]
    c11 = m[1:,  1:]; c01 = m[:-1, 1:]
    cell_mass = 0.25 * (c00 + c10 + c11 + c01)
    return cell_mass / (dx * dx)


def marching_squares(dens, thresh):
    H, W = dens.shape
    segments = []

    # connectivity table
    edge_table = {
        1:  [(3, 0)],
        2:  [(0, 1)],
        3:  [(3, 1)],
        4:  [(1, 2)],
        5:  [(3, 2), (0, 1)],
        6:  [(0, 2)],
        7:  [(3, 2)],
        8:  [(2, 3)],
        9:  [(2, 0)],
        10: [(2, 1), (3, 0)],
        11: [(2, 1)],
        12: [(1, 3)],
        13: [(1, 0)],
        14: [(0, 3)]
    }

    # interpolation
    def interp(pa, pb, va, vb):
        t = (thresh - va) / (vb - va + 1e-12)
        return pa + t * (pb - pa)

    for i in range(H - 1):
        for j in range(W - 1):
            f = [
                dens[i,   j],
                dens[i+1, j],
                dens[i+1, j+1],
                dens[i,   j+1],
            ]
            # 4‐bit index
            idx = sum(1 << c for c in range(4) if f[c] > thresh)
            if idx == 0 or idx == 15:
                continue

            # resolve ambiguity
            connect = edge_table[idx]
            if idx in (5, 10):
                center_val = sum(f) / 4.0
                if (idx == 5 and center_val > thresh) or (idx == 10 and center_val <= thresh):
                    connect = [connect[1], connect[0]]

            # corner positions
            corners = [
                np.array([i,   j  ], float),
                np.array([i+1, j  ], float),
                np.array([i+1, j+1], float),
                np.array([i,   j+1], float),
            ]

            # compute segment
            for e0, e1 in connect:
                def point_on_edge(e):
                    A, B = {0:(0,1), 1:(1,2), 2:(2,3), 3:(3,0)}[e]
                    return interp(corners[A], corners[B], f[A], f[B])

                p0 = point_on_edge(e0)
                p1 = point_on_edge(e1)
                segments.append((p0, p1))

    return segments

if __name__ == '__main__':

    # TODO: set UI. create a window, set canvas, bg color, scene, camera

    gui = ti.GUI("MLS-MPM in 2D", res=512, background_color=0x000000)
    init()

    # rigid body initialization (lowered)
    rigid_center[None] = ti.Vector([0.5, 0.2])
    rigid_radius[None] = 0.15
    omega[None] = 10.0    # constant angular speed
    theta[None] = 0.0

    while gui.running and not gui.get_event(gui.ESCAPE):
        for s in range(substeps):
            sub_step()

        # actually display
        gui.clear(0x000000)
        gui.circles(x.to_numpy(), radius=1 , color=0x068587)

        for k in range(5):
            φ = theta[None] + 2 * 3.1415926 * k / 5
            p0 = rigid_center[None]
            p1 = p0 + ti.Vector([ti.cos(φ), ti.sin(φ)]) * rigid_radius[None]
            gui.line(p0.to_numpy(), p1.to_numpy(), radius=2, color=0xFF0000)
        
        # marching squares
        grid_np = grid.to_numpy()
        dens = compute_density_cell_centers(grid_np, dx)
        segments = marching_squares(dens, thresh=0.02)

        # draw contour segments
        for p0, p1 in segments:
            a = (p0 + 0.5) / grid_res
            b = (p1 + 0.5) / grid_res
            gui.line(a.tolist(), b.tolist(), radius=1, color=0xFFFFFF)
            
        gui.show()



