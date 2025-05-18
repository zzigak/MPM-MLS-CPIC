import taichi as ti
import numpy as np

# Initialize Taichi
ti.init(arch=ti.cpu)

# Parameters
n = 128# Grid resolution
grid_spacing = 1.0 / n  # Grid spacing
window_size = 800

# Define scalar field
scalar_field = ti.field(ti.f32, shape=(n + 1, n + 1))  # +1 for the boundary

# Fields for storing contour line segments
max_lines = n * n * 2  # Maximum possible number of line segments
lines_count = ti.field(ti.i32, shape=())
lines_begin = ti.Vector.field(2, ti.f32, shape=max_lines)
lines_end = ti.Vector.field(2, ti.f32, shape=max_lines)

# Create a pixel buffer for visualization
pixels = ti.Vector.field(3, ti.f32, shape=(window_size, window_size))

@ti.kernel
def initialize_field():
    # Initialize the scalar field with random values between 0 and 1
    for i, j in scalar_field:
        scalar_field[i, j] = ti.random()

@ti.func
def get_case_index(i, j, iso_value):
    # Determine which case we're in based on the 4 corners
    idx = 0
    if scalar_field[i, j] > iso_value:
        idx |= 1
    if scalar_field[i+1, j] > iso_value:
        idx |= 2
    if scalar_field[i+1, j+1] > iso_value:
        idx |= 4
    if scalar_field[i, j+1] > iso_value:
        idx |= 8
    return idx

@ti.func
def interpolate(v1, v2, iso_value):
    # Linear interpolation between two values
    # Avoid early return by using a ternary-like operation
    t = 0.5
    if abs(v1 - v2) >= 1e-5:
        t = (iso_value - v1) / (v2 - v1)
    return t

@ti.kernel
def generate_contours(iso_value: ti.f32):
    # Reset the line counter
    lines_count[None] = 0
    
    for i, j in ti.ndrange(n, n):
        case = get_case_index(i, j, iso_value)
        
        # Get the four corner values
        val_bl = scalar_field[i, j]
        val_br = scalar_field[i+1, j]
        val_tr = scalar_field[i+1, j+1]
        val_tl = scalar_field[i, j+1]
        
        # Get normalized coordinates of the cell
        x0 = i * grid_spacing
        y0 = j * grid_spacing
        x1 = (i + 1) * grid_spacing
        y1 = (j + 1) * grid_spacing
        
        # Compute intersection points for each edge
        # Bottom edge (0)
        point0_x = x0 + interpolate(val_bl, val_br, iso_value) * grid_spacing
        point0_y = y0
        
        # Right edge (1)
        point1_x = x1
        point1_y = y0 + interpolate(val_br, val_tr, iso_value) * grid_spacing
        
        # Top edge (2)
        point2_x = x0 + interpolate(val_tl, val_tr, iso_value) * grid_spacing
        point2_y = y1
        
        # Left edge (3)
        point3_x = x0
        point3_y = y0 + interpolate(val_bl, val_tl, iso_value) * grid_spacing
        
        # Add line segments based on the case
        if case == 1:  # 0-3
            add_line_segment(point0_x, point0_y, point3_x, point3_y)
        elif case == 2:  # 0-1
            add_line_segment(point0_x, point0_y, point1_x, point1_y)
        elif case == 3:  # 1-3
            add_line_segment(point1_x, point1_y, point3_x, point3_y)
        elif case == 4:  # 1-2
            add_line_segment(point1_x, point1_y, point2_x, point2_y)
        elif case == 5:  # Ambiguous case - 0-1, 2-3
            add_line_segment(point0_x, point0_y, point1_x, point1_y)
            add_line_segment(point2_x, point2_y, point3_x, point3_y)
        elif case == 6:  # 0-2
            add_line_segment(point0_x, point0_y, point2_x, point2_y)
        elif case == 7:  # 2-3
            add_line_segment(point2_x, point2_y, point3_x, point3_y)
        elif case == 8:  # 2-3
            add_line_segment(point2_x, point2_y, point3_x, point3_y)
        elif case == 9:  # 0-2
            add_line_segment(point0_x, point0_y, point2_x, point2_y)
        elif case == 10:  # Ambiguous case - 0-3, 1-2
            add_line_segment(point0_x, point0_y, point3_x, point3_y)
            add_line_segment(point1_x, point1_y, point2_x, point2_y)
        elif case == 11:  # 1-2
            add_line_segment(point1_x, point1_y, point2_x, point2_y)
        elif case == 12:  # 1-3
            add_line_segment(point1_x, point1_y, point3_x, point3_y)
        elif case == 13:  # 0-1
            add_line_segment(point0_x, point0_y, point1_x, point1_y)
        elif case == 14:  # 0-3
            add_line_segment(point0_x, point0_y, point3_x, point3_y)
        # Case 0 and 15 have no lines

@ti.func
def add_line_segment(x1, y1, x2, y2):
    # Add a line segment to our arrays
    idx = ti.atomic_add(lines_count[None], 1)
    if idx < max_lines:
        lines_begin[idx] = ti.Vector([x1, y1])
        lines_end[idx] = ti.Vector([x2, y2])

@ti.kernel
def render_field():
    # Render the scalar field to the pixel buffer
    for i, j in pixels:
        # Map pixel coordinates to field coordinates
        x = i / window_size
        y = j / window_size
        
        # Find the corresponding cell in the field
        field_x = x * n
        field_y = y * n
        
        # Get the field value using bilinear interpolation
        x0 = int(field_x)
        y0 = int(field_y)
        x1 = min(x0 + 1, n)
        y1 = min(y0 + 1, n)
        
        fx = field_x - x0
        fy = field_y - y0
        
        val = (1-fx)*(1-fy)*scalar_field[x0, y0] + \
              fx*(1-fy)*scalar_field[x1, y0] + \
              (1-fx)*fy*scalar_field[x0, y1] + \
              fx*fy*scalar_field[x1, y1]
        
        # Apply a colormap (simple blue-green-red)
        r = max(0.0, min(1.0, 1.0 - 2.0 * abs(val - 0.75)))
        g = max(0.0, min(1.0, 1.0 - 2.0 * abs(val - 0.5)))
        b = max(0.0, min(1.0, 1.0 - 2.0 * abs(val - 0.25)))
        
        pixels[i, j] = ti.Vector([r, g, b])

def visualize():
    # Initialize the scalar field
    initialize_field()
    
    iso_value = 0.5  # Initial iso-value
    
    # Create GUI
    gui = ti.GUI("Marching Squares", res=(window_size, window_size))
    
    while gui.running:
        # Handle keyboard input to change iso-value
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.UP:
                iso_value = min(1.0, iso_value + 0.05)
            elif gui.event.key == ti.GUI.DOWN:
                iso_value = max(0.0, iso_value - 0.05)
            elif gui.event.key == 'r':
                # Re-randomize the field
                initialize_field()
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # Render the field to pixels
        render_field()
        
        # Generate contours
        generate_contours(iso_value)
        
        # Display the scalar field
        gui.set_image(pixels)
        
        # Draw contour lines
        line_count = lines_count[None]
        if line_count > 0:
            begin_points = lines_begin.to_numpy()[:line_count]
            end_points = lines_end.to_numpy()[:line_count]
            
            for i in range(line_count):
                gui.line(begin_points[i], end_points[i], radius=1.5, color=0xFFFFFF)
        
        # Display text for iso-value
        gui.text(f"Iso-value: {iso_value:.2f} (Up/Down arrows to adjust, R to randomize)", (0.05, 0.95), color=0xFFFFFF)
        
        gui.show()

if __name__ == "__main__":
    visualize()
