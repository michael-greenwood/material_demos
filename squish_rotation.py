import pygame
import numpy as np
import time
import random
import scipy.ndimage

# Initialize Pygame
pygame.init()

# Get screen resolution
info = pygame.display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h

# Fixed logical grid size (square)
GRID_SIZE = 400  # Logical resolution of the grid
PIXEL_SIZE = 1  # Each logical pixel is a 10x10 block
GRID_WIDTH = GRID_SIZE // PIXEL_SIZE
GRID_HEIGHT = GRID_SIZE // PIXEL_SIZE

# Fullscreen display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Volume-Conserving Squish Simulation")

# Create a pixel array (stores seed ID, -1 means empty)
pixel_map = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int16)

# Seed storage (maps seed ID to color index 0-45)
seed_list = []

# Growth interval (tunable)
GROWTH_INTERVAL = .05  
FINAL_SQUISH = 0.5
# Squish effect variables
is_squishing = False
squish_factor = 1.0  # Starts at normal size (1.0 means no squish)
squish_complete = False  # Flag to stop squishing but keep running
seeding_started = False

def generate_color(value):
    """Convert a value (0-45) to a color gradient from red to blue."""
    r = int(255 * (1 - value / 45))  # Red decreases
    g = 0
    b = int(255 * (value / 45))      # Blue increases
    return (r, g, b)

def update_surface():
    """Convert pixel_map (seed IDs) to a Pygame surface (color gradient, white background)."""
    global squish_factor

    rgb_map = np.full((GRID_HEIGHT, GRID_WIDTH, 3), 255, dtype=np.uint8)  # Default to white
    
    # Apply colors using NumPy masking
    assigned_pixels = pixel_map >= 0
    if assigned_pixels.any():
        colors = np.array([generate_color(seed_list[seed_id]) for seed_id in pixel_map[assigned_pixels]])
        rgb_map[assigned_pixels] = colors

    # Convert NumPy array to surface
    surface = pygame.surfarray.make_surface(np.transpose(rgb_map, (1, 0, 2)))

    # Compute scaling factor to maintain aspect ratio
    scale_factor = min(SCREEN_WIDTH / GRID_SIZE, SCREEN_HEIGHT / GRID_SIZE)
    scaled_size = (int(GRID_SIZE * scale_factor), int(GRID_SIZE * scale_factor))

    # Scale the surface
    scaled_surface = pygame.transform.scale(surface, scaled_size)

    return scaled_surface, scaled_size

growth_front = set()  # Stores (y, x) pixels that can still grow


# Precompute a large table of random numbers for fast access
RANDOM_TABLE_SIZE = 100000
random_table = np.random.rand(RANDOM_TABLE_SIZE, 8)  # 8 neighbors per growth step
random_index_counter = 0  # Tracks current index in random table

# Probability stencils mapped to seed IDs
seed_probabilities = {}



def compute_probability_stencil(color_value):
    """Precomputes a rotated probability stencil for a given seed's anisotropy."""
    
    # Convert color to rotation angle (negative for counterclockwise rotation)
    rotation_angle = -color_value  

    # Define base 3x3 probability stencil
    base_stencil = np.array([
        [0.0, 1.0, 0.0],  # Top row (Up, Left-Right)
        [1.0, 0.0, 1.0],  # Middle row (Left, Center, Right)
        [0.0, 1.0, 0.0]   # Bottom row (Down, Left-Right)
    ])

    # Rotate the stencil using scipy.ndimage.rotate
    rotated_stencil = scipy.ndimage.rotate(base_stencil, rotation_angle, reshape=False, order=1, mode='nearest')

    # Extract 8-neighbor probabilities (ignoring the center)
    probabilities = np.array([
        rotated_stencil[0, 1],  # Up
        rotated_stencil[2, 1],  # Down
        rotated_stencil[1, 0],  # Left
        rotated_stencil[1, 2],  # Right
        rotated_stencil[0, 0],  # Top-left
        rotated_stencil[0, 2],  # Top-right
        rotated_stencil[2, 0],  # Bottom-left
        rotated_stencil[2, 2]   # Bottom-right
    ])

    print(f"Color Value: {color_value}, Probabilities: {probabilities}")

    return probabilities





def grow_pixels():
    """Expands pixels using a Monte Carlo anisotropic approach with precomputed probability stencils."""
    global pixel_map, is_squishing, fixed_pixel_map, growth_front, random_index_counter

    if is_squishing:
        return  # Stop growth if squishing

    new_growth = set()

    for y, x in list(growth_front):
        seed_id = pixel_map[y, x]

        # Use precomputed probability stencil
        probabilities = seed_probabilities[seed_id]

        # Fetch precomputed random numbers (8 values per cell)
        random_values = random_table[random_index_counter]
        random_index_counter = (random_index_counter + 1) % RANDOM_TABLE_SIZE  # Cycle through table

        directions = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1),   # Right
            (-1, -1), # Top-left
            (-1, 1),  # Top-right
            (1, -1),  # Bottom-left
            (1, 1)    # Bottom-right
        ]

        for (dy, dx), prob, rand_val in zip(directions, probabilities, random_values):
            if rand_val < prob:  # Grow if probability threshold is met
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and pixel_map[ny, nx] == -1:
                    pixel_map[ny, nx] = seed_id
                    new_growth.add((ny, nx))

        # Remove pixels that can no longer grow
        if not new_growth:
            growth_front.remove((y, x))

    growth_front.update(new_growth)

    # Check if growth is complete
    if not growth_front and seeding_started:
        print("Solidification completed! Starting compression.")
        fixed_pixel_map = pixel_map.copy()
        start_squish()

def add_new_seed(grid_x, grid_y):
    """Adds a new seed and stores its probability stencil."""
    global seeding_started

    new_seed_id = len(seed_list)
    new_seed_value = random.randint(0, 45)
    seed_list.append(new_seed_value)
    pixel_map[grid_y, grid_x] = new_seed_id
    growth_front.add((grid_y, grid_x))  # Track new seed in growth front
    seeding_started = True

    # Precompute and store probability stencil for this seed
    seed_probabilities[new_seed_id] = compute_probability_stencil(new_seed_value)

def start_squish():
    """Initiates the squish effect."""
    global is_squishing
    is_squishing = True

def apply_squish():
    """Gradually compresses the pixel map from the top down while ensuring smooth progression."""
    global squish_factor, squish_complete, pixel_map, fixed_pixel_map
    squish_factor -= 0.01  # Controlled squish step

    if squish_factor <= FINAL_SQUISH:
        squish_factor = FINAL_SQUISH
        squish_complete = True
        print("Squish complete!")

    # Create a compressed version of the original (fixed) pixel map
    compressed_map = np.full_like(fixed_pixel_map, -1)  # Empty grid
    row_densities = np.sum(fixed_pixel_map >= 0, axis=1)  # Count pixels per row

    # Map old rows to new squished positions
    #new_positions = np.linspace(0, GRID_HEIGHT * squish_factor, GRID_HEIGHT).astype(int)
    new_positions = np.linspace((GRID_HEIGHT - 1)*(1-squish_factor), (GRID_HEIGHT - 1) , GRID_HEIGHT).astype(int)

    for old_row in range(GRID_HEIGHT):
        new_row = new_positions[old_row]
        if row_densities[old_row] > 0 and new_row < GRID_HEIGHT:
            compressed_map[new_row] = fixed_pixel_map[old_row]  # Move pixels downward

    pixel_map[:] = compressed_map  # Update only pixel_map, keeping fixed_pixel_map intact

def display_compressive_strength():
    """Calculates and displays the compressive strength using the Hall-Petch relationship and a bar chart."""
    if not squish_complete:
        return  # Only display after squishing is complete

    sigma_0 = 100  # Base strength (MPa)
    k = 3000  # Hall-Petch coefficient (MPa·mm^0.5)

    num_seeds = len(seed_list)
    if num_seeds == 0:
        strength = sigma_0  # Default strength when no seeds are present
    else:
        avg_grain_size = 10000 / num_seeds  # Estimate grain size
        strength = sigma_0 + k * (avg_grain_size ** -0.5)

    # Define categories and colors
    categories = [
        (175, "Weak", (255, 0, 0)),        # Red
        (350, "OK", (255, 165, 0)),        # Orange
        (500, "Great", (255, 255, 0)),     # Yellow
        (800, "Impressive", (0, 255, 0)),  # Green
    ]

    # Determine category
    for threshold, label, color in categories:
        if strength < threshold:
            break

    # Render text
    font = pygame.font.Font(None, 36)
    text = font.render(f"Compressive Strength: {strength:.2f} MPa", True, (255, 255, 255))
    screen.blit(text, (50, 50))

    # Draw the bar chart
    bar_x, bar_y = 50, 100  # Bar position
    bar_width = 300
    bar_height = 30

    # Normalize the strength for bar length (max 800 MPa)
    normalized_strength = min(strength, 800) / 800  
    filled_width = int(bar_width * normalized_strength)

    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))  # Background bar (grey)
    pygame.draw.rect(screen, color, (bar_x, bar_y, filled_width, bar_height))  # Filled portion

    # Render category label
    label_text = font.render(label, True, color)
    screen.blit(label_text, (bar_x + bar_width + 20, bar_y))

def reset_simulation():
    """Resets the simulation to its initial state."""
    global pixel_map, seed_list, is_squishing, squish_factor, squish_complete, fixed_pixel_map, growth_front, seeding_started

    pixel_map = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int16)
    seed_list.clear()
    growth_front.clear()  # Reset tracked growth pixels
    is_squishing = False
    squish_factor = 1.0
    squish_complete = False
    fixed_pixel_map = None
    seeding_started = False
    print("Simulation Reset!")


def draw_reset_button():
    """Draws a reset button on the screen."""
    button_width, button_height = 150, 50
    button_x, button_y = SCREEN_WIDTH - button_width - 50, SCREEN_HEIGHT - button_height - 50  # Bottom-right corner
    button_color = (200, 200, 200)  # Light grey
    text_color = (0, 0, 0)  # Black

    # Draw the button rectangle
    pygame.draw.rect(screen, button_color, (button_x, button_y, button_width, button_height))

    # Draw text inside the button
    font = pygame.font.Font(None, 36)
    text = font.render("Reset", True, text_color)
    text_rect = text.get_rect(center=(button_x + button_width // 2, button_y + button_height // 2))
    screen.blit(text, text_rect)

    return button_x, button_y, button_width, button_height


running = True
last_update_time = time.time()

while running:
    screen.fill((0, 0, 0))  # Black background
    scaled_surface, scaled_size = update_surface()
    offset_x = (SCREEN_WIDTH - scaled_size[0]) // 2
    offset_y = (SCREEN_HEIGHT - scaled_size[1]) // 2
    screen.blit(scaled_surface, (offset_x, offset_y))
    # Draw the black squish bar at the top
    # Draw the black squish bar at the top
    if is_squishing or squish_complete:
        bar_height = int((1 - squish_factor) * scaled_size[1])  # Growing black bar
        squish_height = int((1 - squish_factor) * scaled_size[1])   # Sync with actual compression
        #pygame.draw.rect(screen, (0, 0, 0), (offset_x, offset_y, scaled_size[0], squish_height))
        pygame.draw.rect(screen, (40, 40, 40), (offset_x - 50, offset_y + squish_height - 50, scaled_size[0] + 100, 50))
        pygame.draw.rect(screen, (40, 40, 40), (
            offset_x + (scaled_size[0] // 2) - 50,  # Centered X-position (100px wide)
            offset_y,  # Starts from the top
            100,  # Width of 100 pixels
            squish_height  # Extends down to squish height
            ))

    # Check if the Reset button was clicked
    button_x, button_y, button_width, button_height = draw_reset_button()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False  # Allow ESC key to exit
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            if button_x <= mx <= button_x + button_width and button_y <= my <= button_y + button_height:
                reset_simulation()
                continue  # Skip further processing if reset was clicked

            # If the simulation is NOT in the squish phase, allow seed placement
            if not is_squishing:
                mx, my = event.pos
                scale_factor = min(SCREEN_WIDTH / GRID_SIZE, SCREEN_HEIGHT / GRID_SIZE)
                grid_x = int((mx - offset_x) / scale_factor) // PIXEL_SIZE
                grid_y = int((my - offset_y) / scale_factor) // PIXEL_SIZE

                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT and pixel_map[grid_y, grid_x] == -1:
                    add_new_seed(grid_x, grid_y)
                    
    
    if not is_squishing and time.time() - last_update_time > GROWTH_INTERVAL:
        grow_pixels()
        last_update_time = time.time()
    
    if is_squishing and not squish_complete and time.time() - last_update_time > GROWTH_INTERVAL:
        apply_squish()
        last_update_time = time.time()
    
    if squish_complete:
        display_compressive_strength()
    # Draw Reset button
    draw_reset_button()
    pygame.display.flip()

pygame.quit()
