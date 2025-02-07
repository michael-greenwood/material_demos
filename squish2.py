import pygame
import numpy as np
import time
import random

# Initialize Pygame
pygame.init()

# Get screen resolution
info = pygame.display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h

# Fixed logical grid size (square)
GRID_SIZE = 400  # Logical resolution of the grid
PIXEL_SIZE = 4  # Each logical pixel is a 10x10 block
GRID_WIDTH = GRID_SIZE // PIXEL_SIZE
GRID_HEIGHT = GRID_SIZE // PIXEL_SIZE

# Fullscreen display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Volume-Conserving Squish Simulation")

# Create a pixel array (stores seed ID, -1 means empty)
pixel_map = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)

# Seed storage (maps seed ID to color index 0-45)
seed_list = []

# Growth interval (tunable)
GROWTH_INTERVAL = .1  
FINAL_SQUISH = 0.5
# Squish effect variables
is_squishing = False
squish_factor = 1.0  # Starts at normal size (1.0 means no squish)
squish_complete = False  # Flag to stop squishing but keep running

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

def grow_pixels():
    """Expands all seeded pixels using NumPy operations."""
    global pixel_map, is_squishing, fixed_pixel_map

    if is_squishing:
        return  # Stop growth if squishing

    new_map = pixel_map.copy()
    seeded_pixels = np.argwhere(pixel_map >= 0)

    for y, x in seeded_pixels:
        seed_id = pixel_map[y, x]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and pixel_map[ny, nx] == -1:
                new_map[ny, nx] = seed_id

    pixel_map = new_map

    # Once fully grown, store it as fixed_pixel_map
    if np.all(pixel_map >= 0):
        print("All pixels occupied! Storing fixed map and starting squish...")
        fixed_pixel_map = pixel_map.copy()  # Store final map
        start_squish()


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
    """Calculates and displays the compressive strength using the Hall-Petch relationship."""
    sigma_0 = 100  # Base strength (MPa)
    k = 30  # Hall-Petch coefficient (MPa·mm^0.5)

    num_seeds = len(seed_list)
    if num_seeds == 0:
        strength = sigma_0  # Default strength when no seeds are present
    else:
        avg_grain_size = GRID_SIZE / num_seeds  # Estimate grain size
        strength = sigma_0 + k * (avg_grain_size ** -0.5)

    # Render text to display on screen
    font = pygame.font.Font(None, 36)  # Default font, size 36
    text = font.render(f"Compressive Strength: {strength:.2f} MPa", True, (255, 255, 255))

    # Position the text at the top-left corner
    screen.blit(text, (50, 50))


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

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False  # Allow ESC key to exit
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not is_squishing:
                mx, my = event.pos
                scale_factor = min(SCREEN_WIDTH / GRID_SIZE, SCREEN_HEIGHT / GRID_SIZE)
                grid_x = int((mx - offset_x) / scale_factor) // PIXEL_SIZE
                grid_y = int((my - offset_y) / scale_factor) // PIXEL_SIZE

                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT and pixel_map[grid_y, grid_x] == -1:
                    new_seed_id = len(seed_list)
                    new_seed_value = random.randint(0, 45)
                    seed_list.append(new_seed_value)
                    pixel_map[grid_y, grid_x] = new_seed_id
    
    if not is_squishing and time.time() - last_update_time > GROWTH_INTERVAL:
        grow_pixels()
        last_update_time = time.time()
    
    if is_squishing and not squish_complete and time.time() - last_update_time > GROWTH_INTERVAL:
        apply_squish()
        last_update_time = time.time()
    
    if squish_complete:
        display_compressive_strength()

    pygame.display.flip()

pygame.quit()
