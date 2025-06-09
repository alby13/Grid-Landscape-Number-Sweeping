import pygame
from pygame.locals import *
import OpenGL.GL as gl
import OpenGL.GLU as glu
import numpy as np
import random
import sys
import math

# --- Window and Performance Configuration ---
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FPS = 60

# --- Terrain Configuration ---
TERRAIN_WIDTH = 200
TERRAIN_DEPTH = 200
GRID_RESOLUTION = 30
MAX_MOUNTAIN_HEIGHT = 15

# --- Animation Configuration ---
SWEEP_DURATION = 3.0
PAUSE_DURATION = 1.0
FADE_TRANSITION_WIDTH = 30.0

SHAPE_CHANGE_INTERVAL = 2.5  # Generate a new shape every 2 seconds
MORPH_DURATION = 1.0         # Time for the morph animation to complete

class CombinedAnimation:
    def __init__(self):
        """Initializes Pygame, OpenGL, and all animation variables."""
        pygame.init()
        self.screen_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        pygame.display.set_mode(self.screen_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Morphing Terrain with Sweeping Numbers")

        self.clock = pygame.time.Clock()
        self.init_opengl()
        
        self.font_textures = self.init_font_rendering()

        # --- Sweep Animation State ---
        self.sweep_state = "SWEEP_FORWARD"
        self.sweep_progress = 0.0
        self.pause_timer = 0.0

        # --- Morph Animation State ---
        self.shape_change_timer = 0.0
        self.morph_progress = 1.0 # Start as 1.0 (finished)

        # --- Camera Control ---
        self.zoom = 120
        self.rotation_angle = 45

        # --- Terrain Geometry ---
        self.current_vertices = self.create_initial_terrain_vertices()
        self.start_vertices = self.current_vertices.copy()
        self.target_vertices = self.current_vertices.copy()
        self.height_values = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION), dtype=int)
        
        # --- Color array for per-vertex alpha ---
        grid_color = [0.2, 0.4, 0.6, 1.0]
        self.colors = np.tile(grid_color, (GRID_RESOLUTION * GRID_RESOLUTION, 1)).astype(np.float32)

        # --- OpenGL Buffers ---
        self.vbo = gl.glGenBuffers(1)
        self.color_vbo = gl.glGenBuffers(1)
        self.update_terrain_vbo()
        self.update_color_vbo()

        self.indices, self.num_indices = self.create_indices()
        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)
        
        xs = self.current_vertices[:, :, 0] + TERRAIN_WIDTH / 2
        zs = self.current_vertices[:, :, 2] + TERRAIN_DEPTH / 2
        self.vertex_distances = np.sqrt(xs**2 + zs**2)
        self.max_sweep_dist = self.vertex_distances.max()

    def init_opengl(self):
        gl.glViewport(0, 0, *self.screen_size)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45, self.screen_size[0] / self.screen_size[1], 0.1, 1000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0) #Background Color
        gl.glLineWidth(1.5)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def init_font_rendering(self):
        pygame.font.init()
        font = pygame.font.Font(pygame.font.get_default_font(), 32)
        textures = {}
        for char in "0123456789":
            s = font.render(char, True, (200, 220, 255), (0,0,0,0)).convert_alpha()
            d = pygame.image.tostring(s, "RGBA", True)
            w, h = s.get_size()
            tid = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, d)
            textures[char] = tid
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return textures

    def create_initial_terrain_vertices(self):
        v = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 3), dtype=np.float32)
        x = np.linspace(-TERRAIN_WIDTH / 2, TERRAIN_WIDTH / 2, GRID_RESOLUTION)
        z = np.linspace(-TERRAIN_DEPTH / 2, TERRAIN_DEPTH / 2, GRID_RESOLUTION)
        v[:, :, 0] = x[:, np.newaxis]
        v[:, :, 2] = z[np.newaxis, :]
        return v

    def create_indices(self):
        indices = []
        for z in range(GRID_RESOLUTION - 1):
            for x in range(GRID_RESOLUTION - 1):
                p1,p2,p3,p4 = z*GRID_RESOLUTION+x, z*GRID_RESOLUTION+(x+1), (z+1)*GRID_RESOLUTION+(x+1), (z+1)*GRID_RESOLUTION+x
                indices.extend([p1, p2, p3, p4])
        return np.array(indices, dtype=np.uint32), len(indices)

    def trigger_new_morph(self):
        """Sets up the variables to start a new morph animation."""
        self.start_vertices = self.current_vertices.copy()
        self.target_vertices = self.generate_new_mountain_shape()
        # IMPORTANT: Update the numbers to reflect the TARGET shape
        self.height_values = self.target_vertices[:, :, 1].astype(int)
        self.morph_progress = 0.0

    def generate_new_mountain_shape(self):
        """Calculates a new terrain heightmap and returns it."""
        new_vertices = self.create_initial_terrain_vertices()
        for _ in range(random.randint(5, 12)):
            px, pz = random.randint(0, GRID_RESOLUTION-1), random.randint(0, GRID_RESOLUTION-1)
            h = random.uniform(MAX_MOUNTAIN_HEIGHT * 0.4, MAX_MOUNTAIN_HEIGHT)
            r = random.uniform(GRID_RESOLUTION * 0.05, GRID_RESOLUTION * 0.2)
            x_idx, z_idx = np.arange(GRID_RESOLUTION)[:,None], np.arange(GRID_RESOLUTION)[None,:]
            d = np.sqrt((x_idx - px)**2 + (z_idx - pz)**2)
            mask = d < r
            falloff = 0.5 * (np.cos(d[mask] / r * math.pi) + 1)
            new_vertices[mask, 1] += h * falloff
        return new_vertices

    def update_terrain_vbo(self):
        vbo_data = self.current_vertices.reshape(-1, 3)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vbo_data.nbytes, vbo_data, gl.GL_DYNAMIC_DRAW)

    def update_color_vbo(self):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, gl.GL_DYNAMIC_DRAW)

    def update_grid_colors(self):
        current_sweep_dist = self.sweep_progress * self.max_sweep_dist
        delta_dist = current_sweep_dist - self.vertex_distances.flatten()
        fade_progress = np.clip(delta_dist / FADE_TRANSITION_WIDTH, 0.0, 1.0)
        self.colors[:, 3] = 1.0 - fade_progress
        self.update_color_vbo()

    def smooth_step(self, t):
        return t * t * (3.0 - 2.0 * t)

    def update(self):
        delta_time = self.clock.get_time() / 1000.0
        self.rotation_angle += delta_time * 5
        self.shape_change_timer += delta_time

        # --- Morphing Logic (runs independently) ---
        if self.shape_change_timer >= SHAPE_CHANGE_INTERVAL:
            self.trigger_new_morph()
            self.shape_change_timer = 0.0

        if self.morph_progress < 1.0:
            self.morph_progress = min(1.0, self.morph_progress + delta_time / MORPH_DURATION)
            sp = self.smooth_step(self.morph_progress)
            self.current_vertices[:,:,1] = (self.start_vertices[:,:,1] * (1-sp) + self.target_vertices[:,:,1] * sp)
            self.update_terrain_vbo()

        # --- Sweep Logic (runs independently) ---
        if self.sweep_state == "SWEEP_FORWARD":
            self.sweep_progress = min(1.0, self.sweep_progress + delta_time / SWEEP_DURATION)
            if self.sweep_progress == 1.0: self.sweep_state, self.pause_timer = "PAUSE_NUMBERS", 0.0
        elif self.sweep_state == "PAUSE_NUMBERS":
            self.pause_timer += delta_time
            if self.pause_timer >= PAUSE_DURATION: self.sweep_state = "SWEEP_BACKWARD"
        elif self.sweep_state == "SWEEP_BACKWARD":
            self.sweep_progress = max(0.0, self.sweep_progress - delta_time / SWEEP_DURATION)
            if self.sweep_progress == 0.0: self.sweep_state, self.pause_timer = "PAUSE_GRID", 0.0
        elif self.sweep_state == "PAUSE_GRID":
            self.pause_timer += delta_time
            if self.pause_timer >= PAUSE_DURATION: self.sweep_state = "SWEEP_FORWARD"
        
        self.update_grid_colors()

    def draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        
        cam_x, cam_z = math.sin(math.radians(self.rotation_angle)) * self.zoom, math.cos(math.radians(self.rotation_angle)) * self.zoom
        glu.gluLookAt(cam_x, self.zoom * 0.9, cam_z, 0, 0, 0, 0, 1, 0)

        # Draw Grid
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_vbo)
        gl.glColorPointer(4, gl.GL_FLOAT, 0, None)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glDrawElements(gl.GL_QUADS, self.num_indices, gl.GL_UNSIGNED_INT, None)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        self.draw_height_numbers()
        pygame.display.flip()

    def draw_height_numbers(self):
        modelview = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)
        cam_r = np.array([modelview[0][0], modelview[1][0], modelview[2][0]])
        cam_u = np.array([modelview[0][1], modelview[1][1], modelview[2][1]])
        
        gl.glEnable(gl.GL_TEXTURE_2D)
        current_sweep_dist = self.sweep_progress * self.max_sweep_dist

        for z in range(GRID_RESOLUTION):
            for x in range(GRID_RESOLUTION):
                dist = self.vertex_distances[z, x]
                if dist < current_sweep_dist:
                    v = self.current_vertices[z, x]
                    h_str = str(self.height_values[z, x])
                    
                    delta = current_sweep_dist - dist
                    alpha = min(1.0, delta / FADE_TRANSITION_WIDTH)
                    gl.glColor4f(1.0, 1.0, 1.0, alpha)

                    char_size = 2.0
                    start_pos = np.array(v) - (cam_r * (len(h_str) * char_size / 2 - char_size / 2))

                    for i, char in enumerate(h_str):
                        if char in self.font_textures:
                            pos = start_pos + cam_r * (i * char_size)
                            p1, p2 = pos - cam_r*0.5*char_size, pos + cam_r*0.5*char_size
                            gl.glBindTexture(gl.GL_TEXTURE_2D, self.font_textures[char])
                            gl.glBegin(gl.GL_QUADS)
                            gl.glTexCoord2f(0,0); gl.glVertex3fv(p1 + cam_u*0.5*char_size)
                            gl.glTexCoord2f(1,0); gl.glVertex3fv(p2 + cam_u*0.5*char_size)
                            gl.glTexCoord2f(1,1); gl.glVertex3fv(p2 - cam_u*0.5*char_size)
                            gl.glTexCoord2f(0,1); gl.glVertex3fv(p1 - cam_u*0.5*char_size)
                            gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

    def run(self):
        print("--- Morphing Terrain with Sweeping Numbers ---")
        print("Controls: Mouse Wheel (Zoom), ESC (Exit)")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit(), sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4: self.zoom = max(20, self.zoom - 10)
                    elif event.button == 5: self.zoom += 10
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == '__main__':
    app = CombinedAnimation()
    app.run()
