import threading, time, queue
import numpy as np
from scipy.ndimage import convolve, label, binary_erosion, binary_dilation, gaussian_filter
from skimage.measure import marching_cubes
import colorsys
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict, deque

from ursina import (
    Ursina, window, color, Entity, Mesh, EditorCamera, application,
    camera, Slider, Text, Button, destroy, mouse, Vec3, raycast,
    invoke
)
from ursina.shaders import lit_with_shadows_shader
from ursina.lights import DirectionalLight, AmbientLight

# ── FractalAgent class - represents a detected stable structure ───────────
class FractalAgent:
    """A detected stable structure that can potentially evolve its own physics"""
    def __init__(self, id, mask, position, volume, age=0):
        self.id = id                # Unique identifier
        self.mask = mask.copy()     # Binary mask of voxels
        self.position = position    # Center of mass
        self.volume = volume        # Number of voxels
        self.age = age              # How many frames it has existed
        self.color = color.random_color()  # Assign a unique color
        self.sub_sim = None         # Will hold a nested simulation if promoted
        
    def update(self, new_mask=None, new_position=None):
        """Update the agent with new data from detection"""
        if new_mask is not None:
            self.mask = new_mask.copy()
        if new_position is not None:
            self.position = new_position
        self.age += 1
        
    def promote_to_sub_simulation(self, parent_grid, local_N=32, min_volume=25):
        """Create a nested simulation for this stable pattern"""
        from copy import deepcopy
        # Only create sub-simulation once the structure is stable enough
        # Added volume threshold check to avoid creating subs for tiny patterns
        if self.age >= 20 and self.volume > min_volume and self.sub_sim is None:
            print(f"Promoting agent {self.id} to a sub-simulation!")
            # Create a downsampled sub-simulation
            self.sub_sim = MiniWoW(N=local_N, topology='none')
            
            # Extract the field values within this agent's mask
            field_values = np.where(self.mask, parent_grid, 0)
            
            # Downsample to initialize the sub_sim
            # Simple method: just take a centered cube that encompasses the structure
            x, y, z = self.position
            half_size = int(local_N / 2)
            x_min, x_max = max(0, int(x) - half_size), min(parent_grid.shape[0], int(x) + half_size)
            y_min, y_max = max(0, int(y) - half_size), min(parent_grid.shape[1], int(y) + half_size)
            z_min, z_max = max(0, int(z) - half_size), min(parent_grid.shape[2], int(z) + half_size)
            
            # Extract and resize
            extract = field_values[x_min:x_max, y_min:y_max, z_min:z_max]
            
            # Handle if we extracted a region smaller than expected
            if extract.shape[0] < local_N or extract.shape[1] < local_N or extract.shape[2] < local_N:
                padded = np.zeros((local_N, local_N, local_N), dtype=np.float32)
                padded[:min(local_N, extract.shape[0]), 
                       :min(local_N, extract.shape[1]), 
                       :min(local_N, extract.shape[2])] = extract[:min(local_N, extract.shape[0]), 
                                                                  :min(local_N, extract.shape[1]), 
                                                                  :min(local_N, extract.shape[2])]
                self.sub_sim.phi = padded
            else:
                # Resize to fit in the sub-simulation grid
                from scipy.ndimage import zoom
                factors = (local_N / extract.shape[0], local_N / extract.shape[1], local_N / extract.shape[2])
                self.sub_sim.phi = zoom(extract, factors, order=1)
            
            # Copy the previous state to avoid immediate collapse
            self.sub_sim.phi_o = self.sub_sim.phi.copy()
            
            # Slightly randomize parameters to encourage diversity
            self.sub_sim.tension = np.random.uniform(0.8, 1.2) * 5.0
            self.sub_sim.pot_lin = np.random.uniform(0.8, 1.2) * 1.0
            self.sub_sim.pot_cub = np.random.uniform(0.8, 1.2) * 0.2
            
            return True
        return False
        
    def step_sub_simulation(self, parent_grid, coupling=0.1):
        """Evolve the sub-simulation and couple back to parent grid"""
        if self.sub_sim is not None:
            # Step the sub-simulation forward
            self.sub_sim.step(2)
            
            # Couple back to parent grid
            # This is a simplified coupling - in a full implementation,
            # you would need a more sophisticated up/down-sampling approach
            x, y, z = self.position
            N = self.sub_sim.N
            half_size = int(N / 2)
            
            # Define the region in the parent grid we'll update
            x_min, x_max = max(0, int(x) - half_size), min(parent_grid.shape[0], int(x) + half_size)
            y_min, y_max = max(0, int(y) - half_size), min(parent_grid.shape[1], int(y) + half_size)
            z_min, z_max = max(0, int(z) - half_size), min(parent_grid.shape[2], int(z) + half_size)
            
            # Handle size differences
            sub_x_max = min(N, x_max - x_min + half_size)
            sub_y_max = min(N, y_max - y_min + half_size)
            sub_z_max = min(N, z_max - z_min + half_size)
            
            # Only update within the agent's mask
            region_mask = self.mask[x_min:x_max, y_min:y_max, z_min:z_max]
            if region_mask.size > 0:
                # Extract the relevant portion of the sub-simulation
                sub_field = self.sub_sim.phi[:sub_x_max, :sub_y_max, :sub_z_max]
                
                # Only update where we have mask and valid sub-field dimensions
                update_slice = parent_grid[x_min:x_max, y_min:y_max, z_min:z_max]
                min_x = min(region_mask.shape[0], sub_field.shape[0], update_slice.shape[0])
                min_y = min(region_mask.shape[1], sub_field.shape[1], update_slice.shape[1])
                min_z = min(region_mask.shape[2], sub_field.shape[2], update_slice.shape[2])
                
                # Apply coupling
                if min_x > 0 and min_y > 0 and min_z > 0:
                    mask_slice = region_mask[:min_x, :min_y, :min_z]
                    parent_grid[x_min:x_min+min_x, y_min:y_min+min_y, z_min:z_min+min_z] = \
                        np.where(
                            mask_slice,
                            (1-coupling) * parent_grid[x_min:x_min+min_x, y_min:y_min+min_y, z_min:z_min+min_z] + 
                            coupling * sub_field[:min_x, :min_y, :min_z],
                            parent_grid[x_min:x_min+min_x, y_min:y_min+min_y, z_min:z_min+min_z]
                        )

# ── mini solver ───────────────────────────────────────────
class MiniWoW:
    def __init__(self, N=64, dt=0.1, damping=0.001,
                 tension=5., pot_lin=1., pot_cub=0.2,
                 topology='none', track_fractals=True):  # Enable fractal tracking by default
        self.N, self.dt, self.damp = N, dt, damping
        self.tension, self.pot_lin, self.pot_cub = tension, pot_lin, pot_cub
        self.topology = topology
        self.track_fractals = track_fractals

        self.lock = threading.Lock()
        self.phi   = np.zeros((N, N, N), np.float32)
        self.phi_o = np.zeros_like(self.phi)

        # Energy tracking
        self.energy_history = deque(maxlen=600)  # 10 seconds at 60fps
        self.total_energy = 0.0

        # Fractal tracking fields
        self.next_agent_id = 1
        self.agents = {}  # Dictionary of tracked fractal agents
        self.fractal_mask = np.zeros((N, N, N), dtype=bool)  # Current binary mask
        self.last_detection_time = 0  # Time of last pattern detection
        
        # Only initialize field if topology is specified
        if topology != 'none':
            self.init_field()
        
        # 6-point Laplacian kernel
        self.kern = np.zeros((3,3,3), np.float32)
        self.kern[1,1,1] = -6
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]:
            self.kern[dx,dy,dz] = 1
            
    def reset(self):
        """Reset the simulation field to the current topology's initial state"""
        with self.lock:
            if self.topology != 'none':
                self.init_field()
                self.phi_o = self.phi.copy()
                # Reset fractal tracking
                self.next_agent_id = 1
                self.agents = {}
                self.fractal_mask = np.zeros((self.N, self.N, self.N), dtype=bool)
                # Reset energy history
                self.energy_history.clear()
            
    def resize_grid(self, new_N):
        """Resize the simulation grid"""
        with self.lock:
            old_topology = self.topology
            # Set topology to none during resize to prevent automatic initialization
            self.topology = 'none'
            self.N = new_N
            self.phi = np.zeros((new_N, new_N, new_N), np.float32)
            self.phi_o = np.zeros_like(self.phi)
            # Reset fractal tracking for new size
            self.fractal_mask = np.zeros((new_N, new_N, new_N), dtype=bool)
            self.next_agent_id = 1
            self.agents = {}
            # Restore topology and initialize
            self.topology = old_topology
            if self.topology != 'none':
                self.init_field()
                self.phi_o = self.phi.copy()
            # Reset energy history    
            self.energy_history.clear()

    def init_field(self):
        N = self.N
        # Create coordinate grids
        x = np.arange(N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        c = N // 2  # center
        r = N // 6  # characteristic radius
        
        if self.topology == 'box':
            # Standard box - gaussian pulse
            r2 = r**2
            self.phi[:] = 2 * np.exp(-((X-c)**2 + (Y-c)**2 + (Z-c)**2) / (2 * r2))
        
        elif self.topology == 'sphere':
            # Spherical topology - shell-like initial condition
            r_shell = N / 3  # Radius of the shell
            thickness = N / 10  # Thickness of the shell
            R2 = (X-c)**2 + (Y-c)**2 + (Z-c)**2
            self.phi[:] = 2 * np.exp(-(np.sqrt(R2) - r_shell)**2 / (2 * thickness**2))
        
        elif self.topology == 'torus':
            # Toroidal topology
            R_major = N / 3  # Major radius of the torus
            R_minor = N / 8  # Minor radius of the torus
            
            # Distance to the circular axis of the torus
            d_circle = np.sqrt((np.sqrt((X-c)**2 + (Y-c)**2) - R_major)**2 + (Z-c)**2)
            self.phi[:] = 2 * np.exp(-(d_circle)**2 / (2 * R_minor**2))
        
        elif self.topology == 'wave':
            # Standing wave pattern
            k = 2 * np.pi / (N / 4)  # Wave number
            self.phi[:] = np.sin(k * (X-c)) * np.sin(k * (Y-c)) * np.sin(k * (Z-c))
        
        elif self.topology == 'random':
            # Random field with some smoothing
            self.phi[:] = np.random.randn(N, N, N) * 0.5
            # Apply some smoothing to avoid too sharp transitions
            self.phi = gaussian_filter(self.phi, sigma=1.0)
            
        elif self.topology == 'custom':
            # Load custom field if it exists, otherwise use default gaussian
            if hasattr(self, 'custom_field') and self.custom_field is not None:
                # Resize custom field if needed
                if self.custom_field.shape[0] != N:
                    from scipy.ndimage import zoom
                    zoom_factor = N / self.custom_field.shape[0]
                    self.phi[:] = zoom(self.custom_field, zoom_factor, order=1)
                else:
                    self.phi[:] = self.custom_field.copy()
            else:
                # Default to gaussian if no custom field
                r2 = r**2
                self.phi[:] = 2 * np.exp(-((X-c)**2 + (Y-c)**2 + (Z-c)**2) / (2 * r2))

    def field_energy(self):
        """Calculate the field energy distribution"""
        # Calculate the gradient components
        dx, dy, dz = np.gradient(self.phi)
        
        # Gradient energy term: 0.5 * |∇Ψ|²
        grad_energy = 0.5 * (dx**2 + dy**2 + dz**2)
        
        # Potential energy term: V(Ψ)
        # For the double-well potential: V(Ψ) = -pot_lin * Ψ + pot_cub * Ψ³
        potential_energy = -self.pot_lin * self.phi + self.pot_cub * self.phi**3
        potential_energy = potential_energy**2  # Ensure positive energy
        
        # Total energy = gradient energy + potential energy
        total_energy = grad_energy + potential_energy
        
        return total_energy
        
    def calculate_total_energy(self):
        """Calculate the total energy of the field"""
        energy_density = self.field_energy()
        total = np.sum(energy_density)
        self.total_energy = total
        self.energy_history.append(total)
        return total

    def detect_stable_patterns(self, iso_threshold=1.0, min_volume=10, max_volume=None, bond_threshold=5.0, color_threshold=0.1):
        """Identify stable patterns (isosurfaces) in the field"""
        if not self.track_fractals:
            return
            
        # Only run detection periodically to save resources
        current_time = time.time()
        if current_time - self.last_detection_time < 0.5:  # Run detection every 0.5 seconds
            return
        self.last_detection_time = current_time
        
        # Create binary mask of values above threshold
        binary_mask = (self.phi > iso_threshold)
        
        # Clean up the mask - remove small holes and smooth edges
        binary_mask = binary_erosion(binary_mask, iterations=1)
        binary_mask = binary_dilation(binary_mask, iterations=1)
        
        # Label connected components
        labeled_mask, num_features = label(binary_mask)
        
        # Process each feature and track it
        active_agent_ids = set()
        for i in range(1, num_features + 1):
            component_mask = (labeled_mask == i)
            volume = np.sum(component_mask)
            
            # Skip components that are too small or too large
            if volume < min_volume:
                continue
            if max_volume is not None and volume > max_volume:
                continue
                
            # Get centroid (center of mass)
            coords = np.where(component_mask)
            position = (np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2]))
            
            # Try to match with existing agents based on position
            matched = False
            closest_agent_id = None
            min_distance = float('inf')
            
            for agent_id, agent in self.agents.items():
                dist = np.sqrt((agent.position[0] - position[0])**2 + 
                               (agent.position[1] - position[1])**2 + 
                               (agent.position[2] - position[2])**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_agent_id = agent_id
            
            # If close enough to an existing agent, update it
            if closest_agent_id is not None and min_distance < 10:  # Adjust threshold as needed
                self.agents[closest_agent_id].update(component_mask, position)
                active_agent_ids.add(closest_agent_id)
                matched = True
            
            # Otherwise, create a new agent
            if not matched:
                new_id = self.next_agent_id
                self.next_agent_id += 1
                self.agents[new_id] = FractalAgent(new_id, component_mask, position, volume)
                active_agent_ids.add(new_id)
        
        # Check for bonds between agents (new chemistry feature)
        to_merge = []
        for agent_id1, agent1 in self.agents.items():
            if agent_id1 not in active_agent_ids:
                continue
                
            for agent_id2, agent2 in self.agents.items():
                if agent_id1 >= agent_id2 or agent_id2 not in active_agent_ids:
                    continue
                    
                # Calculate distance between centroids
                dist = np.sqrt((agent1.position[0] - agent2.position[0])**2 + 
                               (agent1.position[1] - agent2.position[1])**2 + 
                               (agent1.position[2] - agent2.position[2])**2)
                
                # Get color difference 
                hue1 = colorsys.rgb_to_hsv(agent1.color.r, agent1.color.g, agent1.color.b)[0]
                hue2 = colorsys.rgb_to_hsv(agent2.color.r, agent2.color.g, agent2.color.b)[0]
                hue_diff = min(abs(hue1 - hue2), 1 - abs(hue1 - hue2))  # Account for circular nature of hue
                
                # If close enough and similar color, mark for merging
                if dist < bond_threshold and hue_diff < color_threshold:
                    to_merge.append((agent_id1, agent_id2))
                    
        # Process merges
        for id1, id2 in to_merge:
            if id1 in self.agents and id2 in self.agents:  # Ensure both still exist
                # Merge masks
                merged_mask = np.logical_or(self.agents[id1].mask, self.agents[id2].mask)
                
                # Calculate new position (weighted average)
                vol1, vol2 = self.agents[id1].volume, self.agents[id2].volume
                total_vol = vol1 + vol2
                pos1, pos2 = self.agents[id1].position, self.agents[id2].position
                new_pos = ((pos1[0]*vol1 + pos2[0]*vol2)/total_vol,
                          (pos1[1]*vol1 + pos2[1]*vol2)/total_vol,
                          (pos1[2]*vol1 + pos2[2]*vol2)/total_vol)
                          
                # Keep the older agent, update it with merged properties
                target_id = id1 if self.agents[id1].age >= self.agents[id2].age else id2
                other_id = id2 if target_id == id1 else id1
                
                self.agents[target_id].update(merged_mask, new_pos)
                self.agents[target_id].volume = total_vol
                
                # Remove the other agent
                del self.agents[other_id]
                active_agent_ids.discard(other_id)
        
        # Remove agents that weren't matched in this frame
        to_remove = []
        for agent_id in self.agents:
            if agent_id not in active_agent_ids:
                to_remove.append(agent_id)
        
        for agent_id in to_remove:
            # Only remove if it's been missing for a few frames
            self.agents[agent_id].age -= 2  # Decrease age faster when not detected
            if self.agents[agent_id].age <= 0:
                del self.agents[agent_id]
            
        # Update the global fractal mask for visualization
        self.fractal_mask = np.zeros_like(binary_mask)
        for agent in self.agents.values():
            self.fractal_mask = np.logical_or(self.fractal_mask, agent.mask)
        
        # Promote stable agents to have their own simulations
        for agent in list(self.agents.values()):
            if agent.age > 20 and agent.volume > 25 and agent.sub_sim is None:
                agent.promote_to_sub_simulation(self.phi)

    def apply_poke(self, x, y, z, radius=3, amplitude=1.0, sigma=2.0):
        """Apply a gaussian 'poke' to the field at the specified position"""
        with self.lock:
            N = self.N
            # Ensure coordinates are valid
            x, y, z = int(x), int(y), int(z)
            if x < 0 or x >= N or y < 0 or y >= N or z < 0 or z >= N:
                return False
                
            # Define box bounds
            x_min, x_max = max(0, x-radius), min(N-1, x+radius)
            y_min, y_max = max(0, y-radius), min(N-1, y+radius)
            z_min, z_max = max(0, z-radius), min(N-1, z+radius)
            
            # Create coordinate grids for the box
            xx, yy, zz = np.meshgrid(
                np.arange(x_min, x_max+1),
                np.arange(y_min, y_max+1),
                np.arange(z_min, z_max+1),
                indexing='ij'
            )
            
            # Calculate squared distance from center
            dist_sq = (xx-x)**2 + (yy-y)**2 + (zz-z)**2
            
            # Apply gaussian pulse
            self.phi[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] += \
                amplitude * np.exp(-dist_sq / (2 * sigma**2))
                
            return True


    def step(self, n_steps=1):
        for _ in range(n_steps):
            # Choose boundary mode based on topology
            mode = 'wrap' if self.topology in ['torus', 'sphere'] else 'nearest'
            lap = convolve(self.phi, self.kern, mode=mode)
            
            # Non-linear potential and propagation speed
            Vp = -self.pot_lin * self.phi + self.pot_cub * self.phi**3
            c2 = 1.0 / (1.0 + self.tension * self.phi**2 + 1e-6)
            acc = c2 * lap - Vp

            vel = self.phi - self.phi_o
            self.phi_o = self.phi.copy()
            self.phi = self.phi + (1 - self.damp * self.dt) * vel + self.dt**2 * acc
            
            # Step any sub-simulations within agents and couple back
            if self.track_fractals:
                for agent in self.agents.values():
                    if agent.sub_sim is not None:
                        agent.step_sub_simulation(self.phi)
                
                # Detect stable patterns
                self.detect_stable_patterns()
            
            # Calculate energy after each step
            self.calculate_total_energy()
    
    def change_topology(self, new_topology):
        with self.lock:
            self.topology = new_topology
            self.init_field()
            self.phi_o = self.phi.copy()
            # Reset fractal tracking
            self.next_agent_id = 1
            self.agents = {}
            self.fractal_mask = np.zeros((self.N, self.N, self.N), dtype=bool)
            # Reset energy history
            self.energy_history.clear()
            
    def set_custom_field(self, field):
        """Set a custom field to use with 'custom' topology"""
        with self.lock:
            self.custom_field = field.copy()
            if self.topology == 'custom':
                self.init_field()
                self.phi_o = self.phi.copy()

    def save_state(self, filename='state.npz'):
        """Save the current simulation state to a file"""
        with self.lock:
            np.savez(filename, 
                    phi=self.phi, 
                    topology=self.topology,
                    grid_size=self.N,
                    tension=self.tension,
                    pot_lin=self.pot_lin,
                    pot_cub=self.pot_cub,
                    dt=self.dt,
                    damp=self.damp)
            print(f"State saved to {filename}")
            
    def load_state(self, filename='state.npz'):
        """Load a simulation state from a file"""
        with self.lock:
            try:
                data = np.load(filename, allow_pickle=True)
                # Check if grid sizes match
                loaded_N = data['grid_size'].item()
                if loaded_N != self.N:
                    print(f"Warning: Loaded state has grid size {loaded_N}, current is {self.N}")
                    print(f"Resizing grid to {loaded_N}...")
                    self.resize_grid(loaded_N)
                
                # Load state and parameters
                self.phi = data['phi']
                self.phi_o = self.phi.copy()
                self.topology = str(data['topology'])
                self.tension = float(data['tension'])
                self.pot_lin = float(data['pot_lin'])
                self.pot_cub = float(data['pot_cub'])
                self.dt = float(data['dt'])
                self.damp = float(data['damp'])
                
                # Reset tracking data
                self.next_agent_id = 1
                self.agents = {}
                self.fractal_mask = np.zeros((self.N, self.N, self.N), dtype=bool)
                self.energy_history.clear()
                
                print(f"State loaded from {filename}")
                return True
            except Exception as e:
                print(f"Error loading state: {e}")
                return False

# ── Custom shape editor using Tkinter ─────────────────────
class CustomShapeEditor:
    def __init__(self, sim, on_close=None):
        self.sim = sim
        self.on_close = on_close
        self.root = tk.Tk()
        self.root.title("Custom Shape Editor")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set up the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Shape selection
        ttk.Label(self.control_frame, text="Base Shape:").pack(pady=(0, 5))
        self.shape_var = tk.StringVar(value="sphere")
        shapes = ["sphere", "cube", "torus", "wave", "gaussian"]
        shape_combo = ttk.Combobox(self.control_frame, textvariable=self.shape_var, values=shapes)
        shape_combo.pack(fill=tk.X, pady=(0, 10))
        shape_combo.bind("<<ComboboxSelected>>", self.update_preview)
        
        # Parameters frame
        param_frame = ttk.LabelFrame(self.control_frame, text="Parameters")
        param_frame.pack(fill=tk.X, pady=10)
        
        # Grid size
        ttk.Label(param_frame, text="Grid Size:").pack(anchor=tk.W)
        self.grid_size_var = tk.IntVar(value=32)
        grid_sizes = [16, 32, 64, 128]
        grid_combo = ttk.Combobox(param_frame, textvariable=self.grid_size_var, values=grid_sizes)
        grid_combo.pack(fill=tk.X, pady=(0, 10))
        grid_combo.bind("<<ComboboxSelected>>", self.update_preview)
        
        # Size slider
        ttk.Label(param_frame, text="Size:").pack(anchor=tk.W)
        self.size_var = tk.DoubleVar(value=0.3)
        size_slider = ttk.Scale(param_frame, from_=0.1, to=0.8, variable=self.size_var, orient=tk.HORIZONTAL)
        size_slider.pack(fill=tk.X, pady=(0, 10))
        size_slider.bind("<ButtonRelease-1>", self.update_preview)
        
        # Position sliders
        ttk.Label(param_frame, text="Position:").pack(anchor=tk.W)
        pos_frame = ttk.Frame(param_frame)
        pos_frame.pack(fill=tk.X, pady=(0, 10))
        
        # X position
        ttk.Label(pos_frame, text="X:").pack(side=tk.LEFT)
        self.pos_x_var = tk.DoubleVar(value=0.5)
        pos_x_slider = ttk.Scale(pos_frame, from_=0.0, to=1.0, variable=self.pos_x_var, orient=tk.HORIZONTAL)
        pos_x_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        pos_x_slider.bind("<ButtonRelease-1>", self.update_preview)
        
        pos_frame2 = ttk.Frame(param_frame)
        pos_frame2.pack(fill=tk.X, pady=(0, 10))
        
        # Y position
        ttk.Label(pos_frame2, text="Y:").pack(side=tk.LEFT)
        self.pos_y_var = tk.DoubleVar(value=0.5)
        pos_y_slider = ttk.Scale(pos_frame2, from_=0.0, to=1.0, variable=self.pos_y_var, orient=tk.HORIZONTAL)
        pos_y_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        pos_y_slider.bind("<ButtonRelease-1>", self.update_preview)
        
        pos_frame3 = ttk.Frame(param_frame)
        pos_frame3.pack(fill=tk.X, pady=(0, 10))
        
        # Z position
        ttk.Label(pos_frame3, text="Z:").pack(side=tk.LEFT)
        self.pos_z_var = tk.DoubleVar(value=0.5)
        pos_z_slider = ttk.Scale(pos_frame3, from_=0.0, to=1.0, variable=self.pos_z_var, orient=tk.HORIZONTAL)
        pos_z_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        pos_z_slider.bind("<ButtonRelease-1>", self.update_preview)
        
        # Distortion
        ttk.Label(param_frame, text="Distortion:").pack(anchor=tk.W)
        self.distortion_var = tk.DoubleVar(value=0.0)
        distortion_slider = ttk.Scale(param_frame, from_=0.0, to=1.0, variable=self.distortion_var, orient=tk.HORIZONTAL)
        distortion_slider.pack(fill=tk.X, pady=(0, 10))
        distortion_slider.bind("<ButtonRelease-1>", self.update_preview)
        
        # Action buttons
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Apply to Simulation", command=self.apply_to_sim).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset_params).pack(fill=tk.X, pady=5)
        
        # Visualization area
        self.viz_frame = ttk.Frame(self.main_frame)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Set up matplotlib figure
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize preview
        self.custom_field = None
        self.update_preview()
        
    def generate_field(self):
        """Generate a 3D field based on current parameters"""
        grid_size = self.grid_size_var.get()
        size = self.size_var.get()
        pos_x = self.pos_x_var.get()
        pos_y = self.pos_y_var.get()
        pos_z = self.pos_z_var.get()
        distortion = self.distortion_var.get()
        shape = self.shape_var.get()
        
        # Create coordinate grid
        x = np.linspace(0, 1, grid_size)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        
        # Position center
        X = X - pos_x
        Y = Y - pos_y
        Z = Z - pos_z
        
        # Generate base shape
        if shape == "sphere":
            R = np.sqrt(X**2 + Y**2 + Z**2)
            field = np.exp(-(R/size)**2)
        elif shape == "cube":
            # Soft cube using max norm
            R = np.maximum.reduce([np.abs(X), np.abs(Y), np.abs(Z)])
            field = np.exp(-(R/size)**2)
        elif shape == "torus":
            # Torus
            R_major = 0.3
            R_minor = size * 0.5
            d_circle = np.sqrt((np.sqrt(X**2 + Y**2) - R_major)**2 + Z**2)
            field = np.exp(-(d_circle/R_minor)**2)
        elif shape == "wave":
            # Standing wave
            k = 2 * np.pi / size
            field = 0.5 * (np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z) + 1)
        else:  # gaussian
            # Gaussian pulse
            R2 = X**2 + Y**2 + Z**2
            field = 2 * np.exp(-R2/(2*(size**2)))
        
        # Add distortion if needed
        if distortion > 0:
            # Generate random noise
            noise = np.random.randn(*field.shape) * distortion
            # Smooth the noise
            smooth_noise = gaussian_filter(noise, sigma=2.0)
            # Apply to field
            field = field + smooth_noise * field
            # Normalize to [0, 1]
            field = np.clip(field, 0, None)
            field = field / field.max() if field.max() > 0 else field
        
        return field
        
    def update_preview(self, event=None):
        """Update the 3D preview of the shape"""
        # Generate the field
        self.custom_field = self.generate_field()
        
        # Update the plot
        self.ax.clear()
        
        # Use marching cubes to get a surface
        try:
            level = 0.5
            verts, faces, _, _ = marching_cubes(self.custom_field, level=level)
            
            # Scale and center vertices
            verts = verts / self.custom_field.shape[0]
            
            # Plot the surface
            self.ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                                color='green', alpha=0.8, edgecolor='none')
            
            # Set axis limits
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_zlim(0, 1)
            
            # Remove axis labels
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.set_zticklabels([])
            
            # Set title
            self.ax.set_title(f"Custom Shape Preview: {self.shape_var.get().capitalize()}")
            
            # Update canvas
            self.canvas.draw()
        except Exception as e:
            print(f"Error in marching cubes: {e}")
        
    def apply_to_sim(self):
        """Apply the custom shape to the main simulation"""
        if self.custom_field is not None:
            # Apply the field to the simulator
            self.sim.set_custom_field(self.custom_field)
            self.sim.change_topology('custom')
            
            # Show confirmation
            messagebox.showinfo("Applied", "Custom shape applied to simulation!")
        
    def reset_params(self):
        """Reset parameters to defaults"""
        self.shape_var.set("sphere")
        self.grid_size_var.set(32)
        self.size_var.set(0.3)
        self.pos_x_var.set(0.5)
        self.pos_y_var.set(0.5)
        self.pos_z_var.set(0.5)
        self.distortion_var.set(0.0)
        self.update_preview()
    
    def on_closing(self):
        """Handle window closing"""
        if self.on_close:
            self.on_close()
        self.root.destroy()
        
    def run(self):
        """Run the editor"""
        self.root.mainloop()

# ── Parameter sweep module ───────────────────────────────────
class ParameterSweeper:
    """Class to perform parameter sweeps and generate phase diagrams"""
    def __init__(self, sim):
        self.sim = sim
        self.results = {}
        
    def sweep(self, tension_range, pot_lin_range, pot_cub_range, steps=1000, size=32):
        """
        Perform parameter sweep across specified ranges
        Returns a dictionary of results with pattern counts
        """
        # Create a headless simulator for parameter sweeping
        sweep_sim = MiniWoW(N=size, topology='sphere', track_fractals=False)
        
        # Define parameter grid
        tensions = np.linspace(*tension_range, num=int(np.sqrt(steps)))
        pot_lins = np.linspace(*pot_lin_range, num=int(np.sqrt(steps)))
        pot_cubs = np.linspace(*pot_cub_range, num=int(np.sqrt(steps)))
        
        results = {}
        
        # We'll primarily sweep through tension and pot_lin while keeping pot_cub fixed
        # at a few different values
        for pot_cub in pot_cubs:
            results_grid = np.zeros((len(tensions), len(pot_lins)))
            
            for i, tension in enumerate(tensions):
                for j, pot_lin in enumerate(pot_lins):
                    # Update parameters
                    sweep_sim.tension = tension
                    sweep_sim.pot_lin = pot_lin
                    sweep_sim.pot_cub = pot_cub
                    
                    # Reset field
                    sweep_sim.reset()
                    
                    # Run simulation for a while
                    sweep_sim.step(500)  # Reduced step count for faster results
                    
                    # Count patterns/components using a simple thresholding approach
                    binary = (sweep_sim.phi > 1.0)
                    labeled, num_features = label(binary)
                    
                    # Store result
                    results_grid[i, j] = num_features
            
            results[float(pot_cub)] = {
                'grid': results_grid,
                'tensions': tensions,
                'pot_lins': pot_lins
            }
        
        self.results = results
        return results
        
    def save_results(self, filename='param_sweep.npz'):
        """Save parameter sweep results to a file"""
        if not self.results:
            print("No results to save!")
            return False
            
        np.savez(filename, results=self.results)
        print(f"Parameter sweep results saved to {filename}")
        return True
        
    def load_results(self, filename='param_sweep.npz'):
        """Load parameter sweep results from a file"""
        try:
            data = np.load(filename, allow_pickle=True)
            self.results = data['results'].item()
            print(f"Parameter sweep results loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading parameter sweep results: {e}")
            return False
            
    def plot_phase_diagram(self, pot_cub=None, ax=None):
        """Plot a phase diagram for a specific pot_cub value"""
        if not self.results:
            print("No results to plot!")
            return None
            
        # If pot_cub is None, use the first one in the results
        if pot_cub is None:
            pot_cub = list(self.results.keys())[0]
        
        # Get the result data for the specified pot_cub
        if pot_cub not in self.results:
            print(f"No results for pot_cub={pot_cub}")
            return None
            
        data = self.results[pot_cub]
        grid = data['grid']
        tensions = data['tensions']
        pot_lins = data['pot_lins']
        
        # Create a figure if none provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create the heatmap
        im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis',
                      extent=[pot_lins[0], pot_lins[-1], tensions[0], tensions[-1]])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Number of Patterns')
        
        # Set labels
        ax.set_xlabel('pot_lin')
        ax.set_ylabel('tension')
        ax.set_title(f'Phase Diagram (pot_cub={pot_cub:.2f})')
        
        return ax

# ── simulation thread ─────────────────────────────────────
paused = False
current_grid_size = 64  # Default grid size
enable_fractal_tracking = True  # Enable pattern tracking by default

def sim_worker(sim, q, stop_evt):
    while not stop_evt.is_set():
        if not paused:
            sim.step(2)
            with sim.lock:
                if not q.full():
                    q.put((sim.phi.copy(), sim.agents))
        time.sleep(0.01)

# Start with no initial shape topology
sim = MiniWoW(N=current_grid_size, topology='none', track_fractals=enable_fractal_tracking)
field_q = queue.Queue(maxsize=2)
stop_evt = threading.Event()
sim_thread = threading.Thread(target=sim_worker, args=(sim, field_q, stop_evt), daemon=True)
sim_thread.start()

# ── Ursina setup ─────────────────────────
app = Ursina(fullscreen=True, development_mode=False)
window.color = color.black          # or color.rgb(0, 0, 0)
window.title = 'Enhanced Magic Box'

window.fps_counter.enabled = True

editor_cam = EditorCamera()          # WASD + RMB
# Fix: Define a custom base camera speed instead of accessing a non-existent attribute
base_camera_speed = 5.0  # Default movement speed

sun = DirectionalLight(rotation=(45, -45, 45), color=color.white, shadows=True)
AmbientLight(color=color.rgba(0.3, 0.3, 0.5, 0.1))

container = Entity()
surface = Entity(parent=container, double_sided=False, shader=lit_with_shadows_shader)  # opaque

# Create container for agent visualizations
agents_container = Entity(parent=container)
agent_entities = {}  # Dictionary to store agent visualization entities

ground = Entity(model='plane', scale=100, y=-15,
                color=color.dark_gray, texture='white_cube', texture_scale=(100, 100))

# invisible plane to catch clicks in empty space
click_plane = Entity(
    model='plane',
    scale=(100, 100, 1),
    position=(0, 0, 0),       # same y as your volume’s center
    rotation=(90, 0, 0),      # make it horizontal if you like
    collider='box',
    visible=False
)

# Status text for fractal information
fractal_info_text = Text(text="No fractals detected yet", position=(0, 0.45), origin=(0, 0), scale=0.8)


# Energy display
energy_text = Text(text="Energy: 0.00", position=(-0.7, 0.45), origin=(-0.5, 0), scale=0.8)

energy_history_text = Text(text="", position=(-0.7, 0.4), origin=(-0.5, 0), scale=0.6, color=color.yellow)



# Poke indicator (flash particle)
poke_indicator = Entity(model='sphere', scale=0.1, color=color.yellow, visible=False, 
                        always_on_top=True, shader=lit_with_shadows_shader)

# Flag to track if custom shape editor is open
custom_editor_open = False

iso_val, time_val = 1.0, 0.0
visualization_mode = 'field'  # 'field', 'agents', 'both'
poke_radius = 3
poke_amplitude = 1.0

def update_mesh(phi):
    try:
        v, f, n, _ = marching_cubes(phi, level=iso_val)
        if v.size == 0:
            surface.visible = False
            return
        v -= v.mean(0)  # center mesh
        surface.model = Mesh(vertices=v.tolist(),
                            triangles=f.flatten().tolist(),
                            normals=n.tolist(),
                            mode='triangle')
        surface.color = update_color()  # opaque color (α = 1)
        surface.visible = visualization_mode in ['field', 'both']
    except Exception as e:
        print('marching-cubes error:', e)
        surface.visible = False

def update_agent_visualizations(agents):
    """Update visualization of detected stable patterns"""
    global agent_entities, agents_container, fractal_info_text
    
    # Update info text
    if not agents:
        fractal_info_text.text = "No stable patterns detected"
    else:
        fractal_info_text.text = f"{len(agents)} stable patterns detected | {sum(1 for a in agents.values() if a.sub_sim)} with sub-simulations"
    
    # Remove entities for agents that no longer exist
    to_remove = []
    for agent_id in agent_entities:
        if agent_id not in agents:
            destroy(agent_entities[agent_id])
            to_remove.append(agent_id)
    
    for agent_id in to_remove:
        del agent_entities[agent_id]
    
    # Update or create entities for current agents
    for agent_id, agent in agents.items():
        # Skip agents that are too young to visualize
        if agent.age < 5:
            continue
            
        if agent_id in agent_entities:
            # Just update color/scale for existing entities
            entity = agent_entities[agent_id]
            scale_factor = min(1.0, agent.age / 20)  # Grow to full size over time
            entity.scale = 5 * scale_factor
            
            # Highlight agents with sub-simulations
            if agent.sub_sim is not None:
                entity.color = color.yellow
            else:
                entity.color = agent.color
        else:
            # Create new visualization for this agent
            entity = Entity(
                parent=agents_container,
                model='sphere',
                position=(-agent.position[1] + current_grid_size/2, 
                          -agent.position[2] + current_grid_size/2, 
                          -agent.position[0] + current_grid_size/2),  # Adjust for coordinate system
                scale=2,
                color=agent.color,
                shader=lit_with_shadows_shader
            )
            agent_entities[agent_id] = entity
    
    # Set visibility based on visualization mode
    agents_container.enabled = visualization_mode in ['agents', 'both']

def adapt_scale_for_grid_size():
    """Adjust scale-dependent elements for the current grid size"""
    global current_grid_size, iso_val
    
    # Fix: Try to use move_speed if it exists, otherwise do nothing for the camera
    try:
        editor_cam.move_speed = base_camera_speed * (current_grid_size / 64)
    except AttributeError:
        # If move_speed doesn't exist, try other possible attribute names
        for attr_name in ['movement_speed', 'camera_speed', 'speed']:
            if hasattr(editor_cam, attr_name):
                setattr(editor_cam, attr_name, base_camera_speed * (current_grid_size / 64))
                break
        # If none of the above work, print a message
        else:
            print("Warning: Could not find camera speed attribute to adjust")
    
    # Adjust ground texture scale
    ground.texture_scale = (current_grid_size, current_grid_size)
    
    # Adjust isosurface threshold
    default_iso = 1.0
    iso_val = default_iso * (64 / current_grid_size)

def update_color():
    hue = (time_val * 0.1) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return color.rgb(r, g, b)  # α = 1 → opaque


# Function to open custom shape editor
def open_custom_shape_editor():
    global custom_editor_open, paused      # keep paused global
    
    if custom_editor_open:
        return
    
    custom_editor_open = True
    prev_paused = paused
    paused = True  # Pause the simulation while editing
    
    def on_editor_close():
        global custom_editor_open, paused
        custom_editor_open = False
        paused = prev_paused  # Restore previous pause state
    
    # Create and run the editor in a separate thread
    editor_thread = threading.Thread(
        target=lambda: CustomShapeEditor(sim, on_close=on_editor_close).run(),
        daemon=True
    )
    editor_thread.start()

def handle_poke():
    """Handle poking the simulation field"""
    if mouse.left:  # Left mouse button pressed
        # Perform raycast to get position
        hit_info = raycast(camera.world_position, camera.forward, distance=100)
        
        if hit_info.hit:
            # Convert world position to grid coordinates
            world_pos = hit_info.world_point
            N = sim.N
            
            # Convert from world to grid coordinates (taking into account coordinate system differences)
            x = int(N/2 - world_pos.z)
            y = int(N/2 - world_pos.x)
            z = int(N/2 - world_pos.y)
            
            # Check if coordinates are valid
            if 0 <= x < N and 0 <= y < N and 0 <= z < N:
                # Apply poke
                if sim.apply_poke(x, y, z, radius=poke_radius, amplitude=poke_amplitude):
                    # Show poke indicator
                    poke_indicator.position = hit_info.world_point
                    poke_indicator.visible = True
                    
                    # Hide indicator after a short time
                    invoke(setattr, poke_indicator, 'visible', False, delay=0.2)

def adapt_scale_for_grid_size():
    """Adjust scale-dependent elements for the current grid size"""
    global current_grid_size, iso_val
    
    # Fix: Try to use move_speed if it exists, otherwise do nothing for the camera
    try:
        editor_cam.move_speed = base_camera_speed * (current_grid_size / 64)
    except AttributeError:
        # If move_speed doesn't exist, try other possible attribute names
        for attr_name in ['movement_speed', 'camera_speed', 'speed']:
            if hasattr(editor_cam, attr_name):
                setattr(editor_cam, attr_name, base_camera_speed * (current_grid_size / 64))
                break
        # If none of the above work, print a message
        else:
            print("Warning: Could not find camera speed attribute to adjust")
    
    # Adjust ground texture scale
    ground.texture_scale = (current_grid_size, current_grid_size)
    
    # Adjust isosurface threshold
    default_iso = 1.0
    iso_val = default_iso * (64 / current_grid_size)
    
    # Adjust ground texture scale
    ground.texture_scale = (current_grid_size, current_grid_size)
    
    # Adjust isosurface threshold
    default_iso = 1.0
    iso_val = default_iso * (64 / current_grid_size)

def update_energy_display():
    """Update the energy display and energy history text"""
    global energy_text, energy_history_text
    
    # Update the energy text
    energy_text.text = f"Energy: {sim.total_energy:.2e}"
    
    # Update the energy history text
    if sim.energy_history:
        # Show trend arrow
        if len(sim.energy_history) > 1:
            last_energy = sim.energy_history[-1]
            prev_energy = sim.energy_history[-2]
            if last_energy > prev_energy:
                trend = "↑"  # Rising energy
            elif last_energy < prev_energy:
                trend = "↓"  # Falling energy
            else:
                trend = "→"  # Stable energy
                
            # Calculate percentage change
            if prev_energy != 0:
                percent_change = (last_energy - prev_energy) / prev_energy * 100
                energy_history_text.text = f"Trend: {trend} {abs(percent_change):.2f}%"
            else:
                energy_history_text.text = f"Trend: {trend}"
        else:
            energy_history_text.text = "Monitoring energy..."
    else:
        energy_history_text.text = "No energy data yet"


# ── main update loop ──────────────────────────────────────
def update():
    global time_val
    if paused:
        return
    time_val += time.dt
    container.rotation_y += time.dt * 5
    
    # Handle poking
    handle_poke()

    try:
        while True:
            data = field_q.get_nowait()
            if isinstance(data, tuple) and len(data) == 2:
                phi, agents = data
                update_mesh(phi)
                update_agent_visualizations(agents)
                update_energy_display()  # Update the energy display
            else:
                # Handle old format if needed
                update_mesh(data)
    except queue.Empty:
        pass

# ── keyboard / UI ─────────────────────────────────────────
def input(key):
    global iso_val, paused, visualization_mode
    if key == 'left arrow':
        iso_val = max(-2.0, iso_val - 0.05)
        with sim.lock:
            update_mesh(sim.phi.copy())
    elif key == 'right arrow':
        iso_val = min(2.0, iso_val + 0.05)
        with sim.lock:
            update_mesh(sim.phi.copy())
    elif key == 'p':
        paused = not paused
        print('Paused' if paused else 'Resumed')
    elif key == 'r':  # Add reset functionality
        sim.reset()
        print('Simulation reset')
    elif key == 'v':  # Toggle visualization mode
        if visualization_mode == 'field':
            visualization_mode = 'agents'
        elif visualization_mode == 'agents':
            visualization_mode = 'both'
        else:
            visualization_mode = 'field'
        print(f'Visualization mode: {visualization_mode}')
        surface.visible = visualization_mode in ['field', 'both']
        agents_container.enabled = visualization_mode in ['agents', 'both']
    elif key == 'escape':
        stop_evt.set()
        application.quit()
    elif key == 'f':
        window.fullscreen = not window.fullscreen
    elif key == 's':  # Save state
        sim.save_state('state.npz')
    elif key == 'l':  # Load state
        if sim.load_state('state.npz'):
            # Update scaled elements for the loaded grid size
            adapt_scale_for_grid_size()

# ── slider factory ────────────────────────────────────────
def add_slider(label, attr, rng, y):
    txt = Text(text=f'{label}: {getattr(sim, attr):.3f}',
              x=-0.83, y=y + 0.04, parent=camera.ui, scale=0.75)
    sld = Slider(min=rng[0], max=rng[1], default=getattr(sim, attr),
                step=(rng[1] - rng[0]) / 200, x=-0.85, y=y, scale=0.3,
                parent=camera.ui)

    def changed():
        val = sld.value
        setattr(sim, attr, val)
        txt.text = f'{label}: {val:.3f}'
    sld.on_value_changed = changed

# ── topology and grid size selection ────────────────────────
def add_topology_selector():
    # Create a topology selection label
    Text(text='Topology:', x=0.65, y=0.4, parent=camera.ui, scale=0.75)
    
    topologies = ['box', 'sphere', 'torus', 'wave', 'random']
    buttons = []
    
    for i, topo in enumerate(topologies):
        btn = Button(text=topo.capitalize(), scale=(0.15, 0.05), x=0.7, y=0.35 - i*0.06, 
                    parent=camera.ui, color=color.light_gray)
        buttons.append(btn)
        
        def make_on_click(topology):
            def on_click():
                # Reset all button colors
                for b in buttons:
                    b.color = color.light_gray
                # Highlight the selected button
                buttons[topologies.index(topology)].color = color.azure
                # Change the topology
                sim.change_topology(topology)
            return on_click
            
        btn.on_click = make_on_click(topo)
    
    # Add custom shape button
    custom_btn = Button(text='Custom Shape', scale=(0.15, 0.05), x=0.7, y=0.35 - len(topologies)*0.06, 
                       parent=camera.ui, color=color.orange)
    custom_btn.on_click = open_custom_shape_editor
    
    # Add grid size selection - VERTICAL layout
    Text(text='Grid Size:', x=0.65, y=0.0, parent=camera.ui, scale=0.75)
    Text(text='Warning: Large sizes may slow down\nyour system!', 
         x=0.65, y=-0.05, parent=camera.ui, scale=0.5, color=color.red)
    
    grid_sizes = [32, 64, 128, 256, 512]
    grid_buttons = []
    
    for i, size in enumerate(grid_sizes):
        btn = Button(text=str(size), scale=(0.15, 0.05), x=0.7, y=-0.1 - i*0.06, 
                    parent=camera.ui, color=color.light_gray)
        grid_buttons.append(btn)
        
        def make_on_click(grid_size):
            def on_click():
                global current_grid_size, paused
                # Don't do anything if already at this size
                if current_grid_size == grid_size:
                    return
                    
                # Reset all button colors
                for b in grid_buttons:
                    b.color = color.light_gray
                # Highlight the selected button
                grid_buttons[grid_sizes.index(grid_size)].color = color.azure
                
                # Change grid size - this will disrupt the simulation temporarily
                paused_state = paused
                if not paused:
                    # Pause simulation while changing grid
                    paused = True
                    time.sleep(0.1)  # Give time for thread to pause
                
                # Update the simulation grid size
                print(f"Changing grid size to {grid_size}...")
                current_grid_size = grid_size
                sim.resize_grid(grid_size)
                
                # Clear old agent entities when changing grid size
                for ent in list(agent_entities.values()):
                    destroy(ent)
                agent_entities.clear()
                
                # Adjust camera speed, ground texture scale based on new grid size
                adapt_scale_for_grid_size()
                
                # Resume if it was running
                if not paused_state:
                    paused = False
                
            return on_click
            
        btn.on_click = make_on_click(size)
    
    # Set the initial grid size button to be highlighted
    grid_buttons[grid_sizes.index(current_grid_size)].color = color.azure

# ── tracking toggle ───────────────────────────────────────────────────
def add_tracking_toggle():
    """Add button to enable/disable fractal tracking"""
    Text(text='Fractal Tracking:', x=0.65, y=-0.45, parent=camera.ui, scale=0.75)
    
    tracking_btn = Button(
        text='Enabled' if enable_fractal_tracking else 'Disabled',
        scale=(0.15, 0.05), 
        x=0.7, y=-0.5,
        parent=camera.ui,
        color=color.green if enable_fractal_tracking else color.red
    )
    
    def toggle_tracking():
        global enable_fractal_tracking
        enable_fractal_tracking = not enable_fractal_tracking
        tracking_btn.text = 'Enabled' if enable_fractal_tracking else 'Disabled'
        tracking_btn.color = color.green if enable_fractal_tracking else color.red
        sim.track_fractals = enable_fractal_tracking
        print(f"Fractal tracking: {'enabled' if enable_fractal_tracking else 'disabled'}")
        
    tracking_btn.on_click = toggle_tracking

# ── Poke controls ──────────────────────────────────────────────────
def add_poke_controls():
    """Add controls for the poke tool"""
    Text(text='Poke Settings:', x=0.65, y=-0.6, parent=camera.ui, scale=0.75)
    
    # Radius slider
    txt_radius = Text(text=f'Radius: {poke_radius}', x=0.65, y=-0.65, parent=camera.ui, scale=0.6)
    sld_radius = Slider(min=1, max=10, default=poke_radius,
                       x=0.7, y=-0.7, scale=0.15,
                       parent=camera.ui)
    
    def radius_changed():
        global poke_radius
        poke_radius = int(sld_radius.value)
        txt_radius.text = f'Radius: {poke_radius}'
    sld_radius.on_value_changed = radius_changed
    
    # Amplitude slider
    txt_amplitude = Text(text=f'Amplitude: {poke_amplitude:.1f}', x=0.65, y=-0.75, parent=camera.ui, scale=0.6)
    sld_amplitude = Slider(min=0.1, max=5.0, default=poke_amplitude,
                         x=0.7, y=-0.8, scale=0.15,
                         parent=camera.ui)
    
    def amplitude_changed():
        global poke_amplitude
        poke_amplitude = sld_amplitude.value
        txt_amplitude.text = f'Amplitude: {poke_amplitude:.1f}'
    sld_amplitude.on_value_changed = amplitude_changed

# ── Phase diagram button ───────────────────────────────────────────
def add_phase_diagram_button():
    """Add button to run parameter sweep and generate phase diagram"""
    Text(text='Phase Diagram:', x=0.65, y=-0.9, parent=camera.ui, scale=0.75)
    
    phase_btn = Button(
        text='Run Sweep',
        scale=(0.15, 0.05), 
        x=0.7, y=-0.95,
        parent=camera.ui,
        color=color.orange
    )
    
    def run_phase_sweep():
        global paused
        # This is computationally intensive, so pause and run in a separate thread
        was_paused = paused
        paused = True
        
        # Create message to user
        msg = Text(text="Running parameter sweep...\nThis may take a while.", 
                  position=(0, 0), origin=(0, 0), scale=1.5,
                  background=True, background_color=color.rgba(0, 0, 0, 128))
        
        def sweep_thread_func():
            # Run the sweep with smaller steps for speed
            param_sweeper.sweep(
                tension_range=(1.0, 10.0), 
                pot_lin_range=(0.5, 2.0), 
                pot_cub_range=(0.1, 0.3), 
                steps=25,  # Small number for quick results
                size=32    # Small grid for speed
            )
            
            # Save results
            param_sweeper.save_results('phase_diagram.npz')
            
            # Create a simple visualization and save it
            plt.figure(figsize=(8, 6))
            param_sweeper.plot_phase_diagram()
            plt.savefig('phase_diagram.png')
            plt.close()
        
            # Update UI
            destroy(msg)
            if not was_paused:
                global paused            # tell Python we’re writing to module var
                paused = False
                
            # Notify user
            notification = Text(text="Parameter sweep complete!\nResults saved to phase_diagram.npz",
                              position=(0, 0), origin=(0, 0), scale=1.5,
                              background=True, background_color=color.rgba(0, 0, 0, 128))
            # Auto-destroy notification after a few seconds
            invoke(destroy, notification, delay=5)
        
        # Start the sweep thread
        sweep_thread = threading.Thread(target=sweep_thread_func, daemon=True)
        sweep_thread.start()
        
    phase_btn.on_click = run_phase_sweep

# Add the UI elements
add_slider('dt', 'dt', (0.01, 0.2), 0.35)
add_slider('damping', 'damp', (0.0, 0.05), 0.25)
add_slider('tension', 'tension', (0.0, 20.0), 0.15)
add_slider('pot_lin', 'pot_lin', (0.0, 2.0), 0.05)
add_slider('pot_cub', 'pot_cub', (0.0, 1.0), -0.05)
add_topology_selector()
add_tracking_toggle()
add_poke_controls()
add_phase_diagram_button()

Text('WASD+RMB fly | ←/→ iso | P pause | R reset | V toggle view | S save | L load | ESC quit',
    y=-0.45, x=0, origin=(0, 0),
    background=True, background_color=color.rgba(0, 0, 0, 128),
    parent=camera.ui)

# Adjust scale-dependent elements for the initial grid size
adapt_scale_for_grid_size()

# Start the simulation with a sphere topology to initialize
# sim.change_topology('sphere')

app.run()