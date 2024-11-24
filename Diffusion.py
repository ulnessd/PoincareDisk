import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import time
import traceback

class SimulationThread(QThread):
    # Signal to emit the updated positions
    update_signal = pyqtSignal(object)

    def __init__(self, num_particles=5000, step_size=0.05, animation_speed=0.05, steps_per_update=10):
        super().__init__()
        self.num_particles = num_particles
        self.step_size = step_size
        self.animation_speed = animation_speed  # Time between steps in seconds
        self.steps_per_update = steps_per_update
        self.is_running = False
        # Initialize all particles at the origin
        self.positions = np.zeros((self.num_particles, 2))

    def run(self):
        self.is_running = True
        step_counter = 0
        try:
            while self.is_running:
                # Generate random angles for all particles
                theta = np.random.uniform(0, 2 * np.pi, self.num_particles)
                directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)  # Shape: (num_particles, 2)

                # Calculate Euclidean steps
                euclidean_steps = np.tanh(self.step_size / 2) * directions  # Shape: (num_particles, 2)

                # Perform Möbius addition for hyperbolic translation
                a = euclidean_steps
                b = self.positions
                dot_product = np.sum(a * b, axis=1)  # Shape: (num_particles,)
                norm_a_sq = np.sum(a**2, axis=1)  # Shape: (num_particles,)
                norm_b_sq = np.sum(b**2, axis=1)  # Shape: (num_particles,)

                numerator = ((1 + 2 * dot_product + norm_b_sq)[:, np.newaxis] * a +
                             (1 - norm_a_sq)[:, np.newaxis] * b)  # Shape: (num_particles, 2)
                denominator = 1 + 2 * dot_product + (norm_a_sq * norm_b_sq)  # Shape: (num_particles,)

                # Avoid division by zero
                denominator = denominator[:, np.newaxis]
                new_positions = numerator / denominator

                # Ensure new positions are inside the unit disk
                norms = np.linalg.norm(new_positions, axis=1)
                outside = norms >= 1
                if np.any(outside):
                    new_positions[outside] = (new_positions[outside].T / norms[outside]).T * 0.99

                # Update positions
                self.positions = new_positions

                step_counter += 1

                # Emit updated positions after 'steps_per_update' steps
                if step_counter >= self.steps_per_update:
                    self.update_signal.emit(self.positions.copy())
                    step_counter = 0

                # Control animation speed
                time.sleep(self.animation_speed)
        except Exception as e:
            print("Exception in SimulationThread:")
            traceback.print_exc()
            self.is_running = False

    def stop(self):
        self.is_running = False
        self.wait()

    def set_num_particles(self, num_particles):
        self.num_particles = num_particles
        # Reset positions if the number of particles changes
        self.positions = np.zeros((self.num_particles, 2))

    def set_step_size(self, step_size):
        self.step_size = step_size

    def set_animation_speed(self, animation_speed):
        self.animation_speed = animation_speed

    def set_steps_per_update(self, steps_per_update):
        self.steps_per_update = steps_per_update

class RandomWalkDiffusion(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Walk Diffusion on Poincaré Disk")
        self.setGeometry(100, 100, 1000, 800)  # Increased width to accommodate both plots

        # Default parameters
        self.num_particles = 5000
        self.step_size = 0.05
        self.animation_speed = 0.05  # Faster animation
        self.steps_per_update = 10

        # Set up the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        # Set up the matplotlib figure with two subplots
        self.fig = Figure(figsize=(10,8))
        self.canvas = FigureCanvas(self.fig)
        self.ax_main = self.fig.add_subplot(211)
        self.ax_radial = self.fig.add_subplot(212)

        # Configure the Poincaré disk plot
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlim(-1.1, 1.1)
        self.ax_main.set_ylim(-1.1, 1.1)
        self.ax_main.axis('off')

        # Draw the boundary circle
        boundary = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        self.ax_main.add_artist(boundary)

        # Initialize scatter plot for particles
        self.scatter = self.ax_main.scatter([], [], s=1, color='blue', alpha=0.5)

        # Configure the RDF plot
        self.ax_radial.set_xlim(0, 1)
        self.ax_radial.set_ylim(0, 10)  # Initial y-axis limit; will adjust dynamically
        self.ax_radial.set_xlabel('Radial Distance')
        self.ax_radial.set_ylabel('Number of Particles per Unit Area')
        self.ax_radial.set_title('Radial Distribution Function')
        self.ax_radial.grid(True)

        # Initialize RDF bars (to optimize performance)
        self.rdf_bins = 50
        self.rdf_counts = np.zeros(self.rdf_bins)
        self.bin_edges = np.linspace(0, 1, self.rdf_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        self.rdf_bars = self.ax_radial.bar(self.bin_centers, self.rdf_counts, width=self.bin_width, color='green', alpha=0.7, edgecolor='black')

        # Add the canvas to the layout
        self.main_layout.addWidget(self.canvas)

        # Set up control panel
        self.control_layout = QHBoxLayout()

        # Start Button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)
        self.control_layout.addWidget(self.start_button)

        # Pause Button
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_simulation)
        self.pause_button.setEnabled(False)  # Disabled initially
        self.control_layout.addWidget(self.pause_button)

        # Reset Button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.reset_button.setEnabled(False)  # Disabled initially
        self.control_layout.addWidget(self.reset_button)

        # Number of Particles SpinBox
        self.num_particles_label = QLabel("Number of Particles:")
        self.control_layout.addWidget(self.num_particles_label)
        self.num_particles_spinbox = QSpinBox()
        self.num_particles_spinbox.setMinimum(1)
        self.num_particles_spinbox.setMaximum(100000)
        self.num_particles_spinbox.setValue(self.num_particles)
        self.num_particles_spinbox.setToolTip("Set the number of particles in the simulation.")
        self.num_particles_spinbox.valueChanged.connect(self.update_num_particles)
        self.control_layout.addWidget(self.num_particles_spinbox)

        # Step Size Slider
        self.step_size_label = QLabel("Step Size:")
        self.control_layout.addWidget(self.step_size_label)
        self.step_size_slider = QSlider(Qt.Horizontal)
        self.step_size_slider.setMinimum(1)
        self.step_size_slider.setMaximum(100)
        self.step_size_slider.setValue(int(self.step_size * 100))
        self.step_size_slider.setTickInterval(10)
        self.step_size_slider.setTickPosition(QSlider.TicksBelow)
        self.step_size_slider.valueChanged.connect(self.update_step_size)
        self.control_layout.addWidget(self.step_size_slider)

        # Animation Speed Slider
        self.speed_label = QLabel("Speed:")
        self.control_layout.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(int((1 - self.animation_speed) * 100))
        self.speed_slider.setTickInterval(10)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.valueChanged.connect(self.update_animation_speed)
        self.control_layout.addWidget(self.speed_slider)

        # Steps per Update SpinBox
        self.steps_label = QLabel("Steps per Update:")
        self.control_layout.addWidget(self.steps_label)
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setMinimum(1)
        self.steps_spinbox.setMaximum(1000)
        self.steps_spinbox.setValue(self.steps_per_update)
        self.steps_spinbox.setToolTip("Number of steps to execute between each plot update.")
        self.steps_spinbox.valueChanged.connect(self.update_steps_per_update)
        self.control_layout.addWidget(self.steps_spinbox)

        self.main_layout.addLayout(self.control_layout)

        # Initialize the simulation thread
        self.simulation_thread = None

    def start_simulation(self):
        if self.simulation_thread is None or not self.simulation_thread.isRunning():
            # Initialize the simulation thread
            self.simulation_thread = SimulationThread(
                num_particles=self.num_particles,
                step_size=self.step_size,
                animation_speed=self.animation_speed,
                steps_per_update=self.steps_per_update
            )
            # Connect the update signal to the update_plot method
            self.simulation_thread.update_signal.connect(self.update_plot)
            # Start the thread
            self.simulation_thread.start()
            # Disable controls that shouldn't change during simulation
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.num_particles_spinbox.setEnabled(False)  # Disable changing number of particles during simulation

    def pause_simulation(self):
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            # Re-enable the Start button to allow resuming
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)

    def reset_simulation(self):
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
        # Reset particle positions to the origin
        self.positions = np.zeros((self.num_particles, 2))
        self.scatter.set_offsets(self.positions)
        # Reset RDF plot
        self.ax_radial.cla()
        self.ax_radial.set_xlim(0, 1)
        self.ax_radial.set_ylim(0, 10)
        self.ax_radial.set_xlabel('Radial Distance')
        self.ax_radial.set_ylabel('Number of Particles per Unit Area')
        self.ax_radial.set_title('Radial Distribution Function')
        self.ax_radial.grid(True)
        # Re-initialize RDF bars
        self.rdf_bars = self.ax_radial.bar(self.bin_centers, self.rdf_counts, width=self.bin_width, color='green', alpha=0.7, edgecolor='black')
        self.canvas.draw()
        # Re-enable the Start button and other controls
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.num_particles_spinbox.setEnabled(True)

    def update_num_particles(self, value):
        self.num_particles = value
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            # If simulation is running, stop it and reset
            self.simulation_thread.stop()
            self.reset_simulation()

    def update_step_size(self, value):
        self.step_size = value / 100.0
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            self.simulation_thread.set_step_size(self.step_size)

    def update_animation_speed(self, value):
        # Invert the slider value to get faster speed with higher slider value
        self.animation_speed = (100 - value) / 1000.0  # Convert to seconds
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            self.simulation_thread.set_animation_speed(self.animation_speed)

    def update_steps_per_update(self, value):
        self.steps_per_update = value
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            self.simulation_thread.set_steps_per_update(self.steps_per_update)

    def update_plot(self, positions):
        try:
            # Update the scatter plot with new positions
            self.scatter.set_offsets(positions)
            # Compute radial distances
            radial_distances = np.linalg.norm(positions, axis=1)
            # Compute histogram
            counts, bins = np.histogram(radial_distances, bins=self.rdf_bins, range=(0,1))
            # Calculate bin centers and bin widths
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_width = bins[1] - bins[0]
            # Calculate area of each annulus (2 * pi * r * dr)
            areas = 2 * np.pi * bin_centers * bin_width
            # Avoid division by zero by setting areas to a very small number where r=0
            areas = np.where(bin_centers == 0, 1e-8, areas)
            # Normalize counts by area to get density
            rdf = counts / areas
            # Update RDF bars
            for bar, height in zip(self.rdf_bars, rdf):
                bar.set_height(height)
            # Adjust y-axis limit if necessary
            max_rdf = rdf.max() if rdf.size > 0 else 10
            self.ax_radial.set_ylim(0, max_rdf * 1.1)
            # Redraw the canvas
            self.canvas.draw()
        except Exception as e:
            print("Exception in update_plot:")
            traceback.print_exc()

    def closeEvent(self, event):
        # Ensure the simulation thread is stopped when closing the application
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = RandomWalkDiffusion()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()