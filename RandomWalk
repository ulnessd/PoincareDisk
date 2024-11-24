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


class RandomWalkThread(QThread):
    # Signal to emit the updated path
    update_signal = pyqtSignal(object)

    def __init__(self, step_size=0.05, animation_speed=0.1, batch_size=10):
        super().__init__()
        self.step_size = step_size
        self.animation_speed = animation_speed
        self.is_running = False
        self.path = [np.array([0.0, 0.0])]  # Start at the origin
        self.batch_size = batch_size  # Number of steps before emitting an update

    def run(self):
        self.is_running = True
        step_counter = 0
        last_emit_time = time.time()
        emit_interval = 0.05  # Minimum time between emits (in seconds)

        try:
            while self.is_running:
                # Choose a random angle
                theta = np.random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(theta), np.sin(theta)])

                # Perform hyperbolic step
                current = self.path[-1]
                new = self.hyperbolic_step(current, direction, self.step_size)
                self.path.append(new)

                step_counter += 1

                current_time = time.time()
                if (current_time - last_emit_time) >= emit_interval and step_counter >= self.batch_size:
                    self.update_signal.emit(self.path.copy())
                    step_counter = 0
                    last_emit_time = current_time

                # Sleep to control animation speed
                time.sleep(self.animation_speed)
        except Exception as e:
            print(f"Error in RandomWalkThread: {e}")

        # Emit any remaining steps after stopping
        if step_counter > 0:
            self.update_signal.emit(self.path.copy())

    def stop(self):
        self.is_running = False
        self.wait()  # Wait until the thread finishes

    def set_step_size(self, step_size):
        self.step_size = step_size

    def set_animation_speed(self, animation_speed):
        self.animation_speed = animation_speed

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def hyperbolic_step(self, current, direction, step_size):
        # Normalize direction
        direction = direction / np.linalg.norm(direction)

        # Calculate the Euclidean step corresponding to the hyperbolic step size
        euclidean_step = np.tanh(step_size / 2) * direction

        # Möbius addition to perform hyperbolic translation
        a = euclidean_step
        b = current
        numerator = (1 + 2 * np.dot(a, b) + np.linalg.norm(b) ** 2) * a + (1 - np.linalg.norm(a) ** 2) * b
        denominator = 1 + 2 * np.dot(a, b) + (np.linalg.norm(a) ** 2 * np.linalg.norm(b) ** 2)
        if denominator == 0:
            return b  # Avoid division by zero
        new = numerator / denominator

        # Ensure new point is inside the disk
        norm_new = np.linalg.norm(new)
        if norm_new >= 1:
            new = new / norm_new * 0.99  # Slightly inside

        return new


class PoincareDiskRandomWalk(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poincaré Disk Random Walk")
        self.setGeometry(100, 100, 900, 800)  # Increased width to accommodate new control

        # Initialize parameters
        self.step_size = 0.05  # Hyperbolic step size
        self.animation_speed = 0.1  # Seconds between steps
        self.steps_per_update = 10  # Default number of steps per update

        # Set up the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        # Set up the matplotlib figure
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.axis('off')

        # Draw the boundary circle using matplotlib.patches.Circle
        boundary = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        self.ax.add_artist(boundary)

        # Initialize plot elements
        self.path_line, = self.ax.plot([], [], color='blue', linewidth=1)
        self.current_point, = self.ax.plot([], [], 'ro')

        # Add the canvas to the layout
        self.main_layout.addWidget(self.canvas)

        # Set up control panel
        self.control_layout = QHBoxLayout()

        # Start Button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_walk)
        self.control_layout.addWidget(self.start_button)

        # Pause Button
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_walk)
        self.control_layout.addWidget(self.pause_button)

        # Reset Button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_walk)
        self.control_layout.addWidget(self.reset_button)

        # Step Size Slider
        self.step_size_label = QLabel("Step Size:")
        self.control_layout.addWidget(self.step_size_label)
        self.step_size_slider = QSlider(Qt.Horizontal)
        self.step_size_slider.setMinimum(1)
        self.step_size_slider.setMaximum(100)  # Max step size of 1.0
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
        self.speed_slider.setMaximum(100)  # Control sleep time from 0.01 to 1.0 seconds
        self.speed_slider.setValue(int((1 - self.animation_speed) * 100))
        self.speed_slider.setTickInterval(10)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.valueChanged.connect(self.update_speed)
        self.control_layout.addWidget(self.speed_slider)

        # Steps per Update SpinBox
        self.steps_label = QLabel("Steps per Update:")
        self.control_layout.addWidget(self.steps_label)
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setMinimum(1)
        self.steps_spinbox.setMaximum(1000)  # Adjust as needed
        self.steps_spinbox.setValue(self.steps_per_update)
        self.steps_spinbox.setToolTip("Number of steps to execute between each plot update.")
        self.steps_spinbox.valueChanged.connect(self.update_steps_per_update)
        self.control_layout.addWidget(self.steps_spinbox)

        self.main_layout.addLayout(self.control_layout)

        # Initialize the worker thread as None
        self.thread = None

    def start_walk(self):
        if self.thread is None or not self.thread.isRunning():
            # Create a new worker thread with current parameters
            self.thread = RandomWalkThread(
                step_size=self.step_size,
                animation_speed=self.animation_speed,
                batch_size=self.steps_per_update
            )
            # Connect the update signal to the update_plot slot
            self.thread.update_signal.connect(self.update_plot)
            # Start the thread
            self.thread.start()

    def pause_walk(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()

    def reset_walk(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        # Reset path and update plot
        self.path = []
        self.path_line.set_data([], [])
        self.current_point.set_data([], [])
        self.canvas.draw()

    def update_step_size(self, value):
        self.step_size = value / 100.0
        if self.thread is not None:
            self.thread.set_step_size(self.step_size)

    def update_speed(self, value):
        # Invert the slider value to get faster speed with higher slider value
        self.animation_speed = (100 - value) / 100.0
        if self.thread is not None:
            self.thread.set_animation_speed(self.animation_speed)

    def update_steps_per_update(self, value):
        self.steps_per_update = value
        if self.thread is not None:
            self.thread.set_batch_size(self.steps_per_update)

    def update_plot(self, path):
        if not path:
            return
        path_np = np.array(path)

        # Limit the number of points to display
        max_points = 1000  # Adjust as needed
        if len(path_np) > max_points:
            path_np = path_np[-max_points:]

        self.path_line.set_data(path_np[:, 0], path_np[:, 1])
        # Wrap coordinates in lists to make them sequences
        self.current_point.set_data([path_np[-1, 0]], [path_np[-1, 1]])
        self.canvas.draw()

    def closeEvent(self, event):
        # Ensure the thread is stopped when closing the application
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = PoincareDiskRandomWalk()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
