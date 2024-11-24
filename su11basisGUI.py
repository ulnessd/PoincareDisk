import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QFileDialog
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.figure import Figure
from matplotlib import cm


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection="3d")
        super().__init__(self.fig)

    def plot(self, k, m, plot_type, component):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d" if plot_type == "3D Plot" else None)
        x = np.linspace(-1, 1, 200)
        y = np.linspace(-1, 1, 200)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        R2 = X**2 + Y**2

        # Define the basis function
        Psi = np.where(
            R2 < 1,
            (1 - R2)**k * Z**(m - k),
            0
        )

        # Choose the component
        if component == "Abs":
            Zplot = np.abs(Psi)
        elif component == "Re":
            Zplot = np.real(Psi)
        elif component == "Im":
            Zplot = np.imag(Psi)
        elif component == "Arg":
            Zplot = np.angle(Psi)

        # Apply the plot type
        if plot_type == "3D Plot":
            ax.plot_surface(X, Y, Zplot, cmap=cm.viridis, rstride=1, cstride=1, alpha=0.8)
            ax.set_zlabel("Z")
        elif plot_type == "Contour Plot":
            ax.contour(X, Y, Zplot, levels=50, cmap=cm.viridis)
        elif plot_type == "Density Plot":
            ax.imshow(Zplot, extent=(-1, 1, -1, 1), origin="lower", cmap=cm.viridis)

        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        if plot_type == "3D Plot":
            ax.set_zlim(Zplot.min(), Zplot.max())

        self.draw()

    def save_plot(self, dpi):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Plot", "", "PNG Files (*.png);;All Files (*)", options=options
        )
        if file_path:
            self.fig.savefig(file_path, dpi=dpi)
            QMessageBox.information(None, "Success", f"Plot saved to {file_path}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SU(1,1) Basis Function Plotter")

        # Widgets
        self.k_label = QLabel("k:")
        self.k_input = QLineEdit("2")
        self.m_label = QLabel("m:")
        self.m_input = QLineEdit("3")
        self.plot_type_label = QLabel("Plot Type:")
        self.plot_type_dropdown = QComboBox()
        self.plot_type_dropdown.addItems(["3D Plot", "Contour Plot", "Density Plot"])
        self.component_label = QLabel("Component:")
        self.abs_check = QCheckBox("Abs")
        self.re_check = QCheckBox("Re")
        self.im_check = QCheckBox("Im")
        self.arg_check = QCheckBox("Arg")
        self.abs_check.setChecked(True)  # Default selection
        self.plot_button = QPushButton("Plot")
        self.save_button = QPushButton("Save Plot")
        self.dpi_label = QLabel("DPI:")
        self.dpi_input = QLineEdit("300")
        self.canvas = PlotCanvas(self, width=8, height=6)

        # Layouts
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.k_label)
        input_layout.addWidget(self.k_input)
        input_layout.addWidget(self.m_label)
        input_layout.addWidget(self.m_input)
        input_layout.addWidget(self.plot_type_label)
        input_layout.addWidget(self.plot_type_dropdown)

        component_layout = QHBoxLayout()
        component_layout.addWidget(self.component_label)
        component_layout.addWidget(self.abs_check)
        component_layout.addWidget(self.re_check)
        component_layout.addWidget(self.im_check)
        component_layout.addWidget(self.arg_check)

        save_layout = QHBoxLayout()
        save_layout.addWidget(self.dpi_label)
        save_layout.addWidget(self.dpi_input)
        save_layout.addWidget(self.save_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(component_layout)
        main_layout.addWidget(self.plot_button)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(save_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connections
        self.plot_button.clicked.connect(self.plot_basis_function)
        self.save_button.clicked.connect(self.save_plot)

    def plot_basis_function(self):
        try:
            k = float(self.k_input.text())
            m = float(self.m_input.text())
            if m < k:
                raise ValueError("Eigenvalue m must satisfy m >= k.")
            plot_type = self.plot_type_dropdown.currentText()
            component = "Abs" if self.abs_check.isChecked() else \
                        "Re" if self.re_check.isChecked() else \
                        "Im" if self.im_check.isChecked() else \
                        "Arg"
            self.canvas.plot(k, m, plot_type, component)
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))

    def save_plot(self):
        try:
            dpi = int(self.dpi_input.text())
            self.canvas.save_plot(dpi)
        except ValueError:
            QMessageBox.critical(self, "Input Error", "DPI must be an integer.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
