from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QHBoxLayout, QGroupBox, QGridLayout, QComboBox, QRadioButton, QSpinBox, QScrollArea
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import sys
from typing import Union
from eddington import FittingData, fit, fitting_functions_list, fitting_function

import numpy as np
import os
from scipy.signal import medfilt

# Fit Data

# fits = {fitting_functions_list.linear : "linear", fitting_functions_list.normal : "normal", fitting_functions_list.exponential : "exponential"}
# box = "normal"
# a0 = [1, 1, 1]



@fitting_function(
    n=2,
    syntax="a[0] * exp(a[1] * x)",
)  # pylint: disable=C0103
def exponential_0(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return a[0] * np.exp(a[1] * x)

@fitting_function(
    n=3,
    syntax="a[0] * exp( - ((x - a[1]) / a[2]) ^ 2)",
)  # pylint: disable=C0103
def normal_0(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return a[0] * np.exp(-(((x - a[1]) / a[2]) ** 2))

@fitting_function(
    n=4,
    syntax="a[0] * np.cos((a[1] * x) - a[2]) ** 2 + a[3]",
)  # pylint: disable=C0103
def cos2_0(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return a[0] * np.cos((a[1] * x) - a[2]) ** 2 + a[3]

class DataPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.noiseShow = False
        self.setStyleSheet("background-color:#156082;")
        self.setWindowTitle("Fitting Nezequ")

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QHBoxLayout (self.main_widget)

        self.Side = QGroupBox ()
        #self.Side.setStyleSheet("background-color:white;")
        self.SideLayout = QVBoxLayout ()
        
        self.UpperButtonsParentContainer = QGroupBox ()
        self.UpperButtonsParentContainer.setStyleSheet("background-color:white;")
        self.UpperButtonsParent = QVBoxLayout ()
        self.UpperButtons1 = QHBoxLayout ()
        self.UpperButtons2 = QHBoxLayout ()

        #Upper Buttons

        self.load_button = QPushButton("Load Data", self)
        self.load_button.clicked.connect(self.load_data)
        self.UpperButtons1.addWidget(self.load_button)

        self.plot_button = QPushButton("Plot Data", self)
        self.plot_button.clicked.connect(self.plot_data)
        self.UpperButtons1.addWidget(self.plot_button)

        self.Save_button = QPushButton("Save Configurations", self)
        #self.plot_button.clicked.connect(self.plot_data)
        self.UpperButtons2.addWidget(self.Save_button)

        self.Smooth_button = QPushButton("Smooth", self)
        self.Smooth_button.clicked.connect(self.smooth_data)
        self.UpperButtons2.addWidget(self.Smooth_button)  

        self.UpperButtonsParent.addLayout (self.UpperButtons1)
        self.UpperButtonsParent.addLayout (self.UpperButtons2)
        self.UpperButtonsParentContainer.setLayout (self.UpperButtonsParent)
        self.SideLayout.addWidget(self.UpperButtonsParentContainer, stretch=1) 
             

        #x0_x1
        self.x0Value = {}
        self.x1Value = {}
        self.x0cell = {}
        self.x1cell = {}

        # Smooth Options
        self.SmoothOptionsContainer = QGroupBox("Smoothing Options")
        self.SmoothOptionsContainer.setStyleSheet("background-color:white;")
        self.SmoothOptionsLayout = QVBoxLayout()
        
        self.moving_avg_radio = QRadioButton("Moving Average")
        self.moving_med_radio = QRadioButton("Moving Median")
        self.moving_avg_radio.setChecked(True)  # Default choice
        
        self.window_size_label = QLabel("Window Size:")
        self.window_size_input = QLineEdit("1")
        
        self.SmoothOptionsLayout.addWidget(self.moving_avg_radio)
        self.SmoothOptionsLayout.addWidget(self.moving_med_radio)
        self.SmoothOptionsLayout.addWidget(self.window_size_label)
        self.SmoothOptionsLayout.addWidget(self.window_size_input)
        
        self.SmoothOptionsContainer.setLayout(self.SmoothOptionsLayout)

        #Noise

        self.NoiseSectionContainer = QGroupBox ("Noise Fit and Reduction")
        self.NoiseSectionContainer.setStyleSheet("background-color:white;")
        self.NoiseSection = QVBoxLayout ()
        
        self.Noisex0Lable = QLabel("X0")
        self.Noisex1Lable = QLabel("X1")
        self.x0cell ["Noise"] = QLineEdit ()
        self.x1cell ["Noise"] = QLineEdit ()

        self.noise_button = QPushButton("Select Noise", self)
        self.noise_button.clicked.connect(lambda: self.select_roi("Noise", self.FitNum.currentText()))

        self.NoiseFit = QFitWidget ()
        self.NoiseFit.SetFit ("exponential_0")
        

        self.noiseFit_button = QPushButton("Plot Noise Fit", self)
        self.noiseFit_button.clicked.connect(lambda: self.PlotFit (self.NoiseFit.fitting_result.a, self.NoiseFit.FitType.currentText(), "Noise Fit"))
        
        
        self.SelectRoi = QHBoxLayout ()
        self.SelectRoi.addWidget (self.noise_button)
        self.SelectRoi.addWidget (self.Noisex0Lable)
        self.SelectRoi.addWidget (self.x0cell ["Noise"])
        self.SelectRoi.addWidget (self.Noisex1Lable)
        self.SelectRoi.addWidget (self.x1cell ["Noise"])
        self.NoiseSection.addLayout (self.SelectRoi)
        self.NoiseSection.addWidget(self.NoiseFit)
        self.NoiseSection.addWidget(self.noiseFit_button)
        self.NoiseSectionContainer.setLayout (self.NoiseSection)       

        #MainData

        self.MainDataContainer = QGroupBox ("Main Data Fit")
        self.MainDataContainer.setStyleSheet("background-color:white;")
        self.MainData = QGridLayout ()
        self.MainDataProp = QVBoxLayout ()
        self.FitNum = QComboBox ()
        for i in range (10):
            self.FitNum.addItem (str (i + 1))
        self.MainDatax0Lable = QLabel("X0")
        self.MainDatax1Lable = QLabel("X1")
        self.x0cell ["Main"] = QLineEdit ()
        self.x1cell ["Main"] = QLineEdit ()
        self.main_select_button = QPushButton("Select Data", self)
        self.main_select_button.clicked.connect(lambda: self.select_roi("Main", self.FitNum.currentText()))
        self.FitNum.activated.connect (self.update_labels)

        self.MainFit = QFitWidget ()
        self.MainFit.SetFit ("normal_0")
        self.MainFit_button = QPushButton("Plot Noise Fit", self)
        self.MainFit_button.clicked.connect(lambda: self.PlotFit (self.MainFit.fitting_result.a, self.MainFit.FitType.currentText(), self.FitNum.currentText()))

        #Export

        self.ExportBox = QGroupBox ()
        self.ExportBox.setStyleSheet("background-color:white;")
        self.Export = QVBoxLayout ()

        self.ExportAll_button = QPushButton("Export Data", self)
        self.ExportAll_button.clicked.connect(self.ExportResults)
        self.ExportAll_button.setStyleSheet("background-color:#156082;color:white;")
        self.AddLog_button = QPushButton("Add Line To Log File", self)
        self.AddLog_button.clicked.connect(self.AddLineLog)
        self.AddLog_button.setStyleSheet("background-color:#156082;color:white;")
        self.Export.addWidget (self.ExportAll_button)
        self.Export.addWidget (self.AddLog_button)
        self.ExportBox.setLayout (self.Export)
        

        #Connect All

        self.MainDataSection = QHBoxLayout ()
        self.MainDataSection.addWidget (self.main_select_button)
        self.MainDataSection.addWidget (self.MainDatax0Lable)
        self.MainDataSection.addWidget (self.x0cell ["Main"])
        self.MainDataSection.addWidget (self.MainDatax1Lable)
        self.MainDataSection.addWidget (self.x1cell ["Main"])
        self.MainDataProp.addLayout (self.MainDataSection)
        self.MainDataProp.addWidget (self.MainFit)
        self.MainDataProp.addWidget (self.MainFit_button)
        self.MainDataProp.addWidget (self.ExportBox)

        self.MainData.addWidget(self.FitNum, 0, 0)
        self.MainData.addLayout(self.MainDataProp, 1, 0)
        self.MainData.setColumnStretch(0,0)
        self.MainData.setColumnStretch(1,50)
        self.MainDataContainer.setLayout (self.MainData) 

        # self.roi_buttons_layout.addWidget(self.noise_button)

        self.SideLayout.addWidget(self.SmoothOptionsContainer, stretch=2)
        self.SideLayout.addWidget(self.NoiseSectionContainer, stretch=5)
        self.SideLayout.addWidget(self.MainDataContainer, stretch=5)
        self.ScrollSide = QScrollArea(widgetResizable=True)
        
        #figure
        self.plotData = {}

        self.Headlines = QHBoxLayout ()
        self.HeadlineLable = QLabel("Graph Headline: ")
        self.HeadlineLable.setStyleSheet("color:white;")
        self.Headline = QLineEdit ()
        self.Headline.setStyleSheet("background-color:white;")
        self.xLableLable = QLabel("x Lable: ")
        self.xLableLable.setStyleSheet("color:white;")
        self.xLable = QLineEdit ()
        self.xLable.setStyleSheet("background-color:white;")
        self.yLableLable = QLabel("y Lable: ")
        self.yLableLable.setStyleSheet("color:white;")
        self.yLable = QLineEdit ()
        self.yLable.setStyleSheet("background-color:white;")
        self.Headlines.addWidget (self.HeadlineLable)
        self.Headlines.addWidget (self.Headline)
        self.Headlines.addWidget (self.xLableLable)
        self.Headlines.addWidget (self.xLable)
        self.Headlines.addWidget (self.yLableLable)
        self.Headlines.addWidget (self.yLable)

        self.Columns = QHBoxLayout ()
        self.xColumnLable = QLabel("x Column: ")
        self.xColumnLable.setStyleSheet("color:white;")
        self.xColumn = QLineEdit ()
        self.xColumn.setStyleSheet("background-color:white;")
        self.xErrColumnLable = QLabel("x Errors Column: ")
        self.xErrColumnLable.setStyleSheet("color:white;")
        self.xErrColumn = QLineEdit ()
        self.xErrColumn.setStyleSheet("background-color:white;")
        self.yColumnLable = QLabel("y Column: ")
        self.yColumnLable.setStyleSheet("color:white;")
        self.yColumn = QLineEdit ()
        self.yColumn.setStyleSheet("background-color:white;")
        self.yErrColumnLable = QLabel("y Errors Column: ")
        self.yErrColumnLable.setStyleSheet("color:white;")
        self.yErrColumn = QLineEdit ()
        self.yErrColumn.setStyleSheet("background-color:white;")

        self.Columns.addWidget (self.xColumnLable)
        self.Columns.addWidget (self.xColumn)
        self.Columns.addWidget (self.xErrColumnLable)
        self.Columns.addWidget (self.xErrColumn)
        self.Columns.addWidget (self.yColumnLable)
        self.Columns.addWidget (self.yColumn)
        self.Columns.addWidget (self.yErrColumnLable)
        self.Columns.addWidget (self.yErrColumn) 

        self.figurparantLayout = QVBoxLayout ()
        self.figurparant = QGroupBox ()
        self.figure = Figure(figsize=(5, 4), dpi=256)
        self.canvas = FigureCanvas(self.figure)
        self.figurparantLayout.addLayout (self.Headlines)
        self.figurparantLayout.addLayout (self.Columns)
        self.figurparantLayout.addWidget (self.canvas)
        self.figurparant.setLayout (self.figurparantLayout)
        self.Side.setLayout (self.SideLayout)
        self.ScrollSide.setWidget (self.Side)
        # 
        self.layout.addWidget(self.ScrollSide, stretch=1)
        self.layout.addWidget(self.figurparant, stretch=5)
        
        self.ax = self.figure.add_subplot(111)
        self.roi_selector = None
        self.rois = {}

    def load_data(self):
        options = QFileDialog.Options()
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if self.file_path:
            try:
                self.data = pd.read_excel(self.file_path)
                print("Data loaded successfully.")
                self.slash = "/"
                if (self.file_path.rfind (self.slash) == -1):
                    self.slash = "\\"
                self.dir = self.file_path [0:self.file_path.rfind (self.slash)]
                self.fileName = self.file_path [self.file_path.rfind (self.slash) + 1:self.file_path.rfind (".")]
            except Exception as e:
                print(f"Error loading data: {e}")

    def plot_data(self):
        if hasattr(self, 'data'):            
            try:
                data_sort = self.data.sort_values(by=[self.xColumn.text ()])
                x = data_sort[self.xColumn.text ()].to_numpy ()
                y = data_sort[self.yColumn.text ()].to_numpy ()
                x_err = data_sort[self.xErrColumn.text ()].to_numpy ()
                y_err = data_sort[self.yErrColumn.text ()].to_numpy ()
            except KeyError:
                try:
                    x = self.data['Bin number'].to_numpy ()
                    y = self.data['Number of counts'].to_numpy ()
                    x_err = np.sqrt (np.ones (len (x)) * 1/12)
                    y_err = np.sqrt ((y * 0.05)**2 + np.ones (len (y)) * 1/12)
                except KeyError as e:
                    QMessageBox.critical(self, "Error", f"Invalid window size: {e}")
                    return

            self.x0Value = {}
            self.x1Value = {}
            self.plotData ["Raw Data"] = np.column_stack((x, y))
            self.plotData ["Raw Data Err"] = np.column_stack((x_err, y_err))
            self.plotData ["Smooth Data"] = np.column_stack((x, y))
            self.plotData ["Smooth Data Err"] = np.column_stack((x_err, y_err))
            self.plotData ["Noise Fit"] = np.column_stack((x, np.zeros (len (x))))

            self.x0cell ["Noise"].setText (str (self.plotData ["Smooth Data"] [0,0]))
            self.x1cell ["Noise"].setText (str (self.plotData ["Smooth Data"] [len (self.plotData ["Smooth Data"] [:,0]) - 1,0])) 
            self.update_labels()

            self.ax.clear()
            
            self.ax.scatter(self.plotData ["Raw Data"] [:, 0], self.plotData ["Raw Data"] [:, 1], label="Raw Data", marker = 'o')
            if (len (self.plotData ["Raw Data"] [:, 0]) < 100):
                self.ax.errorbar(self.plotData ["Raw Data"] [:, 0], self.plotData ["Raw Data"] [:, 1], yerr=self.plotData ["Raw Data Err"] [:, 1], xerr=self.plotData ["Raw Data Err"] [:, 0], fmt="o")
            self.ax.set_title(self.Headline.text ())
            self.ax.set_xlabel(self.xLable.text ())
            self.ax.set_ylabel(self.yLable.text ())
            self.canvas.draw()
            self.setupData ()
        else:
            print("No data loaded. Please load an Excel file first.")

    def smooth_data(self):
        if not hasattr(self, 'data'):
            QMessageBox.critical(self, "Error", "No data loaded.")
            return

        # Get the smoothing method
        if self.moving_avg_radio.isChecked():
            smoothing_method = 'average'
        elif self.moving_med_radio.isChecked():
            smoothing_method = 'median'
        else:
            QMessageBox.critical(self, "Error", "No smoothing method selected.")
            return

        # Get the window size
        try:
            window_size = int(self.window_size_input.text())
            if window_size <= 0:
                raise ValueError("Window size must be a positive integer.")
            if window_size > len (self.plotData ["Raw Data"] [:, 0]):
                raise ValueError("Window size must be a smaller than the length of the dataset.")
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid window size: {e}")
            return

        data = self.plotData ["Raw Data"]

        # Apply smoothing
        if smoothing_method == 'average':
            self.smoothed_data, self.smoothed_data_err = self.moving_average(self.plotData ["Raw Data"].astype (float), window_size)
        elif smoothing_method == 'median':
            self.smoothed_data, self.smoothed_data_err = self.moving_median(self.plotData ["Raw Data"].astype (float), window_size)

        self.plotData ["Smooth Data"] = self.smoothed_data
        self.plotData ["Smooth Data Err"] = np.sqrt (self.smoothed_data_err**2)
        
        # with open(self.file_path [0:self.file_path.rfind (slash)] + "/debug_zevel.txt", "w") as debug_file:
        #     np.savetxt(debug_file, self.plotData ["Smooth Data"])
        self.plotData ["Noise Fit"] = np.column_stack((self.smoothed_data [:,0], np.zeros (len (self.smoothed_data [:,0]))))
        self.setupData ()

        self.setupData ()
        # Plot smoothed data
        self.ax.clear()
        self.ax.scatter(self.plotData ["Raw Data"] [:, 0], self.plotData ["Raw Data"] [:, 1], label="Raw Data", alpha=0.5, marker = 'o')
        self.ax.scatter(self.plotData ["Smooth Data"] [:, 0], self.plotData ["Smooth Data"] [:, 1], label="Smooth Data", color='g', marker = 'o')
        if (len (self.plotData ["Smooth Data"] [:, 0]) < 100):
            self.ax.errorbar(self.plotData ["Smooth Data"] [:, 0], self.plotData ["Smooth Data"] [:, 1], yerr=self.plotData ["Smooth Data Err"] [:, 1], xerr=self.plotData ["Smooth Data Err"] [:, 0], fmt="o")
        self.ax.set_title(self.Headline.text ())
        self.ax.set_xlabel(self.xLable.text ())
        self.ax.set_ylabel(self.yLable.text ())
        self.ax.legend()
        self.canvas.draw()
        # plt.clear()
        # plt.plot(x, y, label='Original Data', alpha=0.5)
        # plt.plot(self.smoothed_data[:, 0], self.smoothed_data[:, 1], label=f'{smoothing_method.capitalize()} (window size={window_size})', color='r')
        # plt.legend()
        # plt.show ()

    def moving_average(self, arr, window_size):
        """Compute the moving average with a specified window size."""
        half_window = window_size // 2
        arr_avg = np.zeros_like(arr).astype (float)
        arr_avg[:, 0] = arr[:, 0]
        arr_avg_err = np.zeros_like(arr).astype (float)
        arr_avg_err[:, 0] = arr[:, 0]
        
        # Edge cases
        for i in range(half_window):
            arr_avg[i, 1] = np.mean(arr[:i + half_window + 1, 1])
            arr_avg[-(i + 1), 1] = np.mean(arr[-(i + half_window + 1):, 1])
            arr_avg_err[i, 1] = np.std(arr[:i + half_window + 1, 1])/np.sqrt (len (arr[:i + half_window + 1, 1]))
            arr_avg_err[-(i + 1), 1] = np.std(arr[-(i + half_window + 1):, 1])/np.sqrt (len (arr[:i + half_window + 1, 1]))
        
        # Normal cases
        for i in range(half_window, len(arr) - half_window):
            arr_avg[i, 1] = np.mean(arr[i - half_window:i + half_window + 1, 1])
            arr_avg_err[i, 1] = np.std(arr[:i + half_window + 1, 1])/np.sqrt (len (arr[:i + half_window + 1, 1]))
        
        return arr_avg, arr_avg_err

    def moving_median(self, arr, window_size):
        """Compute the moving median with a specified window size."""
        half_window = window_size // 2
        arr_med = np.zeros_like(arr).astype (float)
        arr_med[:, 0] = arr[:, 0]
        
        # Edge cases
        for i in range(half_window):
            arr_med[i, 1] = np.median(arr[:i + half_window + 1, 1])
            arr_med[-(i + 1), 1] = np.median(arr[-(i + half_window + 1):, 1])
        
        # Normal cases
        for i in range(half_window, len(arr) - half_window):
            arr_med[i, 1] = np.median(arr[i - half_window:i + half_window + 1, 1])
        
        return arr_med, arr_med*0.05

    def select_roi(self, roi_name, fit_num):
        if not hasattr(self, 'data'):
            QMessageBox.critical(self, "Error", "No data loaded.")
            return

        if self.roi_selector is not None:
            self.roi_selector.disconnect_events()

        def onselect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            x0, x1 = round(x0), round(x1)
            self.rois[roi_name] = (x0, x1)
            self.x0cell [roi_name].setText (str (x0))
            self.x1cell [roi_name].setText (str (x1))
            if (roi_name != "Noise"):
                self.x0Value [fit_num] = (str (x0))
                self.x1Value [fit_num] = (str (x1))
            self.setupData ()
            
        def toggle_selector(event):
            if event.key == 't':
                self.roi_selector.set_active(True)

        self.roi_selector = RectangleSelector(self.ax, onselect, useblit=True,
                                              button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                              interactive=True)
        self.canvas.mpl_connect('key_press_event', toggle_selector)
        self.canvas.draw()
        self.setupData ()

    def update_labels(self):
        try:
            self.x0cell ["Main"].setText (self.x0Value [self.FitNum.currentText()])
            self.x1cell ["Main"].setText (self.x1Value [self.FitNum.currentText()])
        except KeyError:
            print ("err")
            self.x0cell ["Main"].setText (str (self.plotData ["Smooth Data"] [0,0]))
            self.x1cell ["Main"].setText (str (self.plotData ["Smooth Data"] [len (self.plotData ["Smooth Data"] [:,0]) - 1,0]))
        self.select_roi("None", "-1")
        self.setupData ()
        
        
    def setupData (self):
        print ("try")
        try:
            # print (self.x0cell ["Noise"].text ())
            self.NoiseFit.SetData (self.plotData ["Smooth Data"] [int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x0cell ["Noise"].text ())) [0]):int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x1cell ["Noise"].text ())) [0]),:], self.plotData ["Smooth Data Err"] [int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x0cell ["Noise"].text ())) [0]):int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x1cell ["Noise"].text ())) [0]),:])
            # print ("noise")
        except ValueError as e:
            print (e)
        except AttributeError as e:
            print (e)
        try:
            noise_data = self.plotData ["Smooth Data"] [int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x0cell ["Main"].text ())) [0]):int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x1cell ["Main"].text ())) [0]),:] - np.column_stack((np.zeros(int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x1cell ["Main"].text ())) [0]) - int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x0cell ["Main"].text ()))[0])),  self.plotData ["Noise Fit"] [int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x0cell ["Main"].text ())) [0]):int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x1cell ["Main"].text ())) [0]), 1]))
            self.MainFit.SetData (noise_data, self.plotData ["Smooth Data Err"] [int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x0cell ["Main"].text ())) [0]):int (np.where (self.plotData ["Smooth Data"] [:,0] == float (self.x1cell ["Main"].text ())) [0]),:])
            # print ("main")
        except ValueError as e:
            print (e)
            pass
        except AttributeError as e:
            print (e)
            pass
        

    def PlotFit (self, a, fit, cell):
        fit_list = "fitting_functions_list." + fit
        if (len (self.plotData ["Smooth Data"] [:, 0]) < 10000):
            x = np.linspace(max (self.plotData ["Smooth Data"] [:, 0]), min (self.plotData ["Smooth Data"] [:, 0]), 10000)
        else:
            x = self.plotData ["Smooth Data"] [:, 0]
        if ("Noise Fit" in self.plotData.keys ()):
            self.plotData ["Noise Fit_plot"] = np.array ([x, x*0]).T
        if (fit_list.find ("_0") != -1):
            fit_list = fit
        if (cell == "Noise Fit"):
            print (a)
            exec ("self.plotData [cell + \"_plot\"] = np.array ([x, " + fit_list + " (a, x)]).T")
            exec ("self.plotData [cell] = np.array ([self.plotData [\"Smooth Data\"] [:, 0], " + fit_list + " (a, self.plotData [\"Smooth Data\"] [:, 0])]).T")
            self.plotData [cell + "_Heder"] = {}
            self.plotData [cell + "_Heder"] ["Values"] = self.NoiseFit.fitting_result
        if (cell != "Noise Fit"):
            exec ("self.plotData [cell + \"_plot\"] = np.array ([x, " + fit_list + " (a, x)]).T")
            exec ("self.plotData [cell] = np.array ([self.plotData [\"Smooth Data\"] [:, 0], " + fit_list + " (a, self.plotData [\"Smooth Data\"] [:, 0])]).T")
            self.plotData [cell + "_Heder"] = {}
            self.plotData [cell + "_Heder"] ["Values"] = self.MainFit.fitting_result
            self.plotData [cell + "_Heder"] ["Lable"] = self.MainFit.fit + ":\n"
            for a in self.MainFit.fitting_result.a:
                self.plotData [cell + "_Heder"] ["Lable"] += str (round(a, 2)) + ", "
            self.plotData [cell + "_Heder"] ["Lable"] += "\n"
            for aerr in self.MainFit.fitting_result.aerr:
                self.plotData [cell + "_Heder"] ["Lable"] += str (round(aerr, 2)) + ", "
        # self.setupData ()
        # Plot smoothed data
        self.ax.clear()
        # self.ax.plot(self.plotData ["Raw Data"] [:, 0], self.plotData ["Raw Data"] [:, 1], label="Raw Data", alpha=0.3)
        print (len (self.plotData ["Noise Fit"] [:, 0]))
        print (len (self.plotData ["Smooth Data"] [:, 0]))
        if (("Noise Fit" in self.plotData.keys ()) and ("Noise Fit_plot" in self.plotData.keys ())):
            self.y_noise_plot = self.plotData ["Noise Fit_plot"] [:, 1]
            self.y_noise_plot [(self.y_noise_plot > max (self.plotData ["Smooth Data"] [:, 1]))] = 0
            if (cell == "Noise Fit"):
                self.noiseShow = True
                # self.MainFit.SetData (np.array ([self.smoothed_dat [int (self.x0cell ["Main"].text ()):int (self.x1cell ["Main"].text ()),0], self.smoothed_data [int (self.x0cell ["Main"].text ()):int (self.x1cell ["Main"].text ()),1] - self.plotData ["Noise Fit"] [:, 1]]))
                self.ax.scatter(self.plotData ["Smooth Data"] [:, 0], self.plotData ["Smooth Data"] [:, 1], label="Smooth Data", alpha=0.5, marker = 'o', color='g')
                if (len (self.plotData ["Smooth Data"] [:, 0]) < 100):
                    self.ax.errorbar(self.plotData ["Smooth Data"] [:, 0], self.plotData ["Smooth Data"] [:, 1], yerr=self.plotData ["Smooth Data Err"] [:, 1], xerr=self.plotData ["Smooth Data Err"] [:, 0], fmt="o", color='grey', alpha=0.5)
                self.ax.plot(self.plotData ["Noise Fit_plot"] [:, 0], self.y_noise_plot, label="Noise Fit", color='orange')
                self.ax.set_title(self.Headline.text ())
                self.ax.set_xlabel(self.xLable.text ())
                self.ax.set_ylabel(self.yLable.text ())
                self.ax.legend()
                self.canvas.draw()
                return
            self.noiseShow = False
            smooth_pos = self.plotData ["Smooth Data"] [:, 1]  - self.plotData ["Noise Fit"] [:, 1]
            smooth_pos [smooth_pos < 0] = 0
            self.ax.scatter(self.plotData ["Smooth Data"] [:, 0], smooth_pos, label="Smooth Data", alpha=0.5, marker = 'o', color='grey')
            if (len (self.plotData ["Smooth Data"] [:, 0]) < 100):
                    self.ax.errorbar(self.plotData ["Smooth Data"] [:, 0], smooth_pos, yerr=self.plotData ["Smooth Data Err"] [:, 1], xerr=self.plotData ["Smooth Data Err"] [:, 0], fmt="o", color='grey', alpha=0.5)
            # self.ax.plot(self.plotData ["Noise Fit"] [:, 0], self.y_noise_plot, label="Noise Fit", color='orange')
        for i in range (10):
            try:
                self.ax.plot(self.plotData [str (i + 1) + "_plot"] [:, 0], self.plotData [str (i + 1) + "_plot"] [:, 1], label="Fit " + str (i + 1) + ": " + self.plotData [str (i + 1) + "_Heder"] ["Lable"])
            except KeyError:
                continue
        self.ax.set_title(self.Headline.text ())
        self.ax.set_xlabel(self.xLable.text ())
        self.ax.set_ylabel(self.yLable.text ())
        self.ax.legend()
        self.canvas.draw()

    def ExportResults (self):
        try:
            os.mkdir (self.dir + self.slash + "Results" + self.slash)
        except OSError:
            pass
        with open (self.dir + self.slash + "Results" + self.slash + self.fileName + ".txt", "w") as ResultsFile:
            ResultsFile.write ("========Fitting Data========\n\n\n")
            plt.cla()
            
            if (self.noiseShow):
                plt.scatter(self.plotData ["Smooth Data"] [:, 0], self.plotData ["Smooth Data"] [:, 1], label="Smooth Data", alpha=0.5, marker = 'o', color='grey')
                if (len (self.plotData ["Smooth Data"] [:, 0]) < 100):
                        plt.errorbar(self.plotData ["Smooth Data"] [:, 0], smooth_pos, yerr=self.plotData ["Smooth Data Err"] [:, 1], xerr=self.plotData ["Smooth Data"] [:, 0], fmt="o", color='grey')
                plt.plot(self.plotData ["Noise Fit_plot"] [:, 0], self.y_noise_plot, label="Noise Fit", color='orange')                    
                plt.title(self.Headline.text ())
                plt.xlabel(self.xLable.text ())
                plt.ylabel(self.yLable.text ())   
                plt.legend()
                plt.savefig (self.dir + self.slash + "Results" + self.slash + self.fileName + ".eps" , format='eps')
                return
            smooth_pos = self.plotData ["Smooth Data"] [:, 1]  - self.plotData ["Noise Fit"] [:, 1]
            smooth_pos [smooth_pos < 0] = 0
            plt.scatter(self.plotData ["Smooth Data"] [:, 0], smooth_pos, label="Smooth Data", alpha=0.5, marker = 'o', color='grey')
            if (len (self.plotData ["Smooth Data"] [:, 0]) < 100):
                    plt.errorbar(self.plotData ["Smooth Data"] [:, 0], smooth_pos, yerr=self.plotData ["Smooth Data Err"] [:, 1], xerr=self.plotData ["Smooth Data"] [:, 0], fmt="o", color='grey')
            if (("Noise Fit" in self.plotData.keys ()) and ("Noise Fit_Heder" in self.plotData.keys ())):
                ResultsFile.write ("==Noise Fit==" + str (self.x0cell ["Noise"].text ()) + " -> " + str (self.x1cell ["Noise"].text ()) + "\n\n")
                ResultsFile.write (str (self.plotData ["Noise Fit_Heder"] ["Values"]))
                ResultsFile.write ("\n\n")

            for i in range (10):
                try:
                    plt.plot(self.plotData [str (i + 1) + "_plot"] [:, 0], self.plotData [str (i + 1) + "_plot"] [:, 1], label="Fit " + str (i + 1) + ": " + self.plotData [str (i + 1) + "_Heder"] ["Lable"])
                    ResultsFile.write ("==Fit " + str (i + 1) + "==" + str (self.x0Value [str (i + 1)]) + " -> " + str (self.x1Value [str (i + 1)]) + "\n\n")
                    ResultsFile.write (str (self.plotData [str (i + 1) + "_Heder"] ["Values"]))
                    ResultsFile.write ("\n\n")
                except KeyError:
                    continue
            plt.title(self.Headline.text ())
            plt.xlabel(self.xLable.text ())
            plt.ylabel(self.yLable.text ())   
            plt.legend()
            plt.savefig (self.dir + self.slash + "Results" + self.slash + self.fileName + ".eps" , format='eps')
            #

    def AddLineLog (self):
        try:
            os.mkdir (self.dir + self.slash + "Results" + self.slash)
        except OSError:
            pass
        logname = self.dir + self.slash + "Results" + self.slash
        if (not os.path.isfile(logname + "log.csv")):
            with open (self.dir + self.slash + "Results" + self.slash + "log.csv", "w") as LogFile:
                print (os.path.isfile(logname))
                LogFile.write ("File name,,,")
                for i in range (10):
                    for j in range (6):
                        LogFile.write ("fit " + str(i + 1) + " - a [" + str (j) + "],fit " + str(i + 1) + " - a Error [" + str (j) + "],")
                    LogFile.write ("Chi Squared red,P Prob,")
                LogFile.write ("\n")
        try:
            with open (self.dir + self.slash + "Results" + self.slash + "log.csv", "a") as LogFile:
                LogFile.write (self.fileName + "File name,,,")
                for i in range (10):
                    for j in range (6):
                        try:
                            LogFile.write (str (self.plotData [str (i + 1) + "_Heder"] ["Values"].a [j]) + "," + str (self.plotData [str (i + 1) + "_Heder"] ["Values"].aerr [j]) + ",")
                        except KeyError:
                            LogFile.write ("0,0,")
                        except IndexError:
                            LogFile.write ("0,0,")
                    try:
                        LogFile.write (str (self.plotData [str (i + 1) + "_Heder"] ["Values"].chi2_reduced) + "," + str (self.plotData [str (i + 1) + "_Heder"] ["Values"].p_probability) + ",")
                    except KeyError:
                        LogFile.write ("0,0,")
                    except IndexError:
                        LogFile.write ("0,0,")
                    
                LogFile.write ("\n")
        except PermissionError as e:
            QMessageBox.critical(self, "Error, Log file is open !\nPlease close it before adding new lines.", f"Invalid window size: {e}")
            return

class QFitWidget (QGroupBox):
    def __init__(self):
        super().__init__()
        self.tot = QVBoxLayout ()
        self.setStyleSheet("background-color:#dbdbdb;")

        self.initValsBox = {}
        self.Fits = ["linear", "constant", "parabolic", "straight_power", "inverse_power", "hyperbolic", "exponential", "exponential_0", "cos", "sin", "normal", "normal_0", "poisson", "polynomial", "cos2_0"]
        self.FitType = QComboBox ()
        self.FitType.setStyleSheet("background-color:#708ef0;")
        for fitType in self.Fits:
            self.FitType.addItem (fitType)
        self.FitType.activated.connect (self.SelectFit)

        self.syntax = QLabel ()
        # exec ("self.syntax.setText (fitting_functions_list." + self.fit + ".syntax)")
        self.initVals = QVBoxLayout ()
        line = QHBoxLayout ()
        self.nLable = QLabel ("n = ")
        self.polyn = QSpinBox()
        validpoly = QIntValidator(0, 10)
        # self.polyn.setValidator(validpoly)
        self.polyn.setStyleSheet("background-color:#ffffff;")
        line.addWidget (self.FitType)
        line.addWidget (self.nLable)
        line.addWidget (self.polyn)
        self.initVals.addLayout (line)
        self.initVals.addWidget (self.syntax)
        
        for i in range (6):
            line = QHBoxLayout ()
            self.initValsBox [str (i)] = QLineEdit ()
            valid = QDoubleValidator ()
            self.initValsBox [str (i)].setValidator(valid)
            self.initValsBox [str (i)].setStyleSheet("background-color:#ffffff;")
            self.initValsBox [str (i)].setText ("1.0")
            line.addWidget (QLabel ("a[" + str (i) + "] = "))
            line.addWidget (self.initValsBox [str (i)])
            self.initVals.addLayout (line)

        self.DoFit = QPushButton("Fit !", self)
        self.DoFit.clicked.connect(self.DoFitFunc)        
        
        self.tot.addLayout (self.initVals)         
        self.tot.addWidget (self.DoFit)
        self.setLayout (self.tot)
        self.SelectFit ()
    
    def SetData (self, Data, Err):
        print ("fill data")
        self.data = Data
        self.dataErr = self.data

    def SetFit (self, fit1):
        self.FitType.setCurrentIndex(self.Fits.index(fit1))
        self.SelectFit ()

    def SelectFit (self):
        self.fit = self.FitType.currentText()
        self.n = 0
        fit_list = "fitting_functions_list." + self.fit
        if (fit_list.find ("_0") != -1):
            fit_list = self.fit
        try:
            exec ("self.n = " + fit_list + ".n")
            exec ("self.syntax.setText (" + fit_list + ".syntax)")
        except AttributeError:
            print ("Fit " + self.fit + " is not recognized")
        # print (fitting_functions_list.linear.n)
        try:
            for i in range (6):
                self.initValsBox [str (i)].setEnabled(False)
                self.initValsBox [str (i)].setStyleSheet("background-color:#dbdbdb;")

            for i in range (self.n):
                self.initValsBox [str (i)].setEnabled(True)
                self.initValsBox [str (i)].setStyleSheet("background-color:#ffffff;")
        except KeyError:
            print ("exit")
            return
        
    def DoFitFunc (self):
        print ("fit")
        fit_list = "fitting_functions_list." + self.fit
        if (fit_list.find ("_0") != -1):
            fit_list = self.fit
        self.SelectFit ()
        a = []
        for i in range (self.n):
            a.append (float (self.initValsBox [str (i)].text ()))

        data_dict = {}
        data_dict ["x"] = self.data[:, 0]
        data_dict ["xErr"] = self.dataErr[:, 0]
        data_dict ["y"] = self.data[:, 1]
        data_dict ["yErr"] = self.dataErr[:, 1]
        self.fitting_data = FittingData (data_dict)

        self.fitting_data.x_column = "x"
        self.fitting_data.xerr_column = "xErr"
        self.fitting_data.y_column = "y"
        self.fitting_data.yerr_column = "yErr"
        exec ("self.fitting_result = fit(self.fitting_data, " + fit_list + ", np.array (a))")  # Do the actual fitting
        print (self.fitting_result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    size = screen.size()
    print (size)
    FontScale = int (size.width() / 1980)
    AppFont = QFont("Arial", FontScale * 16)
    app.setFont (AppFont)
    window = DataPlotterApp()
    window.show()
    sys.exit(app.exec_())
