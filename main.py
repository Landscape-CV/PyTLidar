"""
Python adaptation and extension of TREEQSM:

% This file is part of TREEQSM.
%
% TREEQSM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% TREEQSM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with TREEQSM.  If not, see <http://www.gnu.org/licenses/>.

Version: 0.0.1
Date: 9 Feb 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""


try:
    from .treeqsm import treeqsm
    from .Utils.define_input import define_input
    from .Utils.Utils import load_point_cloud
    from .Utils import Utils
    from .plotting.point_cloud_plotting import point_cloud_plotting
    from .plotting.qsm_plotting import qsm_plotting
except ImportError:
    from treeqsm import treeqsm
    from Utils.define_input import define_input
    from Utils.Utils import load_point_cloud
    import Utils.Utils as Utils
    from plotting.point_cloud_plotting import point_cloud_plotting
    from plotting.qsm_plotting import qsm_plotting
import os
from PySide6.QtCore import QObject,QThread,Signal,Qt,QUrl,QProcess
from PySide6.QtWidgets import QWidget,QGridLayout,QVBoxLayout,QLabel,QMainWindow,QPushButton,QApplication,QTextEdit,QToolButton,QComboBox,QHBoxLayout,QSlider,QFileDialog,QMessageBox,QTableWidget,QTableWidgetItem, QCheckBox,QRadioButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QPixmap
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView 

import numpy as np
import multiprocessing as mp
import sys
# from plotly.graph_objects import Figure, Scatter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
import time

class QSMWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.stacked_widget = QStackedWidget(self)
        self.setWindowTitle("TreeQSM")
        self.setGeometry(100, 100, 750, 500)
        # layout.addWidget(self.stacked_widget)

        # Create a button to start the batch processing
        self.button= QPushButton("Batch Processing (Select Folder)", self)
        self.button.clicked.connect(self.start_batch_processing)
        self.button.setGeometry(50, 200, 300, 50)

        # Create a button to start the single file processing
        self.button2 = QPushButton("Single File Processing (Select File)", self)
        self.button2.clicked.connect(self.start_single_file_processing)
        self.button2.setGeometry(400, 200, 300, 50)


        #TextEdit for intensity threshold
        Label = QLabel("Intensity Threshold:",self)
        Label.setGeometry(50, 10, 200, 30)
        
        self.TextEdit = QTextEdit(self)
        self.TextEdit.setGeometry(100, 50, 200, 30)
        self.TextEdit.setText("0")
        self.TextEdit.setToolTip("Set the Intensity threshold, this will automically filter out any points with intensity lower than this value. The default value is 0.")
        self.TextEdit.setToolTipDuration(10000)
        
        #TextEdit for number of min PatchDiam to test
        Label2 = QLabel("Min PatchDiam to Test:",self)
        Label2.setToolTip("""The Min PatchDiam is the minimum of the initial cover sets that will cover the tree shape on the second pass.\nThese cover sets will likely produce the upper branch shapes\nValues that are too large may not properly capture the detail in the branches\nThese values should be smaller than your initial patch diameter\nSee Instructions for how to fill in this field""")
        Label2.setToolTipDuration(10000)
        Label2.setGeometry(50, 100, 250, 30)
        self.TextEdit2 = QTextEdit(self) 
        self.TextEdit2.setGeometry(100, 150, 200, 30)
        self.TextEdit2.setText("1")
        self.TextEdit2.setToolTip("""The Min PatchDiam is the minimum of the initial cover sets that will cover the tree shape on the second pass.\nThese cover sets will likely produce the upper branch shapes\nValues that are too large may not properly capture the detail in the branches\nThese values should be smaller than your initial patch diameter\nSee Instructions for how to fill in this field""")
        self.TextEdit2.setToolTipDuration(10000)

        


        #TextEdit for number of max PatchDiam to test
        Label3 = QLabel("Max PatchDiam to Test:",self)
        Label3.setGeometry(400, 100, 250, 30)
        Label3.setToolTip("""The Max PatchDiam is the maximum of the initial cover sets that will cover the tree shape on the second pass.\nThese cover sets will likely produce the shape of the trunk\nValues that are too small may cause performance issues and produce incorrect segment\nThese values should be larger than your initial patch diameter\nSee Instructions for how to fill in this field""")
        Label3.setToolTipDuration(10000)
        self.TextEdit3 = QTextEdit(self)
        self.TextEdit3.setText("1")
        self.TextEdit3.setGeometry(450, 150, 200, 30)
        self.TextEdit3.setToolTip("""The Max PatchDiam is the maximum of the initial cover sets that will cover the tree shape on the second pass.\nValues that are too small may cause performance issues and produce incorrect segment\nThese values should be larger than your initial patch diameter\nSee Instructions for how to fill in this field""")
        self.TextEdit3.setToolTipDuration(10000)

        


        #slider for number of Initial PatchDiam to test
        Label4 = QLabel("Initial PatchDiam to Test:",self)
        Label4.setGeometry(400, 10, 250, 30)
        Label4.setToolTip("""The Initial PatchDiam is the size of the initial cover sets that will cover the tree shape.\nSmaller values will result in a more detailed model, but will take longer to process.\nLarger values will result in a less detailed model, but may help overcome issues from occlusion.\nSee Instructions for how to fill in this field""")
        Label4.setToolTipDuration(10000)
        self.textEntry1 = QTextEdit(self)
        self.textEntry1.setText("1")
        self.textEntry1.setGeometry(450, 50, 200, 30)
        
        self.textEntry1.setToolTip("""The Initial PatchDiam is the size of the initial cover sets that will cover the tree shape.\nSmaller values will result in a more detailed model, but will take longer to process.\nLarger values will result in a less detailed model, but may help overcome issues from occlusion.\nSee Instructions for how to fill in this field""")
        self.textEntry1.setToolTipDuration(10000)

        self.InputType = QCheckBox("Generate Values",self)
        self.InputType.setGeometry(100, 275, 200, 30)
        self.InputType.setChecked(True)
        instructionLabel = QLabel(
        """Instructions:
1. Select a value for Intensity Threshold, this will filter out all values with intensity below this threshold
2. If you select Generate Values, put a single number in Initial PatchDiam, MinPatchDiam and MaxPatchDiam 
for the number of values you would like to test
    - This will generate N reasonable values for these parameters, where N is the number you input. 
    - All combinations of these values will be tested
3. If you do not select Generate Values, put a list of values in Initial PatchDiam, MinPatchDiam and 
MaxPatchDiam separated by commas for the values you would like to test
    - All combinations of these values will be tested
4. If you select Show Only Optimal Model, only the optimal model based on the selected metric will be shown
    You may also change the metric once the processing is complete
5. Select Batch Processing to process all files in a folder, select Single File Processing to process a single file"""
            ,self)
        instructionLabel.setGeometry(25, 300, 800, 200)

        self.optimumCheck = QCheckBox("Show Only Optimal Model",self)
        self.optimumCheck.setGeometry(300, 275, 200, 30)
        self.optimumCheck.setToolTip("If checked, only the optimal model will be shown in the results. If unchecked, all models will be shown")
        self.optimumCheck.setToolTipDuration(10000)
        self.show_only_optimal = False
        self.optimumCheck.stateChanged.connect(self.optimumCheckChanged)

        self.optimumMetric = QComboBox(self)
        self.optimumMetric.setGeometry(500, 275, 200, 30)

        self.metrics = Utils.get_all_metrics()
        self.optimumMetric.addItems(self.metrics)
        self.optimumMetric.setToolTip("Select the metric to use for the optimal model. The default is 'all_mean_dis'")
        self.optimumMetric.setToolTipDuration(10000)

        self.optimumMetric.setDisabled(True)




    def optimumCheckChanged(self):
        if self.optimumCheck.isChecked():
            self.optimumMetric.setEnabled(True)
            self.show_only_optimal = True
            self.optimumMetric.setCurrentText("all_mean_dis")
        else:
            self.show_only_optimal = False
            self.optimumMetric.setEnabled(False)
           






    def start_batch_processing(self):
            #prompt user for folder path
        
        
        if self.InputType.isChecked():
            try:
                inputs = [int(self.TextEdit.toPlainText()), int(self.TextEdit2.toPlainText()), int(self.TextEdit3.toPlainText()), int(self.textEntry1.toPlainText())]
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for the inputs.")
                return
            # self.batch_window = BatchProcessingWindow(self, folder,inputs,generate_values=True)
        else:
            inputs = [self.TextEdit.toPlainText(), self.TextEdit2.toPlainText(), self.TextEdit3.toPlainText(), self.textEntry1.toPlainText()]
       
        if not self.check_inputs(inputs):
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if not folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder containing LAS or LAZ files.")
            return
        self.batch_window = BatchProcessingWindow(self, folder,inputs,self.InputType.isChecked(),self.show_only_optimal,self.optimumMetric.currentText())
        
        self.batch_window.show()
        self.hide()
        # self.batch_window = BatchProcessingWindow(self,folder)
        # self.batch_window.setLayout(QVBoxLayout())
        # self.stacked_widget.addWidget(self.batch_window) 

        # self.stacked_widget.setCurrentWidget(self.batch_window)
    
    def start_single_file_processing(self):
        #prompt user for file path
        #Intensity threshold, number of min PatchDiam to test, number of max PatchDiam to test, number of Initial PatchDiam to test
        
        if self.InputType.isChecked():
            try:
                inputs = [int(self.TextEdit.toPlainText()), int(self.TextEdit2.toPlainText()), int(self.TextEdit3.toPlainText()), int(self.textEntry1.toPlainText())]
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for the inputs.")
                return
            
        else:
            inputs = [self.TextEdit.toPlainText(), self.TextEdit2.toPlainText(), self.TextEdit3.toPlainText(), self.textEntry1.toPlainText()]
        if not self.check_inputs(inputs):
            return

        file, _ = QFileDialog.getOpenFileName(self, "Select File", "", "LAS/XYZ Files (*.las *.laz *.xyz)")
        if not file:
            QMessageBox.warning(self, "No File Selected", "Please select a LAS or LAZ file.")
            return
        self.single_window = SingleFileProcessingWindow(self, file,inputs,self.InputType.isChecked(),self.show_only_optimal,self.optimumMetric.currentText())
        self.single_window.show()
        self.hide()   

    def check_inputs(self,inputs):
        """
        Check if the inputs are valid.
        If generate_values is True, inputs should be a list of integers.
        If generate_values is False, inputs should be a list of strings.
        """
        if self.InputType.isChecked():
            try:
                inputs = [int(i) for i in inputs]
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for the inputs.")
                return False
        else:
            try:
                inputs[0]=int(inputs[0])
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for the Intensity Threshold.")
                return False
            for i in inputs[1:]:
                if not self.check_diams(i):
                    QMessageBox.warning(self, "Invalid Input", "PatchDiam inputs should be a list of decimals betweeen 0 and 1, separated by commas.\nIf you do not have values in mind, try using generate values instead ")
                    return False     
        

        return inputs
    
    def check_diams(self,diam):
        if diam =="":
            return False
        try:
            diam = [float(i) for i in diam.split(',')]
            for d in diam:
                if d <= 0:
                    return False
                if d >1:
                    return False
        except ValueError:
            return False
        return True

class BatchProcessingWindow(QMainWindow):
    def __init__(self,root,folder,inputs, generate_values, show_only_optimal=False,metric = None):
        super().__init__()
        self.setWindowTitle("Batch Processing")
        self.setGeometry(100, 100, 1600, 900)  
        self.root = root
        self.folder = folder
        self.show_only_optimal = show_only_optimal
        self.optimum_metric = metric
        
        

        files = os.listdir(folder)
        
        files = [f for f in files if f.endswith('.las') or f.endswith('.laz')]
        self.file_table = QTableWidget()
        self.file_table.setRowCount(len(files))
        if self.show_only_optimal:
            self.file_table.setColumnCount(5)
            self.file_table.setHorizontalHeaderLabels(["File Name","Completed","OptPatchDiam1","OptMaxPatchDiam","OptMinPatchDiam"])
        else:
            self.file_table.setColumnCount(2)
            self.file_table.setHorizontalHeaderLabels(["File Name","Completed"])
        self.file_table.clicked.connect(self.table_clicked)
        self.file_data = []
        for i, file in enumerate(files):
            
            item_name = QTableWidgetItem(file)

            self.file_table.setItem(i, 0, item_name)
            status = QTableWidgetItem("Not Completed")
            self.file_data.append({'file': file, 'status': status})
        self.files = files
        self.optimum =np.zeros(len(files),dtype=int)
        self.metric_data = [None for _ in range(len(files))]  # Initialize metric data for each file
        # self.hide()
        self.intensity_threshold = inputs[0]
        self.nPD2Min = inputs[1]
        self.nPD2Max = inputs[2]
        self.nPD1 = inputs[3]
        self.initial_inputs = {'PatchDiam1': self.nPD1,
                        'PatchDiam2Min': self.nPD2Min,
                        'PatchDiam2Max': self.nPD2Max}
        self.generate_values = generate_values
        self.ui = QWidget()

        self.ui.setLayout(QGridLayout())
        self.ui.layout().addWidget(self.file_table,0,0)
        self.setCentralWidget(self.ui)

        self.ui.layout().setColumnStretch(0,1)
        self.ui.layout().setColumnStretch(1,3)
        # self.info.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        # self.info.setWindowTitle("Inputs")
        self.ui.layout().setSpacing(50)


    

        self.info =QWidget()
        self.info.setLayout(QVBoxLayout())
        self.info.layout()
        self.label2 = QLabel(f"Intensity Threshold: {self.intensity_threshold}")
        # self.label2.setGeometry(50, 30, 200, 30)
        self.label2.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label2)
        if generate_values:
            self.label3 = QLabel(f"Number of Initial PatchDiam to Test: {self.nPD1}")
            # self.label3.setGeometry(50, 50, 200, 30)
            self.label3.setStyleSheet("font-size: 16px;")
            self.info.layout().addWidget(self.label3)
            self.label4 = QLabel(f"Number of Min PatchDiam to Test: {self.nPD2Min}")
            # self.label4.setGeometry(50, 70, 200, 30)
            self.label4.setStyleSheet("font-size: 16px;")
            self.info.layout().addWidget(self.label4)
            self.label5 = QLabel(f"Number of Max PatchDiam to Test: {self.nPD2Max}")
            self.info.layout().addWidget(self.label5)
            self.label5.setStyleSheet("font-size: 16px;")
        else:
            self.label3 = QLabel(f"Initial PatchDiam to Test: {self.nPD1}")
            # self.label3.setGeometry(50, 50, 200, 30)
            self.label3.setStyleSheet("font-size: 16px;")
            self.info.layout().addWidget(self.label3)
            self.label4 = QLabel(f"Min PatchDiam to Test: {self.nPD2Min}")
            # self.label4.setGeometry(50, 70, 200, 30)
            self.label4.setStyleSheet("font-size: 16px;")
            self.info.layout().addWidget(self.label4)
            self.label5 = QLabel(f"PatchDiam to Test: {self.nPD2Max}")
            self.info.layout().addWidget(self.label5)
            self.label5.setStyleSheet("font-size: 16px;")
        


        self.nav_buttons = QWidget()
        self.nav_buttons.setLayout(QHBoxLayout())
        self.ui.layout().addWidget(self.nav_buttons, 2, 1)
        # Create a left arrow button
        self.left_arrow_button = QToolButton()
        self.left_arrow_button.setArrowType(Qt.ArrowType.LeftArrow)
        self.nav_buttons.layout().addWidget(self.left_arrow_button)
        self.left_arrow_button.clicked.connect(self.left_button_clicked)
        self.left_arrow_button.setEnabled(False)

        # Create a right arrow button
        self.right_arrow_button = QToolButton()
        self.right_arrow_button.setArrowType(Qt.ArrowType.RightArrow)
        self.nav_buttons.layout().addWidget(self.right_arrow_button)
        self.right_arrow_button.clicked.connect(self.right_button_clicked)
        self.right_arrow_button.setEnabled(False)

        self.buttons_and_progress = QWidget()
        self.buttons_and_progress.setLayout(QVBoxLayout())
        self.ui.layout().addWidget(self.buttons_and_progress, 1, 0)
        self.buttons_and_progress.layout().addWidget(self.info)


        self.screen_buttons = QWidget()
        self.screen_buttons.setLayout(QVBoxLayout())
        self.buttons_and_progress.layout().addWidget(self.screen_buttons)

        self.point_cloud_button = QPushButton("Show Point Cloud", self)
        self.point_cloud_button.clicked.connect(self.show_point_cloud)
        self.screen_buttons.layout().addWidget(self.point_cloud_button)


        self.tree_data_button = QPushButton("Show Tree Data", self)
        self.tree_data_button.clicked.connect(self.show_tree_data)
        self.screen_buttons.layout().addWidget(self.tree_data_button)
        self.tree_data_button.setEnabled(False)

        self.segment_plot_button = QPushButton("Show Segment Plot", self)
        self.segment_plot_button.clicked.connect(self.show_segment_plot)
        self.screen_buttons.layout().addWidget(self.segment_plot_button)
        self.segment_plot_button.setEnabled(False)

        self.cylinder_plot_button = QPushButton("Show Cylinder Plot", self)
        self.cylinder_plot_button.clicked.connect(self.show_cylinder_plot)
        self.screen_buttons.layout().addWidget(self.cylinder_plot_button)
        self.cylinder_plot_button.setEnabled(False)

        self.combo_boxes = QWidget()
        self.combo_boxes.setLayout(QHBoxLayout())
        self.screen_buttons.layout().addWidget(self.combo_boxes)
        self.npd1_label = QLabel("PatchDiam1:")
        self.npd1_combo = QComboBox()

        self.max_pd_label = QLabel("Max PatchDiam:")
        self.max_pd_combo = QComboBox()
        self.min_pd_label = QLabel("Min PatchDiam:")
        self.min_pd_combo = QComboBox()

        self.npd1_combo.setEnabled(False)
        self.max_pd_combo.setEnabled(False)
        self.min_pd_combo.setEnabled(False)
        self.npd1_combo.setToolTip("Select the PatchDiam1 to display")
        self.max_pd_combo.setToolTip("Select the Max PatchDiam to display")
        self.min_pd_combo.setToolTip("Select the Min PatchDiam to display")
        self.combo_boxes.layout().addWidget(self.npd1_label)
        self.combo_boxes.layout().addWidget(self.npd1_combo)
        self.combo_boxes.layout().addWidget(self.max_pd_label)
        self.combo_boxes.layout().addWidget(self.max_pd_combo)
        self.combo_boxes.layout().addWidget(self.min_pd_label)
        self.combo_boxes.layout().addWidget(self.min_pd_combo)
        if self.show_only_optimal:
            self.optimumLayout = QHBoxLayout()
            self.buttons_and_progress.layout().addLayout(self.optimumLayout)
            self.optimumLabel = QLabel("Change Optimum Metric:")
            self.optimumLayout.addWidget(self.optimumLabel)

            self.optimumMetric = QComboBox(self)
            self.optimumLayout.addWidget(self.optimumMetric)
            
            self.optimumMetric.addItems(Utils.get_all_metrics())
            self.optimumMetric.setToolTip("Select the metric to use for the optimal model. The default is 'all_mean_dis'")
            self.optimumMetric.setToolTipDuration(1000)
            self.optimumMetric.setCurrentText(self.optimum_metric)
            self.optimumMetric.setEnabled(True)
            self.optimumMetric.currentTextChanged.connect(self.optimum_changed)

        self.cloud_web_view = QWebEngineView()
        # self.web_view.setHtml(html)
        self.ui.layout().addWidget(self.cloud_web_view, 0, 1,2,1)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.buttons_and_progress.layout().addWidget(self.text_edit)

        self.parallel_options =QVBoxLayout()
        self.parallel_checkbox = QCheckBox("Parallel Processing", self)
        self.parallel_checkbox.setChecked(True)
        self.parallel_checkbox.setToolTip("If checked, each file will be processed in parallel using multiple processes. If unchecked, the processing will be done sequentially.")
        self.parallel_checkbox.setToolTipDuration(10000)

        self.parallel_checkbox.stateChanged.connect(self.parallel_checkbox_changed)
        self.parallel_options.addWidget(self.parallel_checkbox)
        self.parallel_label = QLabel("Number of Processes:")
        self.parallel_options.addWidget(self.parallel_label)
        self.parallel_text_edit = QTextEdit(self)
        self.parallel_text_edit.setText(str(mp.cpu_count()))
        self.parallel_text_edit.setFixedHeight(30)
        self.parallel_text_edit.setToolTip("Set the number of processes to use for parallel processing. Default is the number of CPU cores on your PC.")
        self.parallel_text_edit.setToolTipDuration(10000)
        self.parallel_text_edit.setEnabled(True)
        self.parallel_options.addWidget(self.parallel_text_edit)
        self.buttons_and_progress.layout().addLayout(self.parallel_options)

        self.button = QPushButton("Start Processing", self)
        self.button.clicked.connect(self.process_files)
        self.ui.layout().addWidget(self.button,2,0)
    
        self.selected_index = 0
        self.num_screens = 1
        self.screen = 0
        self.data_canvas = None
        self.seg_web_view = None
        self.cyl_web_view = None
    

    def parallel_checkbox_changed(self):
        if self.parallel_checkbox.isChecked():
            self.parallel_text_edit.setEnabled(True)
        else:
            self.parallel_text_edit.setEnabled(False)
    def table_clicked(self, index):
        # Get the selected row
        self.selected_index = index.row()
        # Get the selected item
        item = self.sender().item(index.row(), 0)
        # Get the text of the selected item 
        file_name = item.text()
        # Get the status of the selected item
        status = self.file_data[index.row()]['status']
        self.append_text(f"Selected file: {file_name}\n")
        if status == "Completed":
            self.tree_data_button.setEnabled(True)
            self.segment_plot_button.setEnabled(True)
            self.cylinder_plot_button.setEnabled(True)
            self.npd1_combo.setEnabled(True)
            self.max_pd_combo.setEnabled(True)
            self.min_pd_combo.setEnabled(True)
            npd1 = self.inputs[self.selected_index]['PatchDiam1']
            self.npd1_combo.clear()
            self.npd1_combo.addItems([str(i) for i in npd1])
            max_pd = self.inputs[self.selected_index]['PatchDiam2Max']
            self.max_pd_combo.clear()
            self.max_pd_combo.addItems([str(i) for i in max_pd])
            min_pd = self.inputs[self.selected_index]['PatchDiam2Min']
            self.min_pd_combo.clear()
            self.min_pd_combo.addItems([str(i) for i in min_pd])
           
                
                
        else:
            self.tree_data_button.setEnabled(False)
            self.segment_plot_button.setEnabled(False)
            self.cylinder_plot_button.setEnabled(False)
            self.npd1_combo.setEnabled(False)
            self.max_pd_combo.setEnabled(False)
            self.min_pd_combo.setEnabled(False)
        self.show_point_cloud()

    def get_selected_inputs(self):
        if not self.show_only_optimal:
            npd1 = max(self.npd1_combo.currentIndex(),0)
            max_pd = max(self.max_pd_combo.currentIndex(),0)
            min_pd = max(self.min_pd_combo.currentIndex(),0)
            if self.generate_values:
                
            
                index = int(npd1)*self.nPD2Max*self.nPD2Min + int(max_pd)*self.nPD2Min + int(min_pd)
            else:
                index = int(npd1)*len(self.nPD2Max)*len(self.nPD2Min) + int(max_pd)*len(self.nPD2Min) + int(min_pd)
            return index
        else:
            # If show_only_optimal is True, return the optimal index
            return self.optimum[self.selected_index]

    def left_button_clicked(self):
        # Handle left button click
        if self.screen <= 0:

            return
        self.screen -= 1
        if self.screen == 0:
            self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(True)
        self.display_tree_data(self.file_data[self.selected_index]['QSM'][self.get_selected_inputs()]['treedata']['figures'][self.screen])


    def right_button_clicked(self): 
        # Handle right button click
        print(self.screen,self.num_screens)
        if self.screen >= self.num_screens - 1:
            return
        self.screen += 1
        if self.screen == self.num_screens - 1:
            self.right_arrow_button.setEnabled(False)
        self.left_arrow_button.setEnabled(True)
        self.display_tree_data(self.file_data[self.selected_index]['QSM'][self.get_selected_inputs()]['treedata']['figures'][self.screen])

    def display_tree_data(self,figure):
        if self.data_canvas != None:
            self.ui.layout().removeWidget(self.data_canvas)
            self.data_canvas.deleteLater()
        figure.dpi = 100
        self.data_canvas = FigureCanvas(figure)
        self.ui.layout().addWidget(self.data_canvas, 0, 1,2,1)

    def show_tree_data(self):
        self.append_text("Showing Tree Data...\n")
        self.left_arrow_button.setEnabled(False)
        self.screen = 0
        self.num_screens = len(self.file_data[self.selected_index]['QSM'][self.get_selected_inputs()]['treedata']['figures'])
        self.display_tree_data(self.file_data[self.selected_index]['QSM'][self.get_selected_inputs()]['treedata']['figures'][self.screen])
        self.right_arrow_button.setEnabled(True)
    
    def show_point_cloud(self):
        
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        cloud = self.file_data[self.selected_index].get('cloud',None)
        if cloud is None:
            self.append_text("Loading Point Cloud...\n")
            file = os.path.join(self.folder, self.file_data[self.selected_index]['file'])
            cloud = load_point_cloud(file, float(self.intensity_threshold))
            cloud = cloud-np.mean(cloud,axis=0)  # Center the point cloud
            self.file_data[self.selected_index]['cloud'] = cloud
        fidelity = min(1,100000/len(self.points))
        html = point_cloud_plotting(cloud,subset=True,fidelity=fidelity,marker_size=1)
        self.cloud_web_view = QWebEngineView()
        self.cloud_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.cloud_web_view, 0, 1,2,1)

    def show_segment_plot(self):
        self.append_text("Showing Segment Plot...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_inputs()
        qsm = self.file_data[self.selected_index]['QSM'][index]
        cover = qsm['cover']
        segments = qsm['segment']
        html = qsm_plotting(self.file_data[self.selected_index]['cloud'], cover, segments,qsm)
        self.seg_web_view = QWebEngineView()
        self.seg_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.seg_web_view, 0, 1,2,1)

    def show_cylinder_plot(self):
        self.append_text("Showing Cylinder Plot...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_inputs()

        html = self.file_data[self.selected_index]['plot'][index]
        self.cyl_web_view = QWebEngineView()
        self.cyl_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.cyl_web_view, 0, 1,2,1)

    
        

    def closeEvent(self, event):
        # Handle the close event
        self.root.show()
        event.accept()

    def append_text(self, text):
        self.text_edit.insertPlainText(text)

    def complete_processing(self,package):
        
        index, data, plot = package
        self.file_table.setItem(index, 1, QTableWidgetItem("Completed"))
        self.file_data[index]['status']="Completed"
        self.file_data[index]['QSM'] = data
        self.file_data[index]['plot'] = plot
        self.tree_data_button.setEnabled(True)
        self.segment_plot_button.setEnabled(True)
        self.cylinder_plot_button.setEnabled(True)
        self.npd1_combo.setEnabled(True)
        self.max_pd_combo.setEnabled(True)
        self.min_pd_combo.setEnabled(True)

        if index == self.selected_index and not self.show_only_optimal:
            npd1 = self.inputs[self.selected_index]['PatchDiam1']
            self.npd1_combo.clear()
            self.npd1_combo.addItems([str(i) for i in npd1])
            max_pd = self.inputs[self.selected_index]['PatchDiam2Max']
            self.max_pd_combo.clear()
            self.max_pd_combo.addItems([str(i) for i in max_pd])
            min_pd = self.inputs[self.selected_index]['PatchDiam2Min']
            self.min_pd_combo.clear()
            self.min_pd_combo.addItems([str(i) for i in min_pd])
        elif self.show_only_optimal:

            self.optimum[index] = self.calculate_optimal(index)
            npd1 = self.file_data[index]['QSM'][self.optimum[index]]['PatchDiam1']
            max_pd = self.file_data[index]['QSM'][self.optimum[index]]['PatchDiam2Max']
            min_pd = self.file_data[index]['QSM'][self.optimum[index]]['PatchDiam2Min']
            self.file_table.setItem(index, 2, QTableWidgetItem(str(npd1)))
            self.file_table.setItem(index, 3, QTableWidgetItem(str(max_pd)))
            self.file_table.setItem(index, 4, QTableWidgetItem(str(min_pd)))

        
        self.append_text(f"Processing Complete for {self.file_data[index]['file']}...\n")

    def add_cloud(self,package):
        index, cloud = package
        self.file_data[index]['cloud'] = cloud
        self.append_text(f"Loaded {self.file_data[index]['file']} with {cloud.shape[0]} points\n")
        
    def set_inputs(self,inputs):
        
        self.inputs = inputs
    def process_files(self):
        self.append_text("Processing file...\n")
        
        task = BatchQSM(self,self.folder,self.files,self.intensity_threshold, self.initial_inputs,self.generate_values,self.parallel_text_edit.toPlainText())
        self.qsm_thread = BackgroundProcess(task)
        task.finished.connect(self.complete_processing)
        task.message.connect(self.append_text)
        task.plot_data.connect(self.add_cloud)
        task.input_list.connect(self.set_inputs)
        self.qsm_thread.start()
        
    def optimum_changed(self):
        self.append_text("Changing optimum metric...\n")
        self.append_text("Changing optimum metric...\n")
        for index in range(len(self.file_data)):
            if self.metric_data[index] is None:
                self.metric_data[index] = Utils.collect_data(self.file_data[index]['QSM'])
            status = self.file_data[index]['status']
            if status != "Completed":
                continue
            

            self.optimum[index] = self.calculate_optimal(index)
            npd1 = self.file_data[index]['QSM'][self.optimum[index]]['PatchDiam1']
            max_pd = self.file_data[index]['QSM'][self.optimum[index]]['PatchDiam2Max']
            min_pd = self.file_data[index]['QSM'][self.optimum[index]]['PatchDiam2Min']
            self.file_table.setItem(index, 2, QTableWidgetItem(str(npd1)))
            self.file_table.setItem(index, 3, QTableWidgetItem(str(max_pd)))
            self.file_table.setItem(index, 4, QTableWidgetItem(str(min_pd)))
            self.append_text(f"Optimal PatchDiam1: {npd1}, Max PatchDiam: {max_pd}, Min PatchDiam: {min_pd}\n")

    def calculate_optimal(self,index=None):
        if index is None:
            index = self.selected_index
        if self.metric_data[index] is None:
            self.append_text("Calculating metrics...\n")
            self.metric_data[index] = Utils.collect_data(self.file_data[index]['QSM'])
        metrics = []
        for i in range(len(self.file_data[index]['QSM'])):
            metrics.append(Utils.compute_metric_value(Utils.select_metric(self.optimumMetric.currentText()), i,self.metric_data[index][0],self.metric_data[index][3]))
        return np.argmax(np.array(metrics))
    

class SingleFileProcessingWindow(QMainWindow):
    def __init__(self,root,file,inputs,generate_values, show_only_optimal=False,metric = None):
        super().__init__()
        self.setWindowTitle("Single File Processing")
        self.setGeometry(100, 100, 1600, 900)  
        self.root = root
        self.args = inputs
        self.generate_values =generate_values
        self.show_only_optimal = show_only_optimal
        self.optimum_metric = metric
        self.optimum_calculated = False
        self.metric_data = None
        self.optimum =0
        self.initModel(file,inputs,generate_values)


        self.ui = QWidget()
        self.ui.setLayout(QGridLayout())
        self.info =QWidget()
        self.info.setLayout(QVBoxLayout())
        self.info.layout()
        self.ui.layout().setColumnStretch(0,1)
        self.ui.layout().setColumnStretch(1,3)
        self.info.setMaximumHeight(200)
        # self.info.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        # self.info.setWindowTitle("Inputs")
        self.info.layout().setSpacing(0)
        self.info.setGeometry(50, 50, 200, 200)
        self.ui.layout().addWidget(self.info,0,0)
        self.ui.layout().setSpacing(100)
        
        #Display inputs in top left corner
        self.label = QLabel("Inputs:")
        # self.label.setGeometry(50, 10, 200, 30)
        self.label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.info.layout().addWidget(self.label)
        self.label2 = QLabel(f"Intensity Threshold: {self.intensity_threshold}")
        # self.label2.setGeometry(50, 30, 200, 30)
        self.label2.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label2)
        self.label3 = QLabel(f"Initial PatchDiam to Test: {self.nPD1_vals}")
        # self.label3.setGeometry(50, 50, 200, 30)
        self.label3.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label3)
        self.label4 = QLabel(f"Min PatchDiam to Test: {self.nPD2Min_vals}")
        # self.label4.setGeometry(50, 70, 200, 30)
        self.label4.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label4)
        self.label5 = QLabel(f"Max PatchDiam to Test: {self.nPD2Max_vals}")
        # self.label5.setGeometry(50, 90, 200, 30)
        self.label5.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label5)
        self.label6 = QLabel(f"File Name: {self.file}")
        # self.label6.setGeometry(50, 110, 200, 30)
        self.label6.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label6)
        self.setCentralWidget(self.ui)


        

        self.button = QPushButton("Start Processing", self)
        self.button.clicked.connect(self.process_file)
        self.ui.layout().addWidget(self.button,2,0)

        

        self.nav_buttons = QWidget()
        self.nav_buttons.setLayout(QHBoxLayout())
        self.ui.layout().addWidget(self.nav_buttons, 2, 1)
        # Create a left arrow button
        self.left_arrow_button = QToolButton()
        self.left_arrow_button.setArrowType(Qt.ArrowType.LeftArrow)
        self.nav_buttons.layout().addWidget(self.left_arrow_button)
        self.left_arrow_button.clicked.connect(self.left_button_clicked)
        self.left_arrow_button.setEnabled(False)

        # Create a right arrow button
        self.right_arrow_button = QToolButton()
        self.right_arrow_button.setArrowType(Qt.ArrowType.RightArrow)
        self.nav_buttons.layout().addWidget(self.right_arrow_button)
        self.right_arrow_button.clicked.connect(self.right_button_clicked)
        self.right_arrow_button.setEnabled(False)

        self.buttons_and_progress = QWidget()
        self.buttons_and_progress.setLayout(QVBoxLayout())
        self.ui.layout().addWidget(self.buttons_and_progress, 1, 0)


        self.screen_buttons = QWidget()
        self.screen_buttons.setLayout(QVBoxLayout())
        self.buttons_and_progress.layout().addWidget(self.screen_buttons)
        
        self.point_cloud_button = QPushButton("Show Point Cloud", self)
        self.point_cloud_button.clicked.connect(self.show_point_cloud)
        self.screen_buttons.layout().addWidget(self.point_cloud_button)
        
        
        self.tree_data_button = QPushButton("Show Tree Data", self)
        self.tree_data_button.clicked.connect(self.show_tree_data)
        self.screen_buttons.layout().addWidget(self.tree_data_button)
        self.tree_data_button.setEnabled(False)

        self.segment_plot_button = QPushButton("Show Segment Plot", self)
        self.segment_plot_button.clicked.connect(self.show_segment_plot)
        self.screen_buttons.layout().addWidget(self.segment_plot_button)
        self.segment_plot_button.setEnabled(False)

        self.cylinder_plot_button = QPushButton("Show Cylinder Plot", self)
        self.cylinder_plot_button.clicked.connect(self.show_cylinder_plot)
        self.screen_buttons.layout().addWidget(self.cylinder_plot_button)
        self.cylinder_plot_button.setEnabled(False)

        self.combo_boxes = QWidget()
        self.combo_boxes.setLayout(QHBoxLayout())
        self.screen_buttons.layout().addWidget(self.combo_boxes)
        self.npd1_label = QLabel("PatchDiam1:")
        self.npd1_combo = QComboBox()

        self.max_pd_label = QLabel("Max PatchDiam:")
        self.max_pd_combo = QComboBox()
        self.min_pd_label = QLabel("Min PatchDiam:")
        self.min_pd_combo = QComboBox()
        
        self.npd1_combo.setEnabled(False)
        self.max_pd_combo.setEnabled(False)
        self.min_pd_combo.setEnabled(False)
        self.npd1_combo.setToolTip("Select the PatchDiam1 to display")
        self.max_pd_combo.setToolTip("Select the Max PatchDiam to display")
        self.min_pd_combo.setToolTip("Select the Min PatchDiam to display")
        self.combo_boxes.layout().addWidget(self.npd1_label)
        self.combo_boxes.layout().addWidget(self.npd1_combo)
        self.combo_boxes.layout().addWidget(self.max_pd_label)
        self.combo_boxes.layout().addWidget(self.max_pd_combo)
        self.combo_boxes.layout().addWidget(self.min_pd_label)
        self.combo_boxes.layout().addWidget(self.min_pd_combo)

        if self.show_only_optimal:
            self.optimumLayout = QHBoxLayout()
            self.buttons_and_progress.layout().addLayout(self.optimumLayout)
            self.optimumLabel = QLabel("Change Optimum Metric:")
            self.optimumLayout.addWidget(self.optimumLabel)

            self.optimumMetric = QComboBox(self)
            self.optimumLayout.addWidget(self.optimumMetric)
            
            self.optimumMetric.addItems(Utils.get_all_metrics())
            self.optimumMetric.setToolTip("Select the metric to use for the optimal model. The default is 'all_mean_dis'")
            self.optimumMetric.setToolTipDuration(1000)
            self.optimumMetric.setCurrentText(self.optimum_metric)
            self.optimumMetric.setEnabled(True)
            self.optimumMetric.currentTextChanged.connect(self.optimum_changed)
            

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.buttons_and_progress.layout().addWidget(self.text_edit)

        self.output_text = OutputText(self.text_edit)
        sys.stdout = self.output_text
        #Exit behavior
        


        


        self.show_point_cloud()
        self.plots = None
        self.data = None
        self.data_canvas = None
        self.seg_web_view = None
        self.cyl_web_view = None
        # self.button.setGeometry(300, 500, 200, 50)
        self.screen = 0
        self.num_screens = 1


    def initModel(self,file,inputs,generate_values):
        self.file = file
        self.intensity_threshold = inputs[0]
        self.nPD2Min = inputs[1]
        self.nPD2Max = inputs[2]
        self.nPD1 = inputs[3]
        self.points = load_point_cloud(self.file,intensity_threshold=float(self.intensity_threshold))

        # Step 3: Define inputs for TreeQSM
        print(np.mean(self.points,axis = 0))
        self.points = self.points - np.mean(self.points,axis = 0)
        if generate_values:
            self.inputs = define_input(self.file,self.nPD1, self.nPD2Min, self.nPD2Max)[0]
        else:
            self.inputs = define_input(self.file,1,2,3)[0]
            self.nPD1 = [float(i.strip()) for i in self.nPD1.split(',')]
            self.nPD2Min = [float(i.strip()) for i in self.nPD2Min.split(',')]
            self.nPD2Max = [float(i.strip()) for i in self.nPD2Max.split(',')]
            self.inputs['PatchDiam1'] = self.nPD1
            self.inputs['PatchDiam2Min'] = self.nPD2Min
            self.inputs['PatchDiam2Max'] = self.nPD2Max
            self.inputs['BallRad1'] = [i+.01 for i in self.inputs['PatchDiam1']]
            self.inputs['BallRad2'] = [i+.01 for i in self.inputs['PatchDiam2Max']]
        # self.inputs['name']=file
        self.nPD1_vals = self.inputs['PatchDiam1']
        self.nPD2Min_vals = self.inputs['PatchDiam2Min']
        self.nPD2Max_vals = self.inputs['PatchDiam2Max']
        self.inputs['plot']=0

    
    
    def show_point_cloud(self):
        self.append_text("Showing Point Cloud...\n")
        fidelity = min(1,100000/len(self.points))
        html = point_cloud_plotting(self.points,subset=True,fidelity=fidelity,marker_size=1)

        self.cloud_web_view = QWebEngineView()
        self.cloud_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        # self.web_view.setHtml(html)
        self.ui.layout().addWidget(self.cloud_web_view, 0, 1,2,1)
        self.cloud_web_view.show()
        self.num_screens = 1
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)

    def left_button_clicked(self):
        # Handle left button click
        if self.screen <= 0:

            return
        self.screen -= 1
        if self.screen == 0:
            self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(True)
        self.display_tree_data(self.tree_data[self.screen])


    def right_button_clicked(self): 
        # Handle right button click
        if self.screen >= self.num_screens - 1:
            return
        self.screen += 1
        if self.screen == self.num_screens - 1:
            self.right_arrow_button.setEnabled(False)
        self.left_arrow_button.setEnabled(True)
        self.display_tree_data(self.tree_data[self.screen])

    def show_tree_data(self):
        self.append_text("Showing Tree Data...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_index()
        self.tree_data = self.data[index]['treedata']['figures']
        self.screen = 0
        self.num_screens = len(self.tree_data)
        self.display_tree_data(self.tree_data[0])
        self.right_arrow_button.setEnabled(True)


    def display_tree_data(self,figure):
        if self.data_canvas != None:
            self.ui.layout().removeWidget(self.data_canvas)
            self.data_canvas.deleteLater()
        figure.dpi = 100
        self.data_canvas = FigureCanvas(figure)

        self.ui.layout().addWidget(self.data_canvas, 0, 1,2,1)
        self.data_canvas.show()

    def show_segment_plot(self):
        self.append_text("Showing Segment Plot...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_index()
        qsm = self.data[index]
        cover = qsm['cover']
        segments = qsm['segment']
        html = qsm_plotting(self.points, cover, segments,qsm)
        self.seg_web_view = QWebEngineView()
        self.seg_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.seg_web_view, 0, 1,2,1)


    def show_cylinder_plot(self):   
        self.append_text("Showing Cylinder Plot, this may take a few moments...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_index()
        self.cyl_web_view = QWebEngineView()
        html = self.cyl_plots[index]
        self.cyl_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.cyl_web_view, 0, 1,2,1)




    def append_text(self, text):
        self.text_edit.insertPlainText(text)

    def process_file(self):
        self.append_text("Processing file. This may take several minutes...\n")
        
        task = SingleQSM(self,self.points,self.inputs)
        self.qsm_thread = BackgroundProcess(task)
        task.finished.connect(self.complete_processing)
        self.qsm_thread.start()


    def get_selected_index(self):

        if not self.show_only_optimal:
            npd1 = max(self.npd1_combo.currentIndex(),0)
            max_pd = max(self.max_pd_combo.currentIndex(),0)
            min_pd = max(self.min_pd_combo.currentIndex(),0)
            if self.generate_values:
                
            
                index = int(npd1)*self.nPD2Max*self.nPD2Min + int(max_pd)*self.nPD2Min + int(min_pd)
            else:
                index = int(npd1)*len(self.nPD2Max)*len(self.nPD2Min) + int(max_pd)*len(self.nPD2Min) + int(min_pd)
            return index
        else:
            if not self.optimum_calculated:
                self.append_text("Calculating optimum model...\n")
                self.optimum = self.calculate_optimal()
            return self.optimum
            #     self.optimum_calculated = True
            #     self.inputs['optimum_metric'] = self.optimumMetric.currentText()
            #     self.data, self.cyl_plots = calculate_optimal_model(self.points,self.inputs,self.optimumMetric.currentText())
            # return 0

    def optimum_changed(self):
        self.append_text("Changing optimum metric...\n")

        self.optimum = self.calculate_optimal()
        npd1 = self.data[self.optimum]['PatchDiam1']
        max_pd = self.data[self.optimum]['PatchDiam2Max']
        min_pd = self.data[self.optimum]['PatchDiam2Min']
        self.append_text(f"Optimal PatchDiam1: {npd1}, Max PatchDiam: {max_pd}, Min PatchDiam: {min_pd}\n")
        
    def complete_processing(self,package):
        self.append_text("Processing Complete...\n")

        data,plot = package
        self.cyl_plots =plot
        self.data=data


        self.tree_data_button.setEnabled(True)
        self.segment_plot_button.setEnabled(True)
        self.cylinder_plot_button.setEnabled(True)


        if not self.show_only_optimal:
            self.npd1_combo.setEnabled(True)
            self.max_pd_combo.setEnabled(True)
            self.min_pd_combo.setEnabled(True)
        
            npd1=self.inputs['PatchDiam1']
            self.npd1_combo.addItems([str(i) for i in npd1])
            max_pd = self.inputs['PatchDiam2Max']
            self.max_pd_combo.addItems([str(i) for i in max_pd])
            min_pd = self.inputs['PatchDiam2Min']
            self.min_pd_combo.addItems([str(i) for i in min_pd])
        else:
            self.metric_data = Utils.collect_data(self.data)
            self.optimum = self.calculate_optimal()
            npd1 = self.data[self.optimum]['PatchDiam1']
            max_pd = self.data[self.optimum]['PatchDiam2Max']
            min_pd = self.data[self.optimum]['PatchDiam2Min']
            self.append_text(f"Optimal PatchDiam1: {npd1}, Max PatchDiam: {max_pd}, Min PatchDiam: {min_pd}\n")
            self.optimum_calculated = True
            
    def calculate_optimal(self):
        if self.metric_data is None:
            self.append_text("Calculating metrics...\n")
            self.metric_data = Utils.collect_data(self.data)
        metrics = []
        for i in range(len(self.data)):
            metrics.append(Utils.compute_metric_value(Utils.select_metric(self.optimumMetric.currentText()), i,self.metric_data[0],self.metric_data[3]))
        return np.argmax(np.array(metrics))

        

    def closeEvent(self, event):
        # Handle the close event
        self.root.show()
        event.accept()


class BatchQSM(QObject):
    finished = Signal(tuple)
    plot_data = Signal(tuple)
    message = Signal(str)
    input_list = Signal(list)
    def __init__(self, root, folder,files,threshold,inputs,generate_values,num_cores=2):
        super().__init__()
        self.root = root
        self.folder = folder
        self.files = files
        self.intensity_threshold = float(threshold)
        self.inputs = inputs
        self.generate_values = generate_values
        self.num_cores = num_cores
        
        # self.process_file()

    def run(self):
        try:
            num_cores = int(self.num_cores)
        except:
            num_cores = mp.cpu_count()
            self.message.emit(f"Invalid number of cores specified. Using {num_cores} cores instead.\n")
        clouds = []
        for i, file in enumerate(self.files):
            point_cloud = load_point_cloud(os.path.join(self.folder, file), self.intensity_threshold)
            if point_cloud is not None:
                
                point_cloud = point_cloud - np.mean(point_cloud,axis = 0)

                clouds.append(point_cloud)
                self.plot_data.emit((i,point_cloud))
        if self.generate_values:
            inputs = define_input(clouds,self.inputs['PatchDiam1'], self.inputs['PatchDiam2Min'], self.inputs['PatchDiam2Max'])
        else:
            inputs = define_input(clouds,1,1,1)
            for cld in inputs:
                cld['PatchDiam1'] = [float(i.strip()) for i in self.inputs['PatchDiam1'].split(',')]
                cld['PatchDiam2Min'] = [float(i.strip()) for i in self.inputs['PatchDiam2Min'].split(',')]
                cld['PatchDiam2Max'] = [float(i.strip()) for i in self.inputs['PatchDiam2Max'].split(',')]
                cld['BallRad1'] = [i+.01 for i in cld['PatchDiam1']]
                cld['BallRad2'] = [i+.01 for i in cld['PatchDiam2Max']]
        self.input_list.emit(inputs)
        for i, input_params in enumerate(inputs):
            input_params['name'] = self.files[i]
            input_params['savemat'] = 0
            input_params['savetxt'] = 1
            input_params['plot'] = 0
        
    # Process each tree
        try:
            mp.set_start_method('spawn')
        except:
            pass
        Q=[]
        P=[]
        
        for i, input_params in enumerate(inputs):

            
            q = mp.Queue()
            p = mp.Process(target=treeqsm, args=(clouds[i],input_params,i,q))
            Q.append(q)
            P.append(p)
        process = 0
    
        while process < len(inputs):
            for i in range(num_cores):
                
                if process+i > len(inputs)-1:
                    break
                self.message.emit(f"Processing {inputs[process+i]['name']}. This may take several minutes...\n")
                
                P[process+i].start()

            for i in range(num_cores):
                if process+i > len(inputs)-1:
                    break
                q=Q[process+i]
                p = P[process+i]
                try:
                    batch,data,plot = q.get()
                    if data =="ERROR":
                        raise Exception("Error in processing file")
                    p.join()
                    # data,plot = treeqsm(clouds[i],input_params,i)
                    finished = self.finished.emit((batch,data,plot)) 
                except:
                    self.message.emit(f"An error occured on file {input_params['name']}. Please try again. Consider checking the console and reporting the bug to us.")  
            process+=num_cores
            
            
        self.message.emit("Processing Complete.\n")


class SingleQSM(QObject):
    finished = Signal(tuple)
    def __init__(self, root, points, inputs):
        super().__init__()
        self.root = root
        self.points = points
        self.inputs = inputs
        
        # self.process_file()



    def run(self):
        try:
            mp.set_start_method('spawn')
        except:
            pass
        try:
            q = mp.Queue()
            p = mp.Process(target=treeqsm, args=(self.points,self.inputs,0,q))
            p.start()
            
            batch,data,plot = q.get()
            if data == "ERROR":
                raise Exception("Error in processing file")
            p.join()
            # data,plot = treeqsm(self.points,self.inputs)
            
            finished = self.finished.emit((data,plot))
        except:
            self.root.append_text("An error occured while processing the file. Please try again. Consider checking the console and reporting the bug to us.\n")
            

class BackgroundProcess(QThread):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        worker.moveToThread(self)
        self.started.connect(worker.run)

    def run(self):
        super().run()
        self.worker = None
             
class OutputSignal(QObject):
    printOccured = Signal(str)

class OutputText(object):
    def __init__(self, text_edit):
        self.text_edit = text_edit
        self.output_signal = OutputSignal()
        self.output_signal.printOccured.connect(self.append_text)
    
    def write(self, text):
        self.output_signal.printOccured.emit(text)
    
    def flush(self):
        pass
    
    def append_text(self, text):
         self.text_edit.insertPlainText(text)


def run():
    app = QApplication([])
    window = QSMWindow()
    window.show()
    # Start the application

    app.exec()  

if __name__ == "__main__":
    app = QApplication([])
    window = QSMWindow()
    window.show()
    # Start the application

    app.exec()