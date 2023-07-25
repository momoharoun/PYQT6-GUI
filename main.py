# All the imports needed/might be needed
# Mahmoud Haroun : 22111275
# Christian Norman : 22100827
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PyQt6.QtCore import Qt, QTimer
from PyQt6 import QtGui, QtCore
from sklearn.preprocessing import LabelEncoder
from MultiColumnLabelEncoder import MultiColumnLabelEncoder
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

from PyQt6.QtWidgets import (
    QWidget,
    QApplication,
    QMainWindow,
    QToolBar,
    QStatusBar,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QCheckBox,
    QDial,
    QSlider,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QStackedLayout,
    QPushButton,
    QComboBox
)

from PyQt6.QtGui import QAction, QIcon, QColor, QPalette
from PyQt6.QtCore import QCoreApplication, Qt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, FigureCanvasQTAgg
from matplotlib.figure import Figure

# About dataset and link 
# https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset

IMAGES_PATH = Path() / "images" 
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, format=fig_extension, dpi=resolution)  

class My_Canvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(My_Canvas, self).__init__(self.figure)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Dementia Prediction")
        self.setFixedSize(500, 500)
        self.initUI()
        self.Data_set_info()
        self.pre_processing()
        self.Model_training()
    

    def show_histogram(self):
        if self.histogram_window.isHidden():
            self.histogram_window.show()

    # Data analysis (pre-processing)
    def Data_set_info(self):
        self.data = pd.read_csv("dementia_dataset.csv")
        # First 10 rows of the dataset 
        print("First 10 rows")
        print(self.data.head(10))

        # Display histogram
        fig, ax = plt.subplots(figsize=(20, 15))
        self.data.hist(bins=50, ax=ax)
        save_fig(fig, "dataset_histogram")
        plt.show()

    # Pre processing of the dataset first must be done
    def pre_processing(self):
        self.data.drop(["Subject ID", "MRI ID","Visit","MR Delay","Hand","EDUC","MMSE","nWBV","ASF"], axis=1, inplace=True)

        mean_value = self.data['SES'].mean()
        self.data['SES'].fillna(mean_value, inplace=True)
        self.data.sort_values(by=["Group"])

        X = self.data.iloc[:, 1:]
        y = self.data.iloc[:,0]

        Mcle = MultiColumnLabelEncoder()
        self.encoded_x = Mcle.fit_transform(X)

        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(y)
        self.encoded_y = label_encoder.transform(y)
    
    # Training of the actual dataset for the feautres 
    def Model_training(self):
        seed = 0
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(self.encoded_x, self.encoded_y, test_size = test_size, random_state = seed)
        eval_set = [(X_train, y_train), (X_test, y_test)]

        self.model = XGBClassifier(max_depth = 2, n_estimators = 50, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8, colsample_bylevel = 0.7)

        self.model.fit(X_train, y_train, early_stopping_rounds = 10, eval_metric = ["merror", "mlogloss"], eval_set = eval_set, verbose = True)

        # Some information about the dataset itself 
        print("General Data on the dataset")
        print(self.data.describe())

        # Checking for missing values 
        print("Missing values")
        print(self.data.isnull())

    # Creating the layout and widgets
    def initUI(self):
        layout = QGridLayout()
        self.age_label = QLabel("Please Pick your age: ", self)
        self.age = QSlider(Qt.Orientation.Horizontal)
        self.age.setRange(60, 98)
        self.age.valueChanged.connect(self.age_value_changed)

        # eTIV: Estimated total intracranial volume
        self.etiv_label = QLabel("Pick your ETIV score given to you: ", self)
        self.etiv = QSlider(Qt.Orientation.Horizontal)
        self.etiv.setRange(1106, 2004)
        self.etiv.valueChanged.connect(self.etiv_value_changed)

        # SES: Socio Economic Status assessed by hollingshead index of social position
        self.ses_label = QLabel("Pick your SES score given to you", self)
        self.ses_score = QDoubleSpinBox()
        self.ses_score.setMinimum(1)
        self.ses_score.setMaximum(5)
        self.ses_score.valueChanged.connect(self.ses_score_value_changed)

        # CDR Clinical dementia rating
        self.cdr_label = QLabel("Pick your CDR score given to you", self)
        self.cdr = QDoubleSpinBox()
        self.cdr.setMinimum(0)
        self.cdr.setMaximum(2)
        self.cdr.setSingleStep(0.5)
        self.cdr.valueChanged.connect(self.cdr_value_changed)

        self.gender_label = QLabel("Please Select Your Gender: ", self)
        self.gender_combobox = QComboBox(self)
        self.gender_combobox.addItem("Male")
        self.gender_combobox.addItem("Female")
        self.gender_combobox.currentIndexChanged.connect(self.gender_value_changed)

        self.result_label = QLabel("Prediction: ", self)

        self.button = QAction("Histogram", self)
        self.button.setStatusTip("Show Histogram")
        self.button.triggered.connect(self.show_histogram)

        self.button1 = QAction("Save Prediction", self)
        self.button1.setStatusTip("The prediction result.")
        self.button1.triggered.connect(self.check_click)

        self.button2 = QAction("Close Application", self)
        self.button2.setStatusTip("Close Application")
        self.button2.triggered.connect(self.closeEvent)

        self.predict = QPushButton("Predict", self)
        self.predict.setStatusTip("Show Prediction")
        self.predict.clicked.connect(self.input_predection)

        menu = self.menuBar()
        self.file_menu = menu.addMenu("&File")
        self.file_menu.addAction(self.button)
        self.file_menu.addAction(self.button1)
        self.file_menu.addAction(self.button2)

        layout.addWidget(self.age_label, 0, 0)
        layout.addWidget(self.age, 1, 0)
        layout.addWidget(self.etiv_label, 2, 0)
        layout.addWidget(self.etiv, 3, 0)
        layout.addWidget(self.ses_label, 4, 0)
        layout.addWidget(self.ses_score, 5, 0)
        layout.addWidget(self.cdr_label, 6, 0)
        layout.addWidget(self.cdr, 7, 0)
        layout.addWidget(self.gender_label, 8, 0)
        layout.addWidget(self.gender_combobox, 9, 0)
        layout.addWidget(self.predict, 10, 0)
        layout.addWidget(self.result_label, 11, 0)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    # Functionality of the functions
    def age_value_changed(self, val):
        self.age_label.setText(f"Please Pick your age: {val}")
        print(val)

    def etiv_value_changed(self, val):
        self.etiv_label.setText(f"Your chosen ETIV score: {val}")
        print(val)

    def ses_score_value_changed(self, val):
        self.ses_label.setText(f"Your chosen SES score: {val}")
        print(val)

    def cdr_value_changed(self, val):
        self.cdr_label.setText(f"Your chosen CDR score: {val}")
        print(val)

    def gender_value_changed(self, index):
        val = self.gender_combobox.currentText()
        self.gender_label.setText(f"Selected gender: {val}")
        print(val)

    def check_click(self, event):
        save_fig(self.sc.figure, "prediction_plot")

    def input_predection(self):
        a = self.age.value()
        e = self.etiv.value()
        s = self.ses_score.value()
        c = self.cdr.value()
        g = self.gender_combobox.currentText()

        if g == "Male":
            g = 0

        elif g == "Female":
            g = 1


        x = np.array([[g, a, s, c, e]])

        pred = self.model.predict(x)
        if pred[0]== 1:
            Res="Demented"
        elif pred[0]==2:
            Res="Not Demented"
        else :
            Res="converted"        

        self.result_label.setText(f"Prediction: {Res}")

        # Plotting feature importance
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_importance(self.model, ax=ax)
        save_fig(fig, "feature_importance_plot")
        plt.show()

        print(pred)

    # Closing argument function
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Save:
            sys.exit()
        else:
            print("Cancel Closing")
            if not isinstance(event, bool):
                event.ignore()

        if reply == QMessageBox.StandardButton.Close:
            print("Close Event reply close")
            sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

# This project was really fun and i hope you enjoy using the GUI in predecting dementia or not! 