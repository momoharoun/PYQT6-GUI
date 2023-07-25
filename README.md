# Haroun, Mahmoud 

# Title: Dementia Prediction 

# https://mygit.th-deg.de/22s-ainb/re-take-as

# Project Description:
This project was giving to us to create a descion system using the scikit learn modul and the end goal was to create a GUI interface using PYQ6 library and reading and analysing a chosen data set in our instance was a Dementia dataset inlcuding features such as (CDR, Gender, SES, eTIV, Age) < meaning of each one is defined in the code itself>. Link of dataset : https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset

# Installation:
In order to run this project youll have to install all the requried modules from the requirement.txt 

# Basic Usage:
Firstly make sure you are in the right directory and run python3 main.py give it some time and check your terminal and that GUI should open along with some graphs. Type whatever inputs you want and click predict a feature importance should come up and if you close it and look at the GUI your result should seen as Predection: " " 

# Implementation of the Requests:
(This was done at the start to pre process the data and take and drop and encode whatever is needed)
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

(This is the actual training of the model)
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

(From line 145 till 219 is the actual widgets and their layouts in the GUI)
(On-wards is the functionality of the functions, input predection values, and finally the closing event)

# Work Done:
Both parties had contributions in getting the whole project done!
