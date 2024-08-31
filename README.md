## Thyroid-Disease-Detection-ML-Project

# Overview:-
Project Title           : Thyroid Disease Detection
Technologies            : Machine Learning Technology
Domain                  : Healthcare
Project Difficulty Level: Intermediate

# Abstract:-
1. Thyroid Disease Detection Project is focused on predicting thyroid disease using machine learning techniques.
2. Various machine learning algorithms are evaluated to identify the best model for prediction.
3. The project involves creating a Flask web application for real-time predictions and deploying it in a cloud environment.
4. Data is sourced from the UCI Machine Learning Repository and processed for model training.
5. Data preprocessing includes handling missing values, encoding categorical features, and normalizing numerical data, necessary for Machine Learning tasks.
6. Technical Documentations, namely High Level Document (HLD) and Low Level Document (LLD) have been provided which, describes the technical process of building as well as deploying the machine learning mode, covering from data preprocessing, feature engineering, model selection, right up to model evaluation and the deployment process has been highlighted along with the ongoing monitoring which is required for maintaining the model performance.
7. The Project Architecture Document details the end-to-end process of data collection, preprocessing, and machine learning operations.
8. Wireframe Document outlines the web application’s user interface design.

# Introduction:-
Thyroid disease is a common cause of medical diagnosis and prediction, with an onset  that is difficult to forecast in medical research. The main goal is to predict the estimated risk on a patient's chance of obtaining thyroid disease or not.
This is an end-to-end Machine Learning Application designed to predict the risk of thyroid diseases using a variety of predictive models. This project also demonstrates the integration of data processing, model training, and web deployment.

1. Thyroid disease is a very common problem in India, more than one crore people are suffering with the disease every year and such disorders.
2. Thyroid disease affects a significant portion of the population, with higher prevalence among women aged 17 – 54.
3. Thyroid disorders, such as hyperthyroidism and hypothyroidism, can lead to severe health issues, including cardiovascular complications, hypertension, high 
cholesterol, depression, and reduced fertility.
4. The thyroid gland produces essential hormones, thyroxine (T4) and triiodothyronine (T3), which regulate the body's metabolism and are critical for the 
proper functioning of cells, tissues, and organs.
5. Irregular thyroid function can accelerate or decelerate the body's metabolism, leading to various health complications.
6. In the modern healthcare landscape, Artificial Intelligence and Machine Learning offer promising solutions for early detection and management of thyroid diseases.
7. This project explores the application of machine learning algorithms to predict the presence of thyroid disease, aiming to enhance diagnostic accuracy and improve patient outcomes.
8. The study compares different algorithms, such as Random Forest Classifier, \Decision Tree Classifier, Logistic Regression, Linear Regression, AdaBoost Classifier, Gradient Boosting Classifier, XGBoost Classifier, Support Vector Classifier, Gaussian Naive Bayes and K-Nearest Neighbors (KNN) Classifier, to determine the most effective model for thyroid disease prediction.

# Problem Statement:-
Thyroid disease is a common cause of medical diagnosis and prediction, with an onset that is difficult to forecast in medical research. The thyroid gland is one of our body's most vital organs. Thyroid hormone releases are responsible for metabolic regulation. Hyperthyroidism and hypothyroidism are one of the two common diseases of the thyroid that releases thyroid hormones in regulating the rate of body's metabolism.
The main goal is to predict the estimated risk on a patient's chance of obtaining thyroid disease or not.

# Approach:-
The classical machine learning tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and Model Testing. Try out different machine 
learning algorithms that’s best fit for the above case. 

# Objectives:-
1.  Predict the risk of hyperthyroidism, hypothyroidism (compensated, primary, secondary), or negative (no thyroid disease) in individuals using machine learning techniques.
2.  Implement a systematic approach involving data exploration, cleaning, feature engineering, model building, and testing to identify the most suitable machine learning model.
3.  Employ Machine Learning algorithms such as Random Forest Classifier, Decision Tree Classifier, Logistic Regression, Linear Regression, AdaBoost Classifier, Gradient Boosting Classifier, XGBoost Classifier, Support Vector Classifier, Gaussian Naïve Bayes and K-Nearest Neighbors (KNN) on the thyroid dataset from the UCI Machine Learning Repository.
4.   Enhance diagnostic accuracy through early detection and identification, aiding in better treatment decisions by healthcare professionals.
5.   Deploy the application on cloud platforms like Google Cloud Platform (GCP), Amazon Web Services (AWS), Microsoft Azure or Heroku using Flask for real-time predictions and accessibility.

# General Description:-
1. Impact: This project enhances the early detection of thyroid disease, thereby reducing the risk of delayed diagnosis. Moreover, this project also demonstrates the potential of Machine Learning in the medical field, offering insights into the application of predictive analytics in healthcare.
2. Scope: Go beyond disease prediction to showcase how advanced data processing and machine learning can revolutionize clinical diagnosis and improve patient outcomes.
3. Product Perspective: Develop a machine learning-based system to detect thyroid disease and guide necessary medical actions.
4. Problem Statement: Create an AI solution capable of detecting thyroid disease and identifying its type in both healthy and unhealthy individuals.
5. Proposed Solution:  Implement a data science model involving data preprocessing (transformation, imputation, encoding, feature selection) and model building, training, evaluation, and selection.
6. Further Imporvements: Explore additional healthcare applications and integrate with other healthcare domain solutions to provide comprehensive diagnostics.
7. Constraints: Ensure accuracy and automation, minimizing user interaction with the system's internal workings.
8. Assumptions: The system will be implemented in hospitals to handle new datasets for thyroid disease detection and reporting.

# Data Requirements:-
Data requirements solely depend on our problem. The following attributes are required:
1. Age 
2. Gender 
3. Thyroxin Treatment 
4. Antithyroid Medication 
5. Pregnancy 
6. Sick at the time of diagnosis 
7. Iodine Test 
8. Lithium Test 
9. Goitre Test 
10. Tumor Test 
11. TSH Level Measure 
12. T3 Level Measure 
13. T4 Level Measure 
14. Free Thyroxin 
15. Thyroxin – Binding Globulin (TBG)

# Tools:-
The tools used in this project include the following: 
1. Python being the language for coding, Database Operations, Data Analytics, Machine Learning and creating the Flask App. 
2. Flask used for creating the Web App. 
3. Numpy and Pandas are Libraries used for Data Analysis. 
4. Matplotlib and Seaborn are Libraries that are essential for Data Visualization. 
5. HTML, CSS and JavaScript are used for the development of the WebPage. 
6. AWS is a cloud based platform to deploy the model. 
7. Scikit-learn is a Library that is essentially used for performing Machine Learning operations.
8. GitHub is being used as a Repository. 
9. MongoDB being the Database from which the data has been retrieved. 
10. JSON has been used for transmission of data in web application. 
11. Jupyter Notebook is an IDE, used to execute the .ipynb files. 
12. Visual Studio Code (popularly known as VSCode) is an IDE that has been used extensively in this project, to write codes for each language, irrespective of 
the file type (.py, .ipynb, .html, .css, .js, .json, .txt, .md files have been written). 

# Project Structure:-
Thyroid-Disease-Detection-ML-Project/
│
├── DataFiles/                  # Contains datasets used for training and testing
├── NotebookOps/                # Jupyter Notebooks for data analysis and preprocessing
├── Pickle_Objects/             # Serialized models and other objects
├── Python_Scripts/             # Python scripts for model training, prediction, etc.
├── WebPage_Materials/          # HTML, CSS, and JavaScript files for the web interface
├── DatabaseOperations/         # Scripts for database operations
├── LogFile/                    # Log files for application runtime
├── Project_Documentation/      # Documentation including reports and presentations
├── README.md                   # This README file
└── .gitignore                  # Git ignore file to exclude unnecessary files from version control

# Architecture Description:-
1. Data Description:
      a. The Data used in this project is the Thyroid Disease Dataset present in UCI Machine Learning Repository.
      b. The Data from the above mentioned repository is exported to MongoDB Database and transformed into a CSV File. 
      c. The CSV File is loaded into Jupyter Notebook (used in VS Code) and read by using Pandas Library. This is done in order to proceed further in the project. 
2. Exploratory Data Analysis:
      a. In this segment of the Project Work, firstly the dataset is explored in the Jupyter Notebook (used in VS Code), in order to get the initial insights               about the dataset. 
      b. The duplicate values are dropped. 
      c. The Missing Values are imputed. 
      d. The Categorical Columns have been encoded, so that they can be used in Machine Learning operations, in the subsequent steps. 
      e. Data Visualization is performed by using Matplotlib and Seaborn Libraries.
3. Machine Learning Operations:
      a. K – Means Algorithm has been used to create clusters in the preprocessed data and the optimum number of clusters is selected by plotting the elbow plot. 
      b. The class imbalance is handled. 
      c. The dataset is split into Training Dataset and Test Dataset. 
      d. Hyperparameter tuning is done, so that the performance of models become better. 
      e. Models are trained. 
      f. The performance of the models are evaluated. 
      g. The Best Performing Model is selected. 
      h. The Best Model is saved.  
4. Web - App Development:
      a. The user interface is designed by using HTML, CSS and JavaScript.
      b. The Flask app is developed.
      c. Cloud Setup is done so that the model can be deployed.
      d. The Application starts, once the model is deployed.
      e. The client enters data as required and once, the Submit button is clicked, the prediction starts.
      f. Once the prediction is done, the Predicted Result is displayed on the screen.

# Workflow Summary:-
1. Firstly, the data of the database has been uploaded to MongoDB and has been successfully retrieved from MongoDB, in the form of a CSV File.
2. The dataset was checked for value consistency and the duplicated values were dropped and missing value imputation was performed.
3. Data encoding was also done.
4. Exploratory Data Analysis (EDA) was performed.
5. The Class Imbalance was handled.
6. The dataset was split into Training Dataset and Test Dataset.
7. Using different algorithms, the dataset was trained, accompanied by CrossValidation and Hyperparameter Tuning.
8. The Model Performance were evaluated.
9. The Model with the best performance was saved in the form of a Pickle object.
10. The Model was deployed on a cloud platform.

# Model Performance:-
1. Purpose:- Utilizes Machine Learning to detect Thyroid Disorders in symptomatic patients, enabling timely treatment.
2. Reusability:- Code is designed for seamless reuse without issues.
3. Application Compatibility:- Python acts as the interface, ensuring proper data transfer between components.
4. Resource Utilization:- Maximizes processing power during task execution.

# Installation:-
1. Clone the Repository:
   git clone https://github.com/ImAni07/Thyroid-Disease-Detection-ML-Project.git
   cd Thyroid-Disease-Detection-ML-Project
2. Create and Activate a Virtual Environment:
   python -m venv venv
   source venv/bin/activate
   On Windows use `venv\Scripts\activate`
3. Install Dependencies:
   pip install -r requirements.txt

# Usage:-
1. Data Processing: Jupyter Notebooks in 'NotebookOps/' handle data exploration, preprocessing, and feature engineering.
2. Model Training: Scripts in 'Python_Scripts/' are used for training various machine learning models.
3. Web Application: The scripts in 'WebPage_Materials/' have developed the webpage.
4. Use 'app.py' in 'Python_Scripts/' to run the Flask Application.

# Features of the Project Work:-
1. Data Preprocessing: Handles missing values, data encoding and dropping of duplicate values have been done.
2. Exploratory Data Analysis: EDA has been performed on the dataset post Data Preprocessing and prior to Machine Learning Model Building.
3. Model Training: Firstly the class imbalance is handled and data clustering is performed. Various Machine Learning Models have been trained. The list of the Machine Learning Models include - Random Forest Classifier, Decision Tree Classifier, Linear Regression, Logistic Regression, AdaBoost Classifier, Gradient Boosting Classifier, XGBoost Classifier, Support Vector Classifier, Gaussian Naive Bayes and K-Nearest Neighbors (KNN) Classifier. Among these, the best performing model has been selected to make the prediction.
4. Web Interface: Provides a user-friendly interface for submitting patient data and receiving predictions.
5. Logging: Tracks application usage and errors.

# Project Documentation:-
1. High Level Document: <https://github.com/ImAni07/Thyroid-Disease-Detection-ML-Project/blob/main/Project_Documentation/HLD.pdf>
2. Low Level Document: <https://github.com/ImAni07/Thyroid-Disease-Detection-ML-Project/blob/main/Project_Documentation/LLD.pdf>
3. Architecture Document: <https://github.com/ImAni07/Thyroid-Disease-Detection-ML-Project/blob/main/Project_Documentation/Architecture_Document.pdf>
4. Wireframe Document: <https://github.com/ImAni07/Thyroid-Disease-Detection-ML-Project/blob/main/Project_Documentation/Wireframe_Document.pdf>
5. Detailed Project Report: <https://github.com/ImAni07/Thyroid-Disease-Detection-ML-Project/blob/main/Project_Documentation/Detailed_Project_Report.pdf>
6. Project Demo Video: <https://github.com/ImAni07/Thyroid-Disease-Detection-ML-Project/blob/main/Project_Documentation/Project_Demo_Video.mp4>

# Conclusion:-
1. Overview of Achievements:- The Thyroid Disease Detection project effectively utilized machine learning techniques to enhance the prediction and early diagnosis of thyroid disorders. By integrating data cleaning, feature engineering, and advanced classification algorithms, the project achieved a robust system capable of accurately assessing the risk of thyroid diseases such as hypothyroidism and hyperthyroidism.
2. Impact on Healthcare:- This project demonstrates the significant potential of machine learning in improving healthcare outcomes. By enabling early detection and personalized treatment strategies, it contributes to better patient care and decision-making processes.
3. Project Scope & Limitations:- The project's scope covered comprehensive data preprocessing, model training, and deployment in a web application. While the solution has shown promising results, continuous improvements and validation with diverse datasets are essential for ensuring reliability and generalization.
4. Future Work:- Future efforts could focus on integrating additional data sources, refining model algorithms, and expanding the application to handle a broader range of thyroid-related conditions. Continuous updates and enhancements will be crucial to maintaining the effectiveness and accuracy of the detection system.
5. Final Thoughts:- The integration of machine learning into the Thyroid Disease Detection system represents a significant advancement in the field of medical diagnostics. This project not only demonstrates technical and analytical capabilities but also underscores the importance of datadriven solutions in modern healthcare.

# Acknowledgement:-
I would like to express my sincere gratitude to PW Skills for providing me with the opportunity to work on the Thyroid Disease Detection project as part of the coursework for the "Decode Data Science with Machine Learning" program. This project, sourced from the PW Skills Experience Portal, has been instrumental in enhancing my understanding of end-to-end machine learning solutions, from data preprocessing to model deployment.
I extend my heartfelt thanks to my mentors and instructors at PW Skills for their guidance and support throughout this course, which eventually guided me to complete this project. Their insights and feedback were invaluable in helping me navigate the challenges of this complex task.
PW Skills (https://pwskills.com/) for providing the project base through the Decode Data Science with Machine Learning course.
UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/102/thyroid+disease) for providing the dataset used in this project.

# Author:-
Kislay Chaturvedi
Profile: <https://www.linkedin.com/in/kislay-chaturvedi/>
