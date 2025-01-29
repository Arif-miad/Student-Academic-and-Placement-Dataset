
# Student Academic and Placement Dataset

This dataset contains information about students' academic achievements, training, and placement status. It is designed to help analyze the various factors that contribute to a student's likelihood of securing placement after graduation.

## Dataset Overview

The dataset includes the following columns:

- **CGPA**: The overall grades achieved by the student (Cumulative Grade Point Average).
- **Internships**: The number of internships a student has completed, showcasing practical exposure.
- **Projects**: The number of projects a student has worked on, reflecting hands-on experience.
- **Workshops/Certifications**: The number of workshops or certifications the student has completed to upskill themselves, particularly in online courses.
- **AptitudeTestScore**: The score achieved by the student in aptitude tests, typically part of recruitment processes to assess quantitative and logical reasoning.
- **SoftSkillRating**: A rating of the student’s communication and interpersonal skills, crucial for placements and overall career success.
- **ExtraCurricularActivities**: A reflection of the student's involvement in activities beyond academics, showcasing their leadership and personality.
- **PlacementTraining**: Training provided by the college to help students excel in placement interviews and recruitment processes.
- **SSC**: Marks obtained by the student in their Senior Secondary Certificate (10th grade).
- **HSC**: Marks obtained by the student in their Higher Secondary Certificate (12th grade).
- **PlacementStatus**: The target column indicating whether the student has been placed (1) or not placed (0) after the recruitment process.

## Dataset Usage

This dataset can be used for various machine learning tasks, such as:

- Predicting placement success based on academic performance, training, and extracurricular involvement.
- Analyzing factors influencing students’ employability.
- Building models for recruitment prediction or recommendation systems.

## Code Implementation
``` python
import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/placement-prediction-dataset/placementdata.csv')
df.head()
df.isnull().sum()
df.info()
# 6. Distribution of target variable
sns.countplot(x='PlacementStatus', data=df, palette='coolwarm')
plt.title("Placement Status Distribution")
plt.show()
# 6. Distribution of target variable
sns.countplot(x='PlacementStatus', data=df, palette='coolwarm')
plt.title("Placement Status Distribution")
plt.show()
# 8. Internship vs Placement
sns.countplot(x='Internships', hue='PlacementStatus', data=df, palette='magma')
plt.title("Internships vs Placement")
plt.show()

# 18. Encoding categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['PlacementStatus', 'PlacementTraining', 'ExtracurricularActivities']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
# 18. Encoding categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['PlacementStatus', 'PlacementTraining', 'ExtracurricularActivities']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

from imblearn.over_sampling import SMOTE
from collections import Counter

# Splitting features and target variable
X = df.drop(columns=['PlacementStatus', 'StudentID'])  # Dropping ID column
y = df['PlacementStatus']

# Checking class distribution before balancing
print("Class distribution before balancing:", Counter(y))

# Applying SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Checking class distribution after balancing
print("Class distribution after balancing:", Counter(y_resampled))

# Creating a new balanced DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['PlacementStatus'] = y_resampled

# Display the first few rows of the balanced dataset
print(df_balanced.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import pandas as pd

# Assuming your dataset is in 'df'
Q1 = df.quantile(0.25)  # 25th percentile
Q3 = df.quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile range

# Filtering out the outliers
df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"Original DataFrame shape: {df.shape}")
print(f"DataFrame shape after removing outliers: {df_no_outliers.shape}")

# Select all numerical columns except 'PlacementStatus'
numerical_cols = df.select_dtypes(include=['number']).columns
numerical_cols = [col for col in numerical_cols if col != 'PlacementStatus']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply scaling to the selected columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop('PlacementStatus', axis=1)  # Replace 'target_column' with the actual target column name
y = df['PlacementStatus']


# Step 6: Split the data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Initialize Models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'MLP': MLPClassifier(),
}

# Step 9: Train and Evaluate Models
results = {}

for model_name, model in models.items():
    # Step 10: Train the model
    model.fit(X_train, y_train)
    
    # Step 11: Make Predictions
    y_pred = model.predict(X_test)
    
    # Step 12: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        'Accuracy': accuracy,
        'Confusion Matrix': cm,
        'Classification Report': cr
    }

# Step 13: Print results
for model_name, result in results.items():
    print(f"{model_name} Accuracy: {result['Accuracy']}")
    print(f"Confusion Matrix:\n{result['Confusion Matrix']}")
    print(f"Classification Report:\n{result['Classification Report']}\n")

# Step 14: Visualize Confusion Matrices for all models
for model_name, result in results.items():
    sns.heatmap(result['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

```
The code implementation demonstrates how to preprocess the data, build machine learning models, and evaluate their performance to predict student placement outcomes.


You can view the code implementation in the following repositories and notebooks:

- **[Kaggle Notebook: Student Placement Prediction]([https://www.kaggle.com/your-kaggle-notebook-link](https://www.kaggle.com/code/arifmia/student-academic-and-placement-predication))** - This notebook demonstrates the entire workflow from data preprocessing, feature engineering, model building, and evaluation. It includes a detailed analysis of the dataset and predictions for placement status.
  


## File Structure

The dataset is structured in a CSV format, where each row represents a student’s data, and each column corresponds to the features mentioned above.

## License

This dataset is made available for educational purposes and research. Feel free to use, modify, and contribute to this project.

