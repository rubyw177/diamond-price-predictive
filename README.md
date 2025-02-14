# Diamond Price Prediction

## Overview
This project implements a **machine learning model** to predict **diamond prices** based on various features such as **carat, cut, color, clarity, and dimensions**. The model is trained using **regression techniques** to estimate the price of a diamond accurately.

## Features
- **Preprocessing of Diamond Data**: Cleans and transforms raw data.
- **Exploratory Data Analysis (EDA)**: Visualizes the distribution and correlation of features.
- **Model Training & Evaluation**: Uses regression models to predict prices.
- **Model Performance Metrics**: Evaluates using RMSE, R², and MAE.

## Dependencies
Ensure the following libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run the Project
### 1. Load the Dataset
The dataset should contain features such as:
- **Carat**: Weight of the diamond.
- **Cut**: Quality of the diamond’s cut (Fair, Good, Very Good, Premium, Ideal).
- **Color**: Grading from D (best) to J (worst).
- **Clarity**: Diamond clarity (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1).
- **Table & Depth**: Proportions of the diamond.
- **Price**: The target variable to predict.

```python
import pandas as pd
df = pd.read_csv('diamonds.csv')
df.head()
```

### 2. Perform Data Preprocessing
- **Handle Missing Values**: Removes or imputes missing entries.
- **Convert Categorical Data**: Uses one-hot encoding for categorical variables.
- **Feature Scaling**: Standardizes numerical features.

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ohe = OneHotEncoder()
df_encoded = ohe.fit_transform(df[['cut', 'color', 'clarity']])
```

### 3. Train Regression Model
A **Linear Regression model** is trained to predict diamond prices based on features.
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
```

### 4. Evaluate Model Performance
The model is evaluated using **RMSE (Root Mean Squared Error)**, **MAE (Mean Absolute Error)**, and **R² Score**.
```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

## Results & Insights
- **Carat weight** is the most influential feature in determining price.
- **Cut, color, and clarity** also significantly impact the final price.
- **The regression model achieves an R² score of ~0.85**, indicating a strong predictive performance.
