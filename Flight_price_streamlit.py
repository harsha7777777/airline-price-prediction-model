import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')  # Ignore warnings for clean output

# App Title
st.title("Flight Price Prediction Application")


# Section 1: Upload Dataset
st.header("Dataset")
uploaded_file = st.file_uploader("Upload your dataset in CSV format", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())  # Display the first few rows

    # Section 2: Exploratory Data Analysis
    st.header("2. Exploratory Data Analysis (EDA)")
    st.write("### Dataset Summary")
    st.write(data.describe())

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.write("### Correlation Heatmap")

        # Encoding Categorical Columns to Numeric (if necessary)
        # Example encoding for categorical columns (update based on your dataset)
        categorical_columns = data.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                # Convert categorical columns to numeric using LabelEncoder or custom encoding
                data[col] = data[col].astype('category').cat.codes  # Simple label encoding for example

        # Now select only numeric columns for correlation
        numeric_data = data.select_dtypes(include=[float, int])

        # Generate correlation matrix
        corr_matrix = numeric_data.corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

    # Section 3: Data Preprocessing
    st.header("3. Data Preprocessing")
    st.write("""
    This section handles categorical variables, missing values, and prepares the data for model training.
    """)

    # Select target and feature columns
    target_column = st.selectbox("Select the Target Column", data.columns)
    feature_columns = st.multiselect("Select Feature Columns", data.columns)

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            st.write("Encoding Categorical Columns:")
            st.write(categorical_columns.tolist())
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        # Handle missing values
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            st.write("Handling Missing Values...")
            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)

        # Scaling
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        st.write("### Processed Features")
        st.dataframe(X.head())

        # Train-Test Split
        st.write("### Train-Test Split")
        test_size = st.slider("Test Size (as a fraction)", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

        # Section 4: Model Building
        st.header("4. Model Building")
        model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])

        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees (for Random Forest)", 10, 200, 100)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

        # Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Results
        st.write("### Model Performance")
        st.write(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error (MSE): {metrics.mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {metrics.r2_score(y_test, y_pred):.2f}")

    

else:
    st.write("Please upload a dataset to proceed.")
