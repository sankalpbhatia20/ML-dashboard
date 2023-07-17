import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Insightify - Build and Deploy Machine Learning Models",
    layout="wide"
)

def main():
    st.title("Insightify - Build and Deploy Machine Learning Models")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file here", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = process_file(uploaded_file)
        st.write("Preview of the uploaded data:")
        st.write(df)

        st.header("Choose a Machine Learning Model")
        model_type = st.selectbox("Select a model type", ["Linear Regression", "Logistic Regression", "Random Forest"])

        if model_type != "Random Forest":
            st.warning("Random Forest models are only available for classification tasks.")

        feature_cols, target_col = select_features_target(df)
        st.write("Selected Feature Variables:", feature_cols)
        st.write("Selected Target Variable:", target_col)

        # Model training and code generation
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Logistic Regression":
            model = LogisticRegression()
        else:
            model = RandomForestClassifier()  # Change to RandomForestRegressor for regression tasks

        X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        st.header("Test the Model")
        test_model(model, feature_cols)

        st.header("Code Snippet for Model")
        st.code(generate_code(model_type, feature_cols, target_col))

def process_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def select_features_target(df):
    st.write("Choose Feature Variables and Target Variable")
    feature_cols = st.multiselect("Select feature variables", df.columns, default=[df.columns[0]])
    target_col = st.selectbox("Select target variable", df.columns)
    return feature_cols, target_col

def generate_code(model_type, feature_cols, target_col):
    code = f"""
    # Import the required libraries
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.{model_type.replace(' ', '')} import {model_type.replace(' ', '')}

    # Load the data
    df = pd.read_csv('your_data.csv')  # Change 'your_data.csv' to your file name

    # Split the data into features and target
    X = df[{feature_cols}]
    y = df['{target_col}']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data if needed (e.g., scaling, encoding, etc.)

    # Initialize and train the model
    model = {model_type.replace(' ', '')}()
    model.fit(X_train, y_train)

    # Now you can use the 'model' variable to make predictions on new data
    # For example:
    # new_data = pd.DataFrame(...)  # Replace '...' with new data
    # predictions = model.predict(new_data)
    """
    return code

def test_model(model, feature_cols):
    st.write("Enter values for feature variables to test the model:")
    test_data = {}
    for col in feature_cols:
        test_data[col] = st.number_input(col, step=1.0)
    test_df = pd.DataFrame([test_data])

    prediction = model.predict(test_df)
    st.header("Predicted Result:")
    st.write(prediction)

if __name__ == "__main__":
    main()
