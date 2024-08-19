import pickle
import dill
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from memory import save_ml_state, load_ml_state, save_state
from memory import load_ml_uploaded_files
import io


def display_data_preview(df):
    st.write("Data preview:")
    st.write(df.head())


def select_ml_target_variable():
    file_name = st.session_state.current_ml_file
    if not file_name:
        st.warning("Please upload or select a file first.")
        return None, None

    load_ml_state(file_name)
    df = pd.read_csv(io.BytesIO(st.session_state.ml_uploaded_files[file_name]))

    current_target = st.session_state.get('target_column')

    target_column = st.selectbox(
        "Choose target variable for Machine Learning",
        df.columns,
        key=f"target_column_selector_{file_name}",
        index=df.columns.get_loc(current_target) if current_target in df.columns else 0
    )

    if target_column != current_target:
        st.session_state.target_column = target_column
        save_ml_state(file_name)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = preprocess_features(X)
    y, task = preprocess_target(y, df[target_column])

    st.session_state.X = X
    st.session_state.y = y
    st.session_state.task = task
    save_ml_state(file_name)

    return file_name, target_column


def preprocess_features(X):
    le = LabelEncoder()
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = le.fit_transform(X[column].astype(str).fillna('unknown'))
        elif X[column].dtype == 'bool':
            X[column] = X[column].astype(int)
        else:
            X[column] = X[column].astype(float).fillna(0)
    return X


def preprocess_target(y, original_target):
    if original_target.dtype == 'object' or original_target.nunique() < 10:
        task = 'classification'
        original_classes = sorted(original_target.unique())
        class_mapping = {val: i + 1 for i, val in enumerate(original_classes)}
        y = y.map(class_mapping)
        st.session_state.class_mapping = {v: k for k, v in class_mapping.items()}
    else:
        task = 'regression'
        y = y.astype(float)
        st.session_state.class_mapping = None
    return y, task


def train_feature_importance_model(X, y, task):
    if task == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def calculate_feature_importance(model, X):
    return pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)


def plot_feature_importance(feature_importance):
    top_15_features = feature_importance.head(15)
    fig = px.bar(top_15_features, x='feature', y='importance',
                 title=f'Top 15 Features by Importance for {st.session_state.target_column}',
                 labels={'feature': 'Features', 'importance': 'Importance'},
                 height=500)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)


def create_preprocessor(X_selected):
    numeric_features = X_selected.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_selected.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def upload_model(file_name):
    if file_name not in st.session_state.file_contexts:
        st.session_state.file_contexts[file_name] = {}

    context = st.session_state.file_contexts[file_name]

    uploaded_file = st.file_uploader("Upload your pre-trained model (.joblib, .pkl, or .dill file)",
                                     type=["joblib", "pkl", "dill"], key=f"model_uploader_{file_name}")

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'joblib':
                model = joblib.load(uploaded_file)
            elif file_extension == 'pkl':
                model = pickle.load(uploaded_file)
            elif file_extension == 'dill':
                model = dill.load(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return None

            if hasattr(model, 'predict') and hasattr(model, 'fit'):
                context['uploaded_model'] = model
                context['uploaded_model_filename'] = uploaded_file.name
                st.success("Model uploaded successfully!")
                save_ml_state(file_name)
                return model
            else:
                st.error("Uploaded file is not a valid scikit-learn model.")
        except Exception as e:
            st.error(f"Error loading the model: {str(e)}")

    if 'uploaded_model' in context:
        st.write(f"Currently uploaded model: {context.get('uploaded_model_filename')}")
        return context['uploaded_model']

    return None


def handle_custom_model(X, y, task, target_column, file_name):
    uploaded_model = upload_model(file_name)

    if uploaded_model is not None:
        st.session_state.trained_model = uploaded_model
        st.session_state.model_trained = True
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.task = task

        save_ml_state(file_name)
    else:
        st.error("Please upload a valid custom model.")

    save_ml_state(file_name)


def create_model(model_choice, task, preprocessor):
    if task == 'classification':
        if model_choice == "Logistic Regression":
            return Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(random_state=42))
            ])
        elif model_choice == "Decision Tree":
            return Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ])
        elif model_choice == "Random Forest":
            return Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        else:  # XGBoost
            return Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(random_state=42))
            ])
    else:  # Regression task
        if model_choice == "Logistic Regression":
            st.warning("Logistic Regression is not suitable for regression tasks. Using Linear Regression instead.")
            return Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
        elif model_choice == "Decision Tree":
            return Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', DecisionTreeRegressor(random_state=42))
            ])
        elif model_choice == "Random Forest":
            return Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
        else:  # XGBoost
            return Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor(random_state=42))
            ])


def select_and_train_model(X, y, task, target_column, file_name):
    load_ml_state(file_name)
    algorithms = ["Upload Custom Model", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]

    # Set default algorithm to "Upload Custom Model"
    default_algorithm = "Upload Custom Model"

    # Get the current algorithm from session state, use default if not set
    current_algorithm = st.session_state.file_contexts[file_name].get('selected_algorithm', default_algorithm)

    # Ensure current_algorithm is in the list, otherwise use default
    if current_algorithm not in algorithms:
        current_algorithm = default_algorithm

    model_choice = st.selectbox("Select an algorithm",
                                algorithms,
                                index=algorithms.index(current_algorithm),
                                key=f'model_selector_{file_name}')

    # Only update and save if the algorithm has changed
    if model_choice != current_algorithm:
        st.session_state.file_contexts[file_name]['selected_algorithm'] = model_choice
        save_ml_state(file_name)

    if model_choice == "Upload Custom Model":
        handle_custom_model(X, y, task, target_column, file_name)
    else:
        # Check if model is already trained
        if not st.session_state.file_contexts[file_name].get('model_trained', False):
            if st.button("Train and Evaluate Model", key=f"train_evaluate_{file_name}"):
                with st.spinner("Training model..."):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    train_and_evaluate_model(model_choice, X_train, X_test, y_train, y_test, task, X.columns, target_column, file_name)
        else:
            st.success("Model already trained. You can make predictions below.")
            if st.button("Retrain Model", key=f"retrain_{file_name}"):
                with st.spinner("Retraining model..."):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    train_and_evaluate_model(model_choice, X_train, X_test, y_train, y_test, task, X.columns, target_column, file_name)


def train_and_evaluate_model(model_choice, X_train, X_test, y_train, y_test, task, feature_columns, target_column, file_name):
    preprocessor = create_preprocessor(X_train)
    model = create_model(model_choice, task, preprocessor)
    model.fit(X_train, y_train)

    try:
        y_pred = model.predict(X_test)

        if task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            st.session_state.accuracy = accuracy
            st.write(f"Model Accuracy: {accuracy:.2f}")
            classification_report_str = classification_report(y_test, y_pred)
            st.session_state.classification_report = classification_report_str
            st.write("Classification Report:")
            st.text(classification_report_str)
        else:
            mse = mean_squared_error(y_test, y_pred)
            st.session_state.mse = mse
            st.write(f"Model Mean Squared Error: {mse:.2f}")

        st.session_state.trained_model = model
        st.session_state.X = X_train
        st.session_state.task = task

        save_ml_state(file_name)

    except Exception as e:
        st.error(f"An error occurred while making predictions: {str(e)}")
        st.write("The model might not be compatible with the current data or task. Please check the model and try again.")


def create_input_fields(feature_columns, data, file_name):
    input_data = {}
    col1, col2 = st.columns(2)

    original_data = st.session_state.file_contexts[file_name].get('original_dataset', data)
    numeric_features = ['PEDS', 'VE_TOTAL', 'MONTH', 'HOUR', 'MINUTENAME', 'FATALS', 'Total_PER']

    # Initialize prediction_input if not exists
    if 'prediction_input' not in st.session_state.file_contexts[file_name]:
        st.session_state.file_contexts[file_name]['prediction_input'] = {}

    @st.cache_data
    def get_unique_values(data, column, max_values=100):
        unique_values = data[column].dropna().unique()
        unique_values = sorted(unique_values, key=lambda x: (str(x).lower(), x))
        unique_values = [str(val) for val in unique_values]
        if len(unique_values) > max_values:
            unique_values = unique_values[:max_values]
        return unique_values

    for i, col in enumerate(feature_columns):
        unique_values = get_unique_values(original_data, col)
        unique_values.append("Enter custom value")

        column = col1 if i % 2 == 0 else col2
        with column:
            selectbox_key = f"pred_{col}_{file_name}"

            # Get the current value from session state, or use the first unique value as default
            current_value = st.session_state.file_contexts[file_name]['prediction_input'].get(col, unique_values[0])

            # Ensure the current value is in the list of options
            if current_value not in unique_values:
                current_value = unique_values[0]

            selected_value = st.selectbox(
                f"Select value for {col}",
                options=unique_values,
                index=unique_values.index(current_value),
                key=selectbox_key
            )

            if selected_value == "Enter custom value":
                custom_key = f"custom_{col}_{file_name}"
                custom_value = st.text_input(f"Enter custom value for {col}", key=custom_key)
                if custom_value:
                    selected_value = custom_value

            # Update the session state
            st.session_state.file_contexts[file_name]['prediction_input'][col] = selected_value

            if col in numeric_features:
                input_data[col] = float(selected_value) if selected_value and selected_value != 'Enter custom value' else 0.0
            else:
                input_data[col] = str(selected_value) if selected_value and selected_value != 'Enter custom value' else 'unknown'

    return input_data


def make_prediction(model, input_data, task, class_mapping, file_name):
    input_df = pd.DataFrame([input_data])
    original_features = st.session_state.file_contexts[file_name]['original_dataset'].columns.tolist()
    numeric_features = ['PEDS', 'VE_TOTAL', 'MONTH', 'HOUR', 'MINUTENAME', 'FATALS', 'Total_PER']

    for feature in original_features:
        if feature not in input_df.columns:
            if feature in numeric_features:
                input_df[feature] = 0.0
            else:
                input_df[feature] = 'unknown'

        if feature in numeric_features:
            input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0).astype(float)
        else:
            input_df[feature] = input_df[feature].astype(str).fillna('unknown')

    input_df = input_df[original_features]

    try:
        prediction = model.predict(input_df)

        if task == 'classification':
            predicted_class = class_mapping.get(prediction[0], prediction[0])
            st.success(f"Predicted class: {predicted_class}")
        else:
            st.success(f"Predicted value: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Input data:")
        st.write(input_df)
        st.write("Input data types:")
        st.write(input_df.dtypes)

    return input_df


@st.cache_data
def get_feature_info(data):
    feature_info = {}
    for col in data.columns:
        unique_values = data[col].dropna().unique()
        if len(unique_values) > 100:
            unique_values = sorted(unique_values)[:100]
        feature_info[col] = list(unique_values)
    return feature_info


def prediction_section(file_name):
    load_ml_state(file_name)

    try:
        model = st.session_state.trained_model or st.session_state.uploaded_model
    except AttributeError:
        model = None

    if model is not None and st.session_state.X is not None:
        feature_columns = st.session_state.X.columns
        data = st.session_state.X

        input_data = create_input_fields(feature_columns, data, file_name)

        if st.button("Predict", key=f"predict_button_{file_name}"):
            make_prediction(model, input_data, st.session_state.task, st.session_state.get('class_mapping'), file_name)
    else:
        st.info("Please upload or train a model before making predictions.")

    predict_class_expl()


def predict_class_expl():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Prediction values for CRSS and FARS:  \n'
                 '- 0 &mdash; :grey-background[No apparent injury]  \n'
                 '- 1 &mdash; :blue-background[Minor Injury]  \n'
                 '- 2 &mdash; :orange-background[Serious Injury]  \n'
                 '- 3 &mdash; :red-background[Fatal Injury]')

    with col2:
        st.write('Prediction values for Level 2:  \n'
                 '- 0 &mdash; :grey-background[Unknown injury]  \n'
                 '- 1 &mdash; :green-background[No Injury]  \n'
                 '- 2 &mdash; :blue-background[Minor Injury]  \n'
                 '- 3 &mdash; :violet-background[Moderate Injury]  \n'
                 '- 4 &mdash; :orange-background[Serious Injury]  \n'
                 '- 5 &mdash; :red-background[Fatal Injury]')

    with col3:
        st.write('Prediction values for DMV:  \n'
                 '- 2 &mdash; :blue-background[Minor Injury]  \n'
                 '- 3 &mdash; :orange-background[Moderate Injury]  \n'
                 '- 4 &mdash; :red-background[Major Injury]')


def choose_ml_file():
    ml_uploaded_files = load_ml_uploaded_files()
    files = list(ml_uploaded_files.keys()) + ["Add new file"]

    selected_file = st.sidebar.selectbox(
        "Choose File for ML Analysis",
        files,
        index=files.index(st.session_state.get('current_ml_file')) if st.session_state.get(
            'current_ml_file') in files else len(files) - 1,
        key="ml_file_selector"
    )

    if selected_file != st.session_state.get('current_ml_file'):
        if selected_file == "Add new file":
            st.session_state.show_ml_upload = True
            st.session_state.current_ml_file = None
        else:
            st.session_state.current_ml_file = selected_file
            st.session_state.show_ml_upload = False
            load_ml_state(selected_file)

    return st.session_state.get('current_ml_file')


def upload_ml_file():
    uploaded_file = st.file_uploader("Choose a CSV file for ML Analysis", type="csv", key="ml_file_uploader")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_content = uploaded_file.read()

        # Save the uploaded file
        st.session_state.ml_uploaded_files[file_name] = file_content

        # Initialize the file context
        if file_name not in st.session_state.file_contexts:
            st.session_state.file_contexts[file_name] = {}

        # Set the current file
        st.session_state.current_ml_file = file_name

        st.success(f"File '{file_name}' uploaded successfully!")
        save_ml_state(file_name)
        save_state()
        st.rerun()


def delete_ml_file():
    if st.session_state.current_ml_file and st.session_state.current_ml_file in st.session_state.ml_uploaded_files:
        if st.sidebar.button("Delete Current File"):
            st.session_state.confirm_ml_delete = True

    if st.session_state.get('confirm_ml_delete', False):
        st.sidebar.warning("Are you sure you want to delete this ML file?")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Yes"):
            # Delete the file from ml_uploaded_files
            del st.session_state.ml_uploaded_files[st.session_state.current_ml_file]

            # Remove associated ML state
            if st.session_state.current_ml_file in st.session_state.file_contexts:
                del st.session_state.file_contexts[st.session_state.current_ml_file]

            # Reset current file and show upload
            st.session_state.current_ml_file = None
            st.session_state.show_ml_upload = True
            st.session_state.confirm_ml_delete = False

            # Clear any ML-specific session state
            for key in ['target_column', 'X', 'y', 'task', 'trained_model', 'uploaded_model',
                        'uploaded_model_filename', 'model_trained', 'selected_features',
                        'classification_report', 'accuracy', 'mse', 'selected_algorithm']:
                if key in st.session_state:
                    del st.session_state[key]

            st.sidebar.success("ML file deleted successfully!")
            save_state()
            st.rerun()
        if col2.button("No"):
            st.session_state.confirm_ml_delete = False
            st.rerun()


@st.cache_data
def process_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = preprocess_features(X)
    y, task = preprocess_target(y, df[target_column])
    return X, y, task


@st.cache_data
def load_and_preprocess_data(file_name):
    df = pd.read_csv(io.BytesIO(st.session_state.ml_uploaded_files[file_name]))
    return df


def ml_analysis_page():
    st.title("Machine Learning Analysis")

    if not st.session_state.ml_uploaded_files:
        upload_ml_file()
        st.info("Please upload a file to begin ML Analysis.")
        return

    file_name = choose_ml_file()

    if file_name == "Add new file":
        upload_ml_file()
        return

    if not file_name:
        upload_ml_file()
        st.info("Please upload a file to begin ML Analysis.")
        return

    load_ml_state(file_name)
    # Load data only if it's not already in the session state
    if 'df' not in st.session_state.file_contexts.get(file_name, {}):
        df = load_and_preprocess_data(file_name)
        st.session_state.file_contexts[file_name]['df'] = df
    else:
        df = st.session_state.file_contexts[file_name]['df']

    if 'original_dataset' not in st.session_state.file_contexts[file_name]:
        st.session_state.file_contexts[file_name]['original_dataset'] = df.copy()

    display_data_preview(df)

    st.subheader("Model Selection and Training")

    file_name, target_column = select_ml_target_variable()

    if target_column:
        # Check if we need to reprocess the data
        current_target = st.session_state.file_contexts[file_name].get('target_column')
        if current_target != target_column:
            X, y, task = process_data(df, target_column)

            st.session_state.file_contexts[file_name]['X'] = X
            st.session_state.file_contexts[file_name]['y'] = y
            st.session_state.file_contexts[file_name]['task'] = task
            st.session_state.file_contexts[file_name]['target_column'] = target_column
            save_ml_state(file_name)
        else:
            # Use cached data
            X = st.session_state.file_contexts[file_name]['X']
            y = st.session_state.file_contexts[file_name]['y']
            task = st.session_state.file_contexts[file_name]['task']

        select_and_train_model(X, y, task, target_column, file_name)

    st.subheader("Make Prediction")
    prediction_section(file_name)
