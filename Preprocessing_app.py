import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pickle

# Define a function to handle missing values
def handle_missing_values(df, column, method, custom_value=None):
    try:
        col_type = df[column].dtype
        if method == "Drop":
            df.dropna(subset=[column], inplace=True)
        elif method == "Fill with Mean":
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == "Fill with Mode":
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif method == "Custom Input":
            if custom_value is not None:
                if col_type == 'object':
                    df[column].fillna(custom_value, inplace=True)
                elif col_type in ['int64', 'float64']:
                    if isinstance(custom_value, int):
                        df[column].fillna(custom_value, inplace=True)
                        if col_type == 'int64':
                            df[column] = df[column].astype(int)
                    elif isinstance(custom_value, float):
                        df[column].fillna(custom_value, inplace=True)
                        df[column] = df[column].astype(float)
                    else:
                        df[column] = df[column].astype('object')
                        df[column].fillna(custom_value, inplace=True)
                else:
                    st.write("Unsupported column type")
            else:
                st.write("Please provide a custom value")
        # Convert column type back if necessary
        if col_type in ['int64', 'float64']:
            try:
                if col_type == 'int64' and df[column].dropna().apply(float.is_integer).all():
                    df[column] = df[column].astype(int)
                elif col_type == 'float64':
                    df[column] = df[column].astype(float)
            except:
                pass
        return True
    except Exception as e:
        st.write(f"Please choose a different option. Missing values have not been handled. Error: {e}")
        return False

# Define a function to cap outliers
def cap_outliers(df, column, factor):
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    top_boundary = q3 + factor * iqr
    bottom_boundary = q1 - factor * iqr
    df[column] = np.where(df[column] > top_boundary, top_boundary, df[column])
    df[column] = np.where(df[column] < bottom_boundary, bottom_boundary, df[column])

def save_progress():
    with open('progress.pkl', 'wb') as f:
        pickle.dump({
            'data': st.session_state.data,
            'mapping': st.session_state.mapping,
            'label_encoded_cols': st.session_state.label_encoded_cols,
            'onehot_encoded_cols': st.session_state.onehot_encoded_cols,
            'handled_missing': st.session_state.handled_missing,
            'capped_outliers': st.session_state.capped_outliers
        }, f)

def load_progress():
    try:
        with open('progress.pkl', 'rb') as f:
            progress = pickle.load(f)
            st.session_state.data = progress.get('data', None)
            st.session_state.mapping = progress.get('mapping', {})
            st.session_state.label_encoded_cols = progress.get('label_encoded_cols', [])
            st.session_state.onehot_encoded_cols = progress.get('onehot_encoded_cols', [])
            st.session_state.handled_missing = progress.get('handled_missing', [])
            st.session_state.capped_outliers = progress.get('capped_outliers', [])
    except FileNotFoundError:
        pass

def reset_to_original():
    try:
        with open('original_dataset.pkl', 'rb') as f:
            st.session_state.data = pickle.load(f)
        st.session_state.mapping = {}
        st.session_state.label_encoded_cols = []
        st.session_state.onehot_encoded_cols = []
        st.session_state.handled_missing = []
        st.session_state.capped_outliers = []
        save_progress()
        st.write("Reset to the original dataset.")
    except FileNotFoundError:
        st.write("Original dataset not found. Please upload a dataset first.")

def apply_mapping(df, column, mapping):
    try:
        col_type = df[column].dtype
        for key, value in mapping.items():
            if value != "":
                df.loc[df[column] == key, column] = value
        # Convert column type back if necessary
        if col_type in ['int64', 'float64']:
            try:
                if col_type == 'int64' and df[column].dropna().apply(float.is_integer).all():
                    df[column] = df[column].astype(int)
                elif col_type == 'float64':
                    df[column] = df[column].astype(float)
            except:
                pass
        return True
    except Exception as e:
        st.write(f"Mapping could not be applied. Error: {e}")
        return False

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

if 'mapping' not in st.session_state:
    st.session_state.mapping = {}

if 'label_encoded_cols' not in st.session_state:
    st.session_state.label_encoded_cols = []

if 'onehot_encoded_cols' not in st.session_state:
    st.session_state.onehot_encoded_cols = []

if 'handled_missing' not in st.session_state:
    st.session_state.handled_missing = []

if 'capped_outliers' not in st.session_state:
    st.session_state.capped_outliers = []

# Load progress from file
load_progress()

def main():
    st.title("Preprocessing Dashboard")

    # Upload dataset
    file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if file is not None:
        if file.name.endswith('.csv'):
            st.session_state.data = pd.read_csv(file)
        else:
            st.session_state.data = pd.read_excel(file)
        with open('original_dataset.pkl', 'wb') as f:
            pickle.dump(st.session_state.data, f)
        save_progress()

    # Add reset button
    if st.button("Reset to Original Dataset"):
        reset_to_original()

    if st.session_state.data is not None:
        df = st.session_state.data.copy()

        # Apply existing mappings
        for col, mapping in st.session_state.mapping.items():
            if col in df.columns:
                apply_mapping(df, col, mapping)

        # Apply existing label encodings
        if st.session_state.label_encoded_cols:
            le = LabelEncoder()
            for col in st.session_state.label_encoded_cols:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col])

        # Apply existing one hot encodings
        if st.session_state.onehot_encoded_cols:
            existing_onehot_cols = [col for col in st.session_state.onehot_encoded_cols if col in df.columns]
            df = pd.get_dummies(df, columns=existing_onehot_cols)

        # Apply existing missing value handling
        for (col, method, custom_value) in st.session_state.handled_missing:
            if col in df.columns:
                handle_missing_values(df, col, method, custom_value)

        # Apply existing outlier capping
        for (col, factor) in st.session_state.capped_outliers:
            if col in df.columns:
                cap_outliers(df, col, factor)

        # Add custom CSS for tooltips
        st.markdown("""
            <style>
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: pointer;
                color: blue;
            }

            .tooltip .tooltiptext {
                visibility: hidden;
                width: 220px;
                background-color: black;
                color: #fff;
                text-align: center;
                border-radius: 5px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%; /* Position the tooltip above the text */
                left: 50%;
                margin-left: -110px;
                opacity: 0;
                transition: opacity 0.3s;
            }

            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display EDA
        st.header("Exploratory Data Analysis")

        # Shape of dataset
        st.markdown('<div>Shape of the dataset <span class="tooltip">ℹ️<span class="tooltiptext">This shows the number of rows and columns in the dataset.</span></span></div>', unsafe_allow_html=True)
        st.write(df.shape)

        # Summary of dataset
        st.markdown('<div>Summary of the dataset <span class="tooltip">ℹ️<span class="tooltiptext">This provides basic statistics of the dataset including mean, standard deviation, and percentiles.</span></span></div>', unsafe_allow_html=True)
        st.write(df.describe())

        # Data types
        st.markdown('<div>Data Types <span class="tooltip">ℹ️<span class="tooltiptext">This shows the data types of each column in the dataset.</span></span></div>', unsafe_allow_html=True)
        st.write(df.dtypes)

        # Missing values and zeros
        st.markdown('<div>Missing Values and Zeros <span class="tooltip">ℹ️<span class="tooltiptext">This shows the number of missing values and zeros for each column in the dataset.</span></span></div>', unsafe_allow_html=True)
        missing_and_zeros = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Zeros': (df == 0).sum()
        })
        st.write(missing_and_zeros)

        # Number of duplicates
        st.markdown('<div>Number of Duplicates <span class="tooltip">ℹ️<span class="tooltiptext">This shows the number of duplicate rows in the dataset.</span></span></div>', unsafe_allow_html=True)
        st.write(df.duplicated().sum())

        # Detect outliers
        st.markdown('<div>Outlier Detection <span class="tooltip">ℹ️<span class="tooltiptext">This identifies columns with outliers using the IQR method.</span></span></div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        outlier_columns = []
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            top_boundary = q3 + 1.5 * iqr
            bottom_boundary = q1 - 1.5 * iqr
            if ((df[col] < bottom_boundary) | (df[col] > top_boundary)).any():
                outlier_columns.append(col)

        st.write("Columns with outliers:", outlier_columns)
        selected_outlier_col = st.selectbox("Select a column to view outliers", outlier_columns)
        if selected_outlier_col:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_outlier_col], ax=ax)
            st.pyplot(fig)

        # First few rows
        st.markdown('<div>First Few Rows <span class="tooltip">ℹ️<span class="tooltiptext">This displays the first few rows of the dataset.</span></span></div>', unsafe_allow_html=True)
        st.write(df.head())

        # Value counts
        st.markdown('<div>Value Counts of Selected Column <span class="tooltip">ℹ️<span class="tooltiptext">This shows the value counts for a selected column in the dataset.</span></span></div>', unsafe_allow_html=True)
        selected_val_count_col = st.selectbox("Select a column for value counts", df.columns)
        if selected_val_count_col:
            st.write(df[selected_val_count_col].value_counts())
            fig, ax = plt.subplots()
            df[selected_val_count_col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        # Preprocessing steps
        st.header("Preprocessing Steps")

        # Fill or drop missing values
        st.markdown('<div>Handle Missing Values <span class="tooltip">ℹ️<span class="tooltiptext">This handles missing values by dropping or filling them with specified methods.</span></span></div>', unsafe_allow_html=True)
        missing_cols = df.columns[df.isnull().any()].tolist()
        selected_missing_col = st.selectbox("Select column with missing values", missing_cols)
        if selected_missing_col:
            missing_method = st.selectbox("Select method to handle missing values", ["Drop", "Fill with Mean", "Fill with Mode", "Custom Input"])
            custom_value = None
            if missing_method == "Custom Input":
                custom_value = st.text_input("Enter custom value")
            if st.button("Handle Missing Values"):
                if handle_missing_values(df, selected_missing_col, missing_method, custom_value):
                    st.session_state.data = df.copy()
                    st.session_state.handled_missing.append((selected_missing_col, missing_method, custom_value))
                    save_progress()
                    st.write("Missing values handled.")

        # Encode selected columns
        st.markdown('<div>Encode Columns <span class="tooltip">ℹ️<span class="tooltiptext">This encodes categorical columns using Label Encoder or One Hot Encoder.</span></span></div>', unsafe_allow_html=True)
        label_encode_cols = st.multiselect("Select columns to encode with Label Encoder", df.columns)
        onehot_encode_cols = st.multiselect("Select columns to encode with One Hot Encoder", df.columns)
        if st.button("Encode Columns"):
            if label_encode_cols:
                le = LabelEncoder()
                for col in label_encode_cols:
                    df[col] = le.fit_transform(df[col])
                    if col not in st.session_state.label_encoded_cols:
                        st.session_state.label_encoded_cols.append(col)
            if onehot_encode_cols:
                df = pd.get_dummies(df, columns=onehot_encode_cols)
                for col in onehot_encode_cols:
                    if col not in st.session_state.onehot_encoded_cols:
                        st.session_state.onehot_encoded_cols.append(col)
            st.session_state.data = df.copy()
            save_progress()
            st.write("Columns encoded.")

        # Cap outliers
        st.markdown('<div>Cap Outliers <span class="tooltip">ℹ️<span class="tooltiptext">This caps outliers in selected columns using the specified factor.</span></span></div>', unsafe_allow_html=True)
        selected_outlier_cols = st.multiselect("Select columns to cap outliers", outlier_columns)
        if selected_outlier_cols:
            outlier_factor = st.slider("Select outlier capping factor", 1.0, 3.0, 1.5)
            if st.button("Cap Outliers"):
                for col in selected_outlier_cols:
                    cap_outliers(df, col, outlier_factor)
                    st.session_state.capped_outliers.append((col, outlier_factor))
                st.session_state.data = df.copy()
                save_progress()
                st.write("Outliers capped.")

        # Map custom values
        st.markdown('<div>Map Custom Values <span class="tooltip">ℹ️<span class="tooltiptext">This maps custom values to unique entries in the selected column.</span></span></div>', unsafe_allow_html=True)
        map_col = st.selectbox("Select column to map values", df.columns)
        if map_col:
            if map_col not in st.session_state.mapping:
                st.session_state.mapping[map_col] = {}
            map_dict = st.session_state.mapping[map_col]
            unique_values = df[map_col].unique().tolist()
            with st.expander("Enter new values for unique entries"):
                new_values = []
                for val in unique_values:
                    new_value = st.text_input(f"Enter new value for {val}", key=f"map_{map_col}_{val}")
                    if new_value:
                        new_values.append((val, new_value))
            if st.button("Map Values"):
                for old_val, new_val in new_values:
                    if new_val:
                        map_dict[old_val] = new_val
                st.session_state.mapping[map_col] = map_dict
                apply_mapping(df, map_col, map_dict)
                st.session_state.data = df.copy()
                save_progress()
                st.write("Values mapped.")

        # Scale selected columns
        st.markdown('<div>Scale Columns <span class="tooltip">ℹ️<span class="tooltiptext">This scales selected columns using Standard Scaler or Min Max Scaler.</span></span></div>', unsafe_allow_html=True)
        scale_cols = st.multiselect("Select columns to scale", df.select_dtypes(include=['float64', 'int64']).columns.tolist())
        if scale_cols:
            scaler_method = st.selectbox("Select scaler method", ["Standard Scaler", "Min Max Scaler"])
            if st.button("Scale Columns"):
                if scaler_method == "Standard Scaler":
                    scaler = StandardScaler()
                elif scaler_method == "Min Max Scaler":
                    scaler = MinMaxScaler()
                df[scale_cols] = scaler.fit_transform(df[scale_cols])
                st.session_state.data = df.copy()
                save_progress()
                st.write("Columns scaled.")

        # Drop duplicates
        st.markdown('<div>Drop Duplicates <span class="tooltip">ℹ️<span class="tooltiptext">This removes duplicate rows from the dataset.</span></span></div>', unsafe_allow_html=True)
        if st.button("Drop Duplicates"):
            df.drop_duplicates(inplace=True)
            st.session_state.data = df.copy()
            save_progress()
            st.write("Duplicates dropped. Current shape:", df.shape)

        # Display preprocessed EDA
        st.header("Preprocessed Exploratory Data Analysis")

        st.markdown('<div>Shape of the dataset <span class="tooltip">ℹ️<span class="tooltiptext">This shows the number of rows and columns in the dataset.</span></span></div>', unsafe_allow_html=True)
        st.write(df.shape)

        st.markdown('<div>Summary of the dataset <span class="tooltip">ℹ️<span class="tooltiptext">This provides basic statistics of the dataset including mean, standard deviation, and percentiles.</span></span></div>', unsafe_allow_html=True)
        st.write(df.describe())

        st.markdown('<div>Data Types <span class="tooltip">ℹ️<span class="tooltiptext">This shows the data types of each column in the dataset.</span></span></div>', unsafe_allow_html=True)
        st.write(df.dtypes)

        st.markdown('<div>Missing Values and Zeros <span class="tooltip">ℹ️<span class="tooltiptext">This shows the number of missing values and zeros for each column in the dataset.</span></span></div>', unsafe_allow_html=True)
        missing_and_zeros = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Zeros': (df == 0).sum()
        })
        st.write(missing_and_zeros)

        st.markdown('<div>Number of Duplicates <span class="tooltip">ℹ️<span class="tooltiptext">This shows the number of duplicate rows in the dataset.</span></span></div>', unsafe_allow_html=True)
        st.write(df.duplicated().sum())

        st.markdown('<div>Outlier Detection <span class="tooltip">ℹ️<span class="tooltiptext">This identifies columns with outliers using the IQR method.</span></span></div>', unsafe_allow_html=True)
        outlier_columns = []
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            top_boundary = q3 + 1.5 * iqr
            bottom_boundary = q1 - 1.5 * iqr
            if ((df[col] < bottom_boundary) | (df[col] > top_boundary)).any():
                outlier_columns.append(col)
        st.write("Columns with outliers:", outlier_columns)
        
        # Add box plot in the Preprocessed EDA section
        selected_outlier_col_preprocessed = st.selectbox("Select a column to view outliers (after preprocessing)", outlier_columns)
        if selected_outlier_col_preprocessed:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_outlier_col_preprocessed], ax=ax)
            st.pyplot(fig)

        st.markdown('<div>First Few Rows <span class="tooltip">ℹ️<span class="tooltiptext">This displays the first few rows of the dataset.</span></span></div>', unsafe_allow_html=True)
        st.write(df.head())

        st.markdown('<div>Value Counts of Selected Column <span class="tooltip">ℹ️<span class="tooltiptext">This shows the value counts for a selected column in the dataset.</span></span></div>', unsafe_allow_html=True)
        selected_val_count_col = st.selectbox("Select a column for value counts (after preprocessing)", df.columns)
        if selected_val_count_col:
            st.write(df[selected_val_count_col].value_counts())
            fig, ax = plt.subplots()
            df[selected_val_count_col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        # Download preprocessed file
        st.markdown('<div>Download Preprocessed File <span class="tooltip">ℹ️<span class="tooltiptext">This allows you to download the preprocessed dataset as a CSV file.</span></span></div>', unsafe_allow_html=True)
        if st.button("Download Preprocessed File"):
            st.download_button(
                label="Download Preprocessed File",
                data=df.to_csv(index=False),
                file_name='preprocessed_data.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
