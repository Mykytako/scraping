import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from memory import save_frequency_state, load_frequency_state, save_state
import io


def calculate_combined_frequency(data, columns, total_accidents, samples, scalar_value, scalar_type):
    frequency_states = load_frequency_state(st.session_state.current_file)

    key = f'combined_frequency_{",".join(columns)}_{total_accidents}_{samples}_{scalar_type}'

    if key not in frequency_states:
        combined_frequency = data.groupby(columns).size().reset_index(name='sample_accident_count')

        if scalar_type == "crss_scalar":
            # CRSS scalar: scale up from sample to population
            combined_frequency['estimated_total_accidents'] = combined_frequency['sample_accident_count'] * (
                        total_accidents / samples)
        elif scalar_type == "level2_scalar":
            # Level2 scalar: use the provided scalar value
            combined_frequency['estimated_total_accidents'] = combined_frequency['sample_accident_count'] * scalar_value
        else:  # "no_scalar"
            # No scalar: use sample counts as is
            combined_frequency['estimated_total_accidents'] = combined_frequency['sample_accident_count']

        # Calculate frequency as a proportion of total accidents
        combined_frequency['accident_frequency'] = combined_frequency['estimated_total_accidents'] / total_accidents

        # The total accident frequency should sum to 1 (or very close to 1 due to rounding)
        total_accident_frequency = combined_frequency['accident_frequency'].sum()

        frequency_states[key] = {
            'combined_frequency': combined_frequency,
            'total_accident_frequency': total_accident_frequency,
        }
        save_frequency_state(st.session_state.current_file, frequency_states)

    return frequency_states[key]['combined_frequency'], frequency_states[key]['total_accident_frequency']


@st.cache_data
def calculate_specific_frequency(data, columns, filter_conditions, total_accidents, samples, scalar_value):

    filtered_data = data.copy()
    for col in columns:
        if col in filter_conditions and col in filtered_data.columns:
            val = filter_conditions[col]
            if isinstance(val, list):
                if val:  # Only apply filter if the list is not empty
                    filtered_data = filtered_data[filtered_data[col].isin(val)]
            elif isinstance(val, tuple):
                filtered_data = filtered_data[(filtered_data[col] >= val[0]) & (filtered_data[col] <= val[1])]
            else:
                filtered_data = filtered_data[filtered_data[col] == val]

    if filtered_data.empty:
        return pd.DataFrame(), 0

    combined_frequency = filtered_data.groupby(columns).size().reset_index(name='sample_accident_count')

    combined_frequency['estimated_total_accidents'] = combined_frequency['sample_accident_count'] * scalar_value
    combined_frequency['accident_frequency'] = combined_frequency['estimated_total_accidents'] / total_accidents
    total_accident_frequency = combined_frequency['accident_frequency'].sum()

    return combined_frequency, total_accident_frequency


def calculate_feature_importance(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical variables
    for column in X.select_dtypes(include=['object']):
        X[column] = pd.Categorical(X[column]).codes

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return feature_importance


def display_data_info(data):
    st.write(f"Number of rows: {len(data)}")
    st.write(f"Number of columns: {len(data.columns)}")


def update_frequency_state():
    if st.session_state.current_frequency_file:
        frequency_states = load_frequency_state(st.session_state.current_frequency_file)
        st.session_state.user_selected_columns = frequency_states.get('selected_columns', [])
        st.session_state.filter_conditions = frequency_states.get('filter_conditions', {})
        st.session_state.feature_importance_target_column = frequency_states.get('feature_importance_target_column')
        st.session_state.selected_scalar = frequency_states.get('selected_scalar', 'crss_scalar')
        st.session_state.total_accidents = frequency_states.get('total_accidents', 5930000)


def feature_importance_section(data):
    st.header("Guidance for pre-selected columns")

    frequency_states = load_frequency_state(st.session_state.current_frequency_file)

    numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    default_target = frequency_states.get('feature_importance_target_column',
                                          numerical_columns[0] if numerical_columns else None)
    # Use the saved target column if available, otherwise use the default
    if 'feature_importance_target_column' not in st.session_state:
        st.session_state.feature_importance_target_column = default_target

    # Generate a stable key for the selectbox widget
    selectbox_key = f"target_column_selector_{st.session_state.current_file}"

    # Use st.selectbox with a callback function to update the session state
    def update_target_column():
        st.session_state.feature_importance_target_column = st.session_state[selectbox_key]
        st.session_state.target_changed = True
        frequency_states['feature_importance_target_column'] = st.session_state.feature_importance_target_column
        save_frequency_state(st.session_state.current_frequency_file, frequency_states)

    target_column = st.selectbox(
        "Select target column for frequency calculation",
        numerical_columns,
        index=numerical_columns.index(
            st.session_state.feature_importance_target_column) if st.session_state.feature_importance_target_column in numerical_columns else 0,
        key=selectbox_key,
        on_change=update_target_column
    )

    # Move the "Recalculate Feature Importance" button here
    recalculate_button = st.button("Calculate Recommended Columns", key="recalculate_feature_importance")

    if st.session_state.get('target_changed', False) or recalculate_button:
        frequency_states['feature_importance_target_column'] = st.session_state.feature_importance_target_column
        frequency_states['feature_importance_calculated'] = False
        frequency_states['feature_importance'] = None
        frequency_states['top_5_features'] = []
        st.session_state.target_changed = False

        # Recalculate feature importance immediately
        if target_column:
            try:
                feature_importance = calculate_feature_importance(data, target_column)

                if isinstance(feature_importance, pd.DataFrame):
                    frequency_states['feature_importance'] = feature_importance.to_dict('records')
                    frequency_states['top_5_features'] = feature_importance['feature'].head(5).tolist()
                    st.session_state.user_selected_columns = frequency_states[
                        'top_5_features']  # Update selected columns
                    frequency_states['feature_importance_calculated'] = True
                else:
                    st.error("Feature importance calculation returned unexpected result")
            except Exception as e:
                st.error(f"Feature importance calculation failed: {str(e)}")
        else:
            st.warning("Please select a target column for feature importance calculation.")

        save_frequency_state(st.session_state.current_frequency_file, frequency_states)
    return target_column


def combined_frequency_section(data, target_column):
    st.subheader("Frequency Calculations")
    frequency_states = load_frequency_state(st.session_state.current_frequency_file)

    available_columns = [col for col in data.columns if col != target_column]

    # Generate a stable key for the multiselect widget
    multiselect_key = f"columns_selector_{st.session_state.current_file}"

    # Initialize the session state for user selected columns if it doesn't exist
    if 'user_selected_columns' not in st.session_state:
        st.session_state.user_selected_columns = frequency_states.get('selected_columns', [])

    # Ensure columns_selected_updated is initialized
    if 'columns_selected_updated' not in st.session_state:
        st.session_state.columns_selected_updated = False

    # Filter out any columns that are no longer available
    st.session_state.user_selected_columns = [col for col in st.session_state.user_selected_columns if col in available_columns]

    # Use st.multiselect with a callback function to update the session state
    def update_selected_columns():
        st.session_state.user_selected_columns = st.session_state[columns_key]
        st.session_state.columns_selected_updated = True
        frequency_states['selected_columns'] = st.session_state.user_selected_columns
        save_frequency_state(st.session_state.current_frequency_file, frequency_states)

    columns_key = "selected_columns_" + multiselect_key
    if columns_key not in st.session_state:
        st.session_state[columns_key] = st.session_state.user_selected_columns

    st.warning('The following selected columns are for guidance only and helps frequency calculations.')

    st.multiselect(
        "Select columns for grouping",
        options=available_columns,
        default=st.session_state.user_selected_columns,
        key=columns_key,
        on_change=update_selected_columns
    )

    # Update the frequency state if the selection has changed
    if st.session_state.columns_selected_updated:
        st.session_state.columns_selected_updated = False
        frequency_states['selected_columns'] = st.session_state.user_selected_columns
        save_frequency_state(st.session_state.current_file, frequency_states)

    columns = st.session_state.user_selected_columns

    total_accidents = st.number_input("Total number of accidents",
                                      value=frequency_states.get('total_accidents', 5930000),
                                      key="total_accidents_input")

    samples = st.number_input("Number of samples",
                              value=len(data),
                              key="samples_input")
    st.info('Number of samples is taken from the amount of rows in the dataset.')
    scalar_options = {
        "crss_scalar": total_accidents / samples,
        "level2_scalar": 0.011904761904761904,
        "no_scalar": 1
    }

    # Generate a stable key for the selectbox widget
    selectbox_key = f"scalar_selector_{st.session_state.current_file}"

    # Initialize the session state for scalar selection if it doesn't exist
    if 'selected_scalar' not in st.session_state:
        st.session_state.selected_scalar = frequency_states.get('selected_scalar', "crss_scalar")

    # Ensure selected_scalar is a valid option
    if st.session_state.selected_scalar not in scalar_options:
        st.session_state.selected_scalar = "crss_scalar"  # Default to crss_scalar if invalid

    # Use st.selectbox with a callback function to update the session state
    def update_selected_scalar():
        st.session_state.selected_scalar = st.session_state[selectbox_key]
        st.session_state.scalar_changed = True

    selected_scalar = st.selectbox(
        "Select scalar",
        options=list(scalar_options.keys()),
        index=list(scalar_options.keys()).index(st.session_state.selected_scalar),
        key=selectbox_key,
        on_change=update_selected_scalar
    )

    st.info('Scalar: This dropdown will provide the correct scale number depending on the dataset. \
     Level 2 and CRSS will scale the number of accidents, \
      while FARS will not, as it represents the total fatal accidents in the US.')

    scalar_value = scalar_options[selected_scalar]

    if st.button("Calculate Combined Frequency", key="calculate_combined_frequency") or frequency_states.get(
            'combined_frequency_calculated', False):
        frequency_states['combined_frequency_calculated'] = True

        if len(columns) > 0:
            combined_frequency, total_accident_frequency = calculate_combined_frequency(
                data, columns, total_accidents, samples, scalar_value, selected_scalar)
            st.write("Combined Frequency Results:")
            st.dataframe(combined_frequency)

            st.warning('When the frequency calculation shows more than 2000 unique combinations, \
             the majority will have a small frequency, affecting the risk. Since the event is rare, \
              the risk will always be low. We recommend having fewer than 2000 unique combinations')

            # Save the total accident frequency to session state
            st.session_state.accident_frequency = total_accident_frequency

        else:
            st.warning("Please select at least one column for grouping.")

    # Save state after each interaction
    frequency_states['selected_columns'] = columns
    frequency_states['total_accidents'] = total_accidents
    frequency_states['samples'] = samples
    frequency_states['selected_scalar'] = selected_scalar
    frequency_states['scalar_value'] = scalar_value
    frequency_states['accident_frequency'] = st.session_state.get('accident_frequency')
    save_frequency_state(st.session_state.current_frequency_file, frequency_states)

    return columns, total_accidents, samples, scalar_value, selected_scalar


def specific_frequency_section(data, columns, total_accidents, samples, scalar_value, selected_scalar):
    st.subheader("Specific Frequency Calculation")

    frequency_states = load_frequency_state(st.session_state.current_frequency_file)

    # Initialize filter_conditions in session state if it doesn't exist
    if 'filter_conditions' not in st.session_state:
        st.session_state.filter_conditions = frequency_states.get('filter_conditions', {})

    # Remove any filter conditions for columns that don't exist in the current dataset
    st.session_state.filter_conditions = {col: val for col, val in st.session_state.filter_conditions.items() if col in data.columns}

    col1, col2 = st.columns(2)
    for i, col in enumerate(columns):
        if col in data.columns:  # Only create filter conditions for columns that exist in the dataset
            if i % 2 == 0:
                with col1:
                    add_filter_condition(data, col)
            else:
                with col2:
                    add_filter_condition(data, col)

    if 'specific_frequency_results' not in st.session_state:
        st.session_state.specific_frequency_results = None

    if st.button("Calculate Specific Frequency", key="calculate_specific_frequency"):
        st.session_state.specific_frequency_results = calculate_specific_frequency(
            data, columns, st.session_state.filter_conditions, total_accidents, samples, scalar_value
        )

    if st.session_state.specific_frequency_results is not None:
        specific_frequency, specific_total_accident_frequency = st.session_state.specific_frequency_results
        st.write("Specific Frequency Results:")
        st.dataframe(specific_frequency)

    # Save state after each interaction
    frequency_states['filter_conditions'] = st.session_state.filter_conditions
    save_frequency_state(st.session_state.current_frequency_file, frequency_states)


def add_filter_condition(data, col):
    frequency_states = load_frequency_state(st.session_state.current_frequency_file)

    unique_values = data[col].dropna().unique()
    if len(unique_values) > 100:
        unique_values = sorted(unique_values)[:100]

    # Generate a stable key for the multiselect widget
    multiselect_key = f"filter_{col}_{st.session_state.current_file}"

    # Initialize the session state for this column's filter if it doesn't exist
    if multiselect_key not in st.session_state:
        st.session_state[multiselect_key] = st.session_state.filter_conditions.get(col, [])

    # Use st.multiselect with a callback function to update the session state
    def update_filter_condition():
        st.session_state.filter_conditions[col] = st.session_state[multiselect_key]
        frequency_states['filter_conditions'] = st.session_state.filter_conditions
        save_frequency_state(st.session_state.current_file, frequency_states)
        # Clear the specific frequency results when filter changes
        st.session_state.specific_frequency_results = None

    selected_values = st.multiselect(
        f"Select values for {col}",
        options=unique_values,
        default=st.session_state[multiselect_key],
        key=multiselect_key,
        on_change=update_filter_condition
    )

    # Update the filter conditions
    if selected_values:
        st.session_state.filter_conditions[col] = selected_values
    elif col in st.session_state.filter_conditions:
        del st.session_state.filter_conditions[col]

    # Save state after each interaction
    frequency_states['filter_conditions'] = st.session_state.filter_conditions
    save_frequency_state(st.session_state.current_frequency_file, frequency_states)


def on_frequency_file_change():
    selected_file = st.session_state.frequency_file_selector
    if selected_file == "Add new file":
        st.session_state.show_frequency_upload = True
        st.session_state.current_frequency_file = None
    else:
        st.session_state.current_frequency_file = selected_file
        st.session_state.show_frequency_upload = False
        update_frequency_state()  # Add this line
    save_state()


def choose_frequency_file():
    if 'frequency_uploaded_files' not in st.session_state:
        st.session_state.frequency_uploaded_files = {}

    files = list(st.session_state.frequency_uploaded_files.keys()) + ["Add new file"]
    index = files.index(
        st.session_state.current_frequency_file) if st.session_state.current_frequency_file in files else len(files) - 1

    st.sidebar.selectbox(
        "Choose File for Frequency Analysis",
        files,
        index=index,
        key="frequency_file_selector",
        on_change=on_frequency_file_change
    )


def upload_frequency_file():
    uploaded_file = st.file_uploader("Choose a CSV file for Frequency Analysis", type="csv",
                                     key="frequency_file_uploader")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_content = uploaded_file.read()

        if 'frequency_uploaded_files' not in st.session_state:
            st.session_state.frequency_uploaded_files = {}

        st.session_state.frequency_uploaded_files[file_name] = file_content
        st.success(f"File '{file_name}' uploaded successfully!")
        st.session_state.current_frequency_file = file_name
        st.session_state.show_frequency_upload = False

        # Initialize frequency state for the new file
        save_frequency_state(file_name, {})
        save_state()
        st.rerun()


def delete_frequency_file():
    if st.session_state.current_frequency_file and st.session_state.current_frequency_file in st.session_state.frequency_uploaded_files:
        if st.sidebar.button("Delete Current File"):
            st.session_state.confirm_frequency_delete = True

    if st.session_state.get('confirm_frequency_delete', False):
        st.sidebar.warning("Are you sure you want to delete this frequency file?")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Yes"):
            # Delete the file from frequency_uploaded_files
            del st.session_state.frequency_uploaded_files[st.session_state.current_frequency_file]

            # Remove associated frequency state
            if st.session_state.current_frequency_file in st.session_state.get('frequency_states', {}):
                del st.session_state.frequency_states[st.session_state.current_frequency_file]

            # Reset current file and show upload
            st.session_state.current_frequency_file = None
            st.session_state.show_frequency_upload = True
            st.session_state.confirm_frequency_delete = False

            # Clear any frequency-specific session state
            if 'user_selected_columns' in st.session_state:
                del st.session_state.user_selected_columns
            if 'filter_conditions' in st.session_state:
                del st.session_state.filter_conditions
            if 'specific_frequency_results' in st.session_state:
                del st.session_state.specific_frequency_results

            st.sidebar.success("Frequency file deleted successfully!")
            save_state()
            st.rerun()
        if col2.button("No"):
            st.session_state.confirm_frequency_delete = False
            st.rerun()


def frequency_page():
    st.title("Feature Importance and Frequency Calculations")

    choose_frequency_file()

    if st.session_state.show_frequency_upload:
        upload_frequency_file()
        return

    if not st.session_state.current_frequency_file:
        upload_frequency_file()
        st.info("Please upload a file to begin Frequency Analysis.")
        return

    file_content = st.session_state.frequency_uploaded_files[st.session_state.current_frequency_file]
    data = pd.read_csv(io.BytesIO(file_content))

    frequency_states = load_frequency_state(st.session_state.current_frequency_file)

    # Load saved filter conditions
    if 'filter_conditions' in frequency_states:
        st.session_state.filter_conditions = frequency_states['filter_conditions']

    display_data_info(data)

    target_column = feature_importance_section(data)

    columns, total_accidents, samples, scalar_value, selected_scalar = combined_frequency_section(data, target_column)

    specific_frequency_section(data, columns, total_accidents, samples, scalar_value, selected_scalar)

    save_frequency_state(st.session_state.current_frequency_file, frequency_states)