import streamlit as st
import json
from memory import load_normalization_state, save_normalization_state, save_normalization_uploaded_files, \
    load_normalization_uploaded_files, save_normalization_settings, save_state


def save_current_normalization_state():
    if st.session_state.current_normalization_file:
        save_normalization_state(st.session_state.current_normalization_file, {
            'frequency_columns': st.session_state.get(f"freq_cols_{st.session_state.current_normalization_file}", []),
            'selected_locations': st.session_state.get(f"locations_{st.session_state.current_normalization_file}", []),
            'selected_months': st.session_state.get(f"months_{st.session_state.current_normalization_file}", []),
            'calculation_results': st.session_state.get('calculation_results', [])
        })
        save_state()


def on_normalization_file_change():
    selected_file = st.session_state.normalization_file_selector
    if selected_file == "Add new file":
        st.session_state.show_normalization_upload = True
        st.session_state.current_normalization_file = None
    else:
        # Save the current state before changing files
        save_current_normalization_state()

        st.session_state.current_normalization_file = selected_file
        st.session_state.show_normalization_upload = False

        # Load the state for the newly selected file
        state = load_normalization_state(selected_file)
        st.session_state[f"freq_cols_{selected_file}"] = state.get('frequency_columns', [])
        st.session_state[f"locations_{selected_file}"] = state.get('selected_locations', [])
        st.session_state[f"months_{selected_file}"] = state.get('selected_months', [])
        st.session_state['calculation_results'] = state.get('calculation_results', [])

    save_state()


def choose_normalization_file():
    if 'normalization_uploaded_files' not in st.session_state:
        st.session_state.normalization_uploaded_files = {}

    files = list(st.session_state.normalization_uploaded_files.keys()) + ["Add new file"]
    index = files.index(st.session_state.get('current_normalization_file')) if st.session_state.get(
        'current_normalization_file') in files else len(files) - 1

    st.sidebar.selectbox(
        "Choose File for Normalization",
        files,
        index=index,
        key="normalization_file_selector",
        on_change=on_normalization_file_change
    )


def upload_normalization_file():
    uploaded_file = st.file_uploader("Choose a JSON file for Normalization", type="json", key="normalization_file_uploader")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_content = uploaded_file.read()

        normalization_uploaded_files = load_normalization_uploaded_files()
        normalization_uploaded_files[file_name] = file_content
        save_normalization_uploaded_files(normalization_uploaded_files)

        normalization_dict = json.loads(file_content.decode('utf-8'))
        save_normalization_settings(file_name, normalization_dict)

        st.success(f"File '{file_name}' uploaded successfully!")
        st.session_state.current_normalization_file = file_name
        st.session_state.show_normalization_upload = False
        save_normalization_state(file_name, {})
        save_state()
        st.rerun()


def update_selection(key):
    file = st.session_state.current_normalization_file
    state = load_normalization_state(file)
    session_key = f"{key}_{file}"
    if session_key in st.session_state:
        state[key] = st.session_state[session_key]
        save_normalization_state(file, state)
    save_current_normalization_state()


def normalized_section(normalization_settings):
    st.subheader("Normalized Frequency Calculation")

    if not normalization_settings:
        st.warning("No normalization settings found. Please upload a normalization JSON file.")
        return

    # Load the current normalization state
    normalization_state = load_normalization_state(st.session_state.current_normalization_file)

    # Hard-coded frequency columns
    frequency_columns = ['Daylight_Frequency', 'Dark_Frequency', 'Clear', 'Rain', 'Snow']

    # Allow user to choose multiple frequency columns
    freq_cols_key = f"freq_cols_{st.session_state.current_normalization_file}"
    selected_frequency_columns = st.multiselect(
        "Select frequency columns",
        frequency_columns,
        default=normalization_state.get('frequency_columns', []),
        key=freq_cols_key,
        on_change=update_selection,
        args=('frequency_columns',)
    )

    # Determine if we're using 'Region' or 'States'
    location_column = 'Region' if 'Region' in normalization_settings else 'State'

    # Create multiselect for Region/States and Month
    location_options = sorted(list(set(normalization_settings[location_column].values())))
    month_options = sorted(list(set(normalization_settings['Month'].values())))

    locations_key = f"locations_{st.session_state.current_normalization_file}"
    selected_locations = st.multiselect(
        f"Select {location_column}s",
        location_options,
        default=normalization_state.get('selected_locations', []),
        key=locations_key,
        on_change=update_selection,
        args=('selected_locations',)
    )

    months_key = f"months_{st.session_state.current_normalization_file}"
    selected_months = st.multiselect(
        "Select Months",
        month_options,
        default=normalization_state.get('selected_months', []),
        key=months_key,
        on_change=update_selection,
        args=('selected_months',)
    )

    if st.button("Calculate Normalized Frequencies",
                 key=f"calc_norm_freq_{st.session_state.current_normalization_file}"):
        results = []
        for location in selected_locations:
            for month in selected_months:
                for freq_column in selected_frequency_columns:
                    # Find the index that matches the selected location and month
                    selected_index = None
                    for index, (loc, mon) in enumerate(zip(normalization_settings[location_column].values(),
                                                           normalization_settings['Month'].values())):
                        if loc == location and int(mon) == int(month):
                            selected_index = str(index)
                            break

                    if selected_index is not None:
                        frequency = normalization_settings[freq_column][selected_index]
                        result = f"{freq_column} for {location} in month {month}: {round(frequency)} accidents per 100 million miles."
                        results.append(result)
                    else:
                        results.append(f"No matching data found for {location} in month {month} for {freq_column}.")

        st.session_state['calculation_results'] = results
        for result in results:
            st.write(result)

        save_current_normalization_state()


def delete_normalization_file():
    if st.session_state.current_normalization_file and st.session_state.current_normalization_file in st.session_state.normalization_uploaded_files:
        if st.sidebar.button("Delete Current File"):
            st.session_state.confirm_normalization_delete = True

    if st.session_state.get('confirm_normalization_delete', False):
        st.sidebar.warning("Are you sure you want to delete this normalization file?")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Yes"):
            # Delete the file from normalization_uploaded_files
            del st.session_state.normalization_uploaded_files[st.session_state.current_normalization_file]

            # Remove associated normalization state
            if st.session_state.current_normalization_file in st.session_state.get('normalization_states', {}):
                del st.session_state.normalization_states[st.session_state.current_normalization_file]

            # Reset current file and show upload
            st.session_state.current_normalization_file = None
            st.session_state.show_normalization_upload = True
            st.session_state.confirm_normalization_delete = False

            # Clear any normalization-specific session state
            if 'calculation_result' in st.session_state:
                del st.session_state.calculation_result

            st.sidebar.success("Normalization file deleted successfully!")
            save_state()
            st.rerun()
        if col2.button("No"):
            st.session_state.confirm_normalization_delete = False
            st.rerun()


def normalization_page():
    st.title("Data Normalization")

    choose_normalization_file()

    if not st.session_state.current_normalization_file:
        upload_normalization_file()
        st.info("Please upload a file to begin Normalization.")
        return

    if 'normalization_uploaded_files' not in st.session_state:
        st.session_state.normalization_uploaded_files = {}

    file_content = st.session_state.normalization_uploaded_files[st.session_state.current_normalization_file]
    normalization_settings = json.loads(file_content.decode('utf-8'))

    normalized_section(normalization_settings)