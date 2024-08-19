import io
import pandas as pd
import streamlit as st
from frequency import frequency_page, delete_frequency_file
from memory import initialize_memory
from normalization import normalization_page, delete_normalization_file
from ml_analysis import ml_analysis_page, delete_ml_file
from CosineTransform import CosineTransform
from risk_analysis import risk_analysis_page


def load_and_preprocess_data(file_content):
    file_object = io.BytesIO(file_content)
    df = pd.read_csv(file_object, on_bad_lines='skip', low_memory=False)
    return df


def ml_page():
    initialize_memory()

    # Subpage selection
    subpages = ["Frequency Analysis", "Normalization", "Machine Learning Analysis", 'Risk Analysis']
    selected_subpage = st.sidebar.selectbox("Select Analysis Type", subpages)

    # Display selected subpage
    if selected_subpage == "Frequency Analysis":
        st.sidebar.title("Frequency File Selection")
        frequency_page()

        st.sidebar.markdown("## ")
        st.sidebar.markdown("## ")
        st.sidebar.markdown("## ")

        delete_frequency_file()

    elif selected_subpage == "Normalization":
        normalization_page()

        st.sidebar.markdown("## ")
        st.sidebar.markdown("## ")
        st.sidebar.markdown("## ")

        delete_normalization_file()

    elif selected_subpage == "Machine Learning Analysis":
        # File selection in sidebar only for ML Analysis
        st.sidebar.title("ML File Selection")
        ml_analysis_page()

        st.sidebar.markdown("## ")
        st.sidebar.markdown("## ")
        st.sidebar.markdown("## ")

        delete_ml_file()

    else:
        st.subheader('Risk Analysis Page')
        risk_analysis_page()


if __name__ == "__main__":
    ml_page()