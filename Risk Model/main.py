import streamlit as st
from custom_transformers import CosineTransformer
from CosineTransform import CosineTransform
from dashboard import upload_file, choose_file, display_dashboard, delete_file, initialize_memory
from memory import save_state
from ml import ml_page

st.set_page_config(layout="wide")


def main():
    initialize_memory()
    # CosineTransformer()

    menu = ['Home', 'Dashboard', 'ML']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.title('Home')
        st.write("Welcome to the Dashboard and ML Analysis tool.")
        st.write("Use the menu on the left to navigate between pages.")
        st.write("- Dashboard: Visualize and explore your data: \n"
                 "      You are able to work with any dataset you want.")
        st.write("- ML: Perform frequency and machine learning analysis:  \n  \t"
                 "- Frequency Page:  \n  \t  \t"
                 "- Upload :orange[crss_frequency.csv], :orange[fars_frequency.csv], :orange[level_2.csv] and :orange[dmv.csv] files.  \n  \t"
                 "- Normalization Page:  \n  \t  \t"
                 "- Upload :orange[crss_normalization.json], :orange[fars_normalization.json] and :orange[level_2.json] files.  \n  \t"
                 "- Machine Learning Page (ML):  \n  \t  \t"
                 "- For :blue-background[dataset]:  \n  \t  \t  \t"
                 "- Upload :orange[crss_and_fars_ml.csv], :orange[level_2.csv], :orange[dmv.csv] files.  \n  \t  \t"
                 "- For :blue-background[model upload]  \n  \t  \t  \t"
                 "- Upload :orange[crss_and_fars_model.joblib], :orange[dmv_model.joblib],:orange[*level_2_model.joblib] files.")

    elif choice == 'Dashboard':
        st.sidebar.title("Dashboard")
        st.title("Dashboard")
        file_name = choose_file()

        if file_name == "Add new file":
            upload_file()
            st.info("Please choose a file from the sidebar or upload a new file to view its dashboard.")
            return

        if st.session_state.show_upload:
            st.title("Upload New File")
            upload_file()
        elif st.session_state.current_file:
            st.title(f"Dashboard - {st.session_state.current_file}")
            display_dashboard()
        else:
            st.title("Dashboard")
            st.info("Please choose a file from the sidebar or upload a new file to view its dashboard.")

        st.sidebar.markdown("## ")
        st.sidebar.markdown("## ")
        st.sidebar.markdown("## ")

        delete_file()

    elif choice == 'ML':

        ml_page()

    save_state()


if __name__ == '__main__':
    main()