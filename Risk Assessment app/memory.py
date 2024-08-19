import os
import pickle
import pandas as pd
import streamlit as st

# File to store the serialized session state
STATE_FILE = 'app_state.pkl'


def save_state():
    """Save the current session state to a file."""
    state_to_save = {
        'uploaded_files': st.session_state.uploaded_files,
        'current_file': st.session_state.current_file,
        'show_upload': st.session_state.show_upload,
        'last_selected_file': st.session_state.last_selected_file,
        'file_configs': st.session_state.file_configs,
        'charts_per_row': st.session_state.charts_per_row,
        'normalization_settings': st.session_state.get('normalization_settings', {}),
        'frequency_states': st.session_state.get('frequency_states', {}),
        'frequency_uploaded_files': st.session_state.get('frequency_uploaded_files', {}),
        'current_frequency_file': st.session_state.get('current_frequency_file'),
        'show_frequency_upload': st.session_state.get('show_frequency_upload', False),
        'ml_states': st.session_state.get('ml_states', {}),
        'ml_uploaded_files': st.session_state.get('ml_uploaded_files', {}),
        'current_ml_file': st.session_state.get('current_ml_file'),
        'show_ml_upload': st.session_state.get('show_ml_upload', False),
        'file_contexts': st.session_state.file_contexts,
        'normalization_states': st.session_state.get('normalization_states', {}),
        'normalization_uploaded_files': st.session_state.get('normalization_uploaded_files', {}),
        'current_normalization_file': st.session_state.get('current_normalization_file'),
        'show_normalization_upload': st.session_state.get('show_normalization_upload', False),
        'risk_analysis_state': st.session_state.get('risk_analysis_state', {}),

    }
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(state_to_save, f)


def load_state():
    """Load the session state from a file if it exists."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'rb') as f:
            saved_state = pickle.load(f)

        st.session_state.uploaded_files = saved_state.get('uploaded_files', {})
        st.session_state.current_file = saved_state.get('current_file')
        st.session_state.show_upload = saved_state.get('show_upload', False)
        st.session_state.last_selected_file = saved_state.get('last_selected_file')
        st.session_state.file_configs = saved_state.get('file_configs', {})
        st.session_state.charts_per_row = saved_state.get('charts_per_row', 2)
        st.session_state.normalization_settings = saved_state.get('normalization_settings', {})
        st.session_state.frequency_states = saved_state.get('frequency_states', {})
        st.session_state.datasets = saved_state.get('datasets', {})
        st.session_state.frequency_uploaded_files = saved_state.get('frequency_uploaded_files', {})
        st.session_state.current_frequency_file = saved_state.get('current_frequency_file')
        st.session_state.show_frequency_upload = saved_state.get('show_frequency_upload', False)
        st.session_state.ml_states = saved_state.get('ml_states', {})
        st.session_state.ml_uploaded_files = saved_state.get('ml_uploaded_files', {})
        st.session_state.current_ml_file = saved_state.get('current_ml_file')
        st.session_state.show_ml_upload = saved_state.get('show_ml_upload', False)
        st.session_state.file_contexts = saved_state.get('file_contexts', {})

        # Add these lines for normalization
        st.session_state.normalization_states = saved_state.get('normalization_states', {})
        st.session_state.normalization_uploaded_files = saved_state.get('normalization_uploaded_files', {})
        st.session_state.current_normalization_file = saved_state.get('current_normalization_file')
        st.session_state.show_normalization_upload = saved_state.get('show_normalization_upload', False)
        st.session_state.risk_analysis_state = saved_state.get('risk_analysis_state', {})

        return saved_state
    return None


def initialize_memory():
    saved_state = load_state()
    if saved_state:
        st.session_state.uploaded_files = saved_state['uploaded_files']
        st.session_state.current_file = saved_state['current_file']
        st.session_state.show_upload = saved_state['show_upload']
        st.session_state.last_selected_file = saved_state['last_selected_file']
        st.session_state.file_configs = saved_state.get('file_configs', {})
        st.session_state.charts_per_row = saved_state.get('charts_per_row', 2)
        st.session_state.frequency_states = saved_state.get('frequency_states', {})
        st.session_state.ml_states = saved_state.get('ml_states', {})
        st.session_state.normalization_settings = saved_state.get('normalization_settings', {})
        st.session_state.ml_uploaded_files = saved_state.get('ml_uploaded_files', {})
        st.session_state.current_ml_file = saved_state.get('current_ml_file')
        st.session_state.show_ml_upload = saved_state.get('show_ml_upload', False)
        st.session_state.normalization_states = saved_state.get('normalization_states', {})
        st.session_state.normalization_uploaded_files = saved_state.get('normalization_uploaded_files', {})
        st.session_state.current_normalization_file = saved_state.get('current_normalization_file')
        st.session_state.show_normalization_upload = saved_state.get('show_normalization_upload', False)
        st.session_state.file_contexts = saved_state.get('file_contexts', {})
        st.session_state.setdefault('risk_analysis_state', {})

    else:
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = {}
        if 'current_file' not in st.session_state:
            st.session_state.current_file = None
        if 'show_upload' not in st.session_state:
            st.session_state.show_upload = False
        if 'last_selected_file' not in st.session_state:
            st.session_state.last_selected_file = None
        if 'file_configs' not in st.session_state:
            st.session_state.file_configs = {}
        if 'charts_per_row' not in st.session_state:
            st.session_state.charts_per_row = 2
        if 'ml_states' not in st.session_state:
            st.session_state.ml_states = {}
        if 'normalization_states' not in st.session_state:
            st.session_state.normalization_states = {}
        if 'normalization_uploaded_files' not in st.session_state:
            st.session_state.normalization_uploaded_files = {}
        if 'current_normalization_file' not in st.session_state:
            st.session_state.current_normalization_file = None
        if 'show_normalization_upload' not in st.session_state:
            st.session_state.show_normalization_upload = False
        if 'frequency_states' not in st.session_state:
            st.session_state.frequency_states = {}
        if 'ml_uploaded_files' not in st.session_state:
            st.session_state.ml_uploaded_files = {}
        if 'current_ml_file' not in st.session_state:
            st.session_state.current_ml_file = None
        if 'show_ml_upload' not in st.session_state:  # Add this block
            st.session_state.show_ml_upload = False
        if 'ml_uploaded_files' not in st.session_state:
            st.session_state.ml_uploaded_files = {}
        if 'current_ml_file' not in st.session_state:
            st.session_state.current_ml_file = None
        if 'show_ml_upload' not in st.session_state:
            st.session_state.show_ml_upload = False
        if 'file_contexts' not in st.session_state:
            st.session_state.file_contexts = {}
        if 'risk_analysis_state' not in st.session_state:
            st.session_state.risk_analysis_page = {}

    if 'file_contexts' not in st.session_state:
        st.session_state.file_contexts = {}

    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False

    if 'prediction_input' not in st.session_state:
        st.session_state.prediction_input = {}

    if 'file_contexts' not in st.session_state:
        st.session_state.file_contexts = {}


def save_frequency_state(file_name, state):
    if 'frequency_states' not in st.session_state:
        st.session_state.frequency_states = {}

    # Convert DataFrame to dict before saving
    if 'feature_importance' in state:
        if isinstance(state['feature_importance'], pd.DataFrame):
            state['feature_importance'] = state['feature_importance'].to_dict('records')
        elif not isinstance(state['feature_importance'], list):
            del state['feature_importance']

    # Save additional state information
    state['user_selected_columns'] = st.session_state.get('user_selected_columns', [])
    state['filter_conditions'] = st.session_state.get('filter_conditions', {})
    state['feature_importance_target_column'] = st.session_state.get('feature_importance_target_column')
    state['accident_frequency'] = st.session_state.get('accident_frequency')
    state['selected_scalar'] = st.session_state.get('scalar_selection')  # Add this line
    state['scalar_value'] = state.get('scalar_value')  # Add this line

    st.session_state.frequency_states[file_name] = state
    save_state()


def load_frequency_state(file_name):
    if 'frequency_states' in st.session_state and file_name in st.session_state.frequency_states:
        state = st.session_state.frequency_states[file_name]

        # Convert feature_importance back to DataFrame if it exists
        if 'feature_importance' in state and isinstance(state['feature_importance'], list):
            state['feature_importance'] = pd.DataFrame(state['feature_importance'])

        # We don't need to modify session state here anymore
        return state
    return {}


def save_frequency_uploaded_files(files):
    st.session_state.frequency_uploaded_files = files
    save_state()


def load_frequency_uploaded_files():
    return st.session_state.get('frequency_uploaded_files', {})


def save_normalization_settings(file_name, settings):
    if 'normalization_settings' not in st.session_state:
        st.session_state.normalization_settings = {}
    st.session_state.normalization_settings[file_name] = settings
    save_state()


def clear_normalized_dataset(file_name):
    if 'normalized_datasets' in st.session_state and file_name in st.session_state.normalized_datasets:
        del st.session_state.normalized_datasets[file_name]
    save_state()


def save_normalization_state(file_name, state):
    if 'normalization_states' not in st.session_state:
        st.session_state.normalization_states = {}

    # We don't need to modify the state here anymore, as we're passing the complete state
    st.session_state.normalization_states[file_name] = state
    save_state()


def load_normalization_state(file_name):
    if 'normalization_states' in st.session_state and file_name in st.session_state.normalization_states:
        return st.session_state.normalization_states[file_name]
    return {}


def save_normalization_uploaded_files(files):
    st.session_state.normalization_uploaded_files = files
    save_state()


def load_normalization_uploaded_files():
    return st.session_state.get('normalization_uploaded_files', {})


def save_ml_state(file_name):
    if 'file_contexts' not in st.session_state:
        st.session_state.file_contexts = {}

    if file_name not in st.session_state.file_contexts:
        st.session_state.file_contexts[file_name] = {}

    context = st.session_state.file_contexts[file_name]

    # Save all relevant data for the file
    context['target_column'] = st.session_state.get('target_column')
    context['X'] = st.session_state.get('X')
    context['y'] = st.session_state.get('y')
    context['task'] = st.session_state.get('task')
    context['prediction_input'] = st.session_state.get('prediction_input', {})
    context['trained_model'] = st.session_state.get('trained_model')
    context['uploaded_model'] = context.get('uploaded_model')  # Use the context, not session_state
    context['uploaded_model_filename'] = context.get('uploaded_model_filename')  # Use the context
    context['model_trained'] = st.session_state.get('model_trained', False)
    context['model_evaluated'] = st.session_state.get('model_evaluated', False)
    context['selected_features'] = st.session_state.get('selected_features')
    context['classification_report'] = st.session_state.get('classification_report')
    context['accuracy'] = st.session_state.get('accuracy')
    context['mse'] = st.session_state.get('mse')
    context['selected_algorithm'] = st.session_state.get('selected_algorithm')
    context['prediction_input'] = st.session_state.file_contexts[file_name].get('prediction_input', {})
    context['model_trained'] = st.session_state.file_contexts[file_name].get('model_trained', False)

    save_state()


def load_ml_state(file_name):
    if 'file_contexts' not in st.session_state:
        st.session_state.file_contexts = {}

    if file_name not in st.session_state.file_contexts:
        st.session_state.file_contexts[file_name] = {}

    if 'prediction_input' not in st.session_state.file_contexts[file_name]:
        st.session_state.file_contexts[file_name]['prediction_input'] = {}

    context = st.session_state.file_contexts[file_name]

    # Load all relevant data for the file
    st.session_state.target_column = context.get('target_column')
    st.session_state.X = context.get('X')
    st.session_state.y = context.get('y')
    st.session_state.task = context.get('task')
    st.session_state.prediction_input = context.get('prediction_input', {})
    st.session_state.trained_model = context.get('trained_model')
    st.session_state.model_trained = context.get('model_trained', False)
    st.session_state.selected_features = context.get('selected_features')
    st.session_state.classification_report = context.get('classification_report')
    st.session_state.accuracy = context.get('accuracy')
    st.session_state.mse = context.get('mse')
    st.session_state.selected_algorithm = context.get('selected_algorithm')
    st.session_state.prediction_input = context.get('prediction_input', {})
    st.session_state.file_contexts[file_name]['model_trained'] = context.get('model_trained', False)

    return context


def load_ml_uploaded_files():
    return st.session_state.get('ml_uploaded_files', {})


def delete_file_memory(file_name):
    """Delete all memory associated with a specific file."""
    if file_name in st.session_state.uploaded_files:
        del st.session_state.uploaded_files[file_name]
    if file_name in st.session_state.file_configs:
        del st.session_state.file_configs[file_name]
    if file_name in st.session_state.normalization_settings:
        del st.session_state.normalization_settings[file_name]
    if file_name in st.session_state.normalization_states:
        del st.session_state.normalization_states[file_name]
    if 'normalization_uploaded_files' in st.session_state and file_name in st.session_state.normalization_uploaded_files:
        del st.session_state.normalization_uploaded_files[file_name]
    if file_name in st.session_state.ml_states:
        ml_state = st.session_state.ml_states[file_name]
        if 'uploaded_model' in ml_state:
            del ml_state['uploaded_model']
        if 'uploaded_model_filename' in ml_state:
            del ml_state['uploaded_model_filename']
        del st.session_state.ml_states[file_name]
        if file_name in st.session_state.frequency_states:
            del st.session_state.frequency_states[file_name]
    clear_normalized_dataset(file_name)

    # Clear the current session state for the deleted file
    if 'uploaded_model' in st.session_state:
        del st.session_state.uploaded_model
    if 'uploaded_model_filename' in st.session_state:
        del st.session_state.uploaded_model_filename

    save_state()