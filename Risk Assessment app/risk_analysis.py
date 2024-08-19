import streamlit as st
from memory import save_state, initialize_memory


def save_risk_analysis_state():
    st.session_state.risk_analysis_state = {
        'crss_and_fars_frq': st.session_state.get('crss_and_fars_frq', 0.0),
        'crss_and_fars_sev': st.session_state.get('crss_and_fars_sev', 0),
        'level2_frq': st.session_state.get('level2_frq', 0.0),
        'level2_sev': st.session_state.get('level2_sev', 0),
        'dmv_frq': st.session_state.get('dmv_frq', 0.0),
        'dmv_sev': st.session_state.get('dmv_sev', 2),
        'crss_and_fars_risk': st.session_state.get('crss_and_fars_risk_result', 'Not calculated'),
        'level2_risk': st.session_state.get('level2_risk_result', 'Not calculated'),
        'dmv_risk': st.session_state.get('dmv_risk_result', 'Not calculated')
    }
    save_state()


def load_risk_analysis_state():
    return st.session_state.get('risk_analysis_state', {})

def calculate_risk_score(accident_frequency, predicted_severity, max_severity, frequency_weight=0.5):
    if accident_frequency is None or predicted_severity is None:
        return None
    normalized_severity = predicted_severity / max_severity
    risk_score = (accident_frequency * frequency_weight + normalized_severity * (1 - frequency_weight))
    return min(max(risk_score, 0), 1)


def categorize_risk(risk_score):
    if risk_score is None:
        return "Unknown"
    elif risk_score < 0.5:
        return "Low Risk"
    elif risk_score < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"


def risk_analysis_page():
    initialize_memory()

    state = load_risk_analysis_state()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('CRSS and Fars Calculation')
        frequency = st.number_input('Accident Frequency number', key='crss_and_fars_frq', format="%.5f",
                                    value=state.get('crss_and_fars_frq', 0.0), on_change=save_risk_analysis_state)
        severity_options = [0, 1, 2, 3]
        severity = st.selectbox('Accident Severity class', options=severity_options, key='crss_and_fars_sev',
                                index=severity_options.index(state.get('crss_and_fars_sev', 0)),
                                on_change=save_risk_analysis_state)

        if st.button('Predict Risk', key='crss_and_fars_risk'):
            risk_score = calculate_risk_score(frequency, severity, 3)
            risk_category = categorize_risk(risk_score)
            st.session_state.crss_and_fars_risk_result = risk_category
            st.write(f'Risk score is: {risk_category}')
            save_risk_analysis_state()

    with col2:
        st.write('Level 2 Calculation')
        frequency = st.number_input('Accident Frequency number', key='level2_frq', format="%.5f",
                                    value=state.get('level2_frq', 0.0), on_change=save_risk_analysis_state)
        severity_options = [0, 1, 2, 3, 4, 5]
        severity = st.selectbox('Accident Severity class', options=severity_options, key='level2_sev',
                                index=severity_options.index(state.get('level2_sev', 0)),
                                on_change=save_risk_analysis_state)

        if st.button('Predict Risk', key='level2_risk'):
            risk_score = calculate_risk_score(frequency, severity, 5)
            risk_category = categorize_risk(risk_score)
            st.session_state.level2_risk_result = risk_category
            st.write(f'Risk score is: {risk_category}')
            save_risk_analysis_state()

    with col3:
        st.write('DMV Calculation')
        frequency = st.number_input('Accident Frequency number', key='dmv_frq', format="%.5f",
                                    value=state.get('dmv_frq', 0.0), on_change=save_risk_analysis_state)
        severity_options = [2, 3, 4]
        default_severity = state.get('dmv_sev', 2)
        severity_index = severity_options.index(default_severity) if default_severity in severity_options else 0
        severity = st.selectbox('Accident Severity class', options=severity_options, key='dmv_sev',
                                index=severity_index,
                                on_change=save_risk_analysis_state)

        if st.button('Predict Risk', key='dmv_risk'):
            risk_score = calculate_risk_score(frequency, severity, 4)
            risk_category = categorize_risk(risk_score)
            st.session_state.dmv_risk_result = risk_category
            st.write(f'Risk score is: {risk_category}')
            save_risk_analysis_state()
