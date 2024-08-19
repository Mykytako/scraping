import io
import pandas as pd
import plotly.express as px
import streamlit as st
from memory import save_state, initialize_memory, delete_file_memory, clear_normalized_dataset


def create_chart_wrapper(chart_type, df, x_col, y_col, z_col, title, width, height, show_std_dev):
    return create_chart(chart_type, df, x_col, y_col, z_col, title, width, height, show_std_dev)


@st.cache_data
def cached_process_data(data, x_axis, y_axis, z_axis=None):
    return filter_data(data, x_axis, y_axis, z_axis)


def update_chart_settings(i, key):
    if f"{key}_{i}" in st.session_state:
        value = st.session_state[f"{key}_{i}"]
        if st.session_state.current_file not in st.session_state.file_configs:
            st.session_state.file_configs[st.session_state.current_file] = {}
        if 'chart_settings' not in st.session_state.file_configs[st.session_state.current_file]:
            st.session_state.file_configs[st.session_state.current_file]['chart_settings'] = []
        while len(st.session_state.file_configs[st.session_state.current_file]['chart_settings']) <= i:
            st.session_state.file_configs[st.session_state.current_file]['chart_settings'].append({})
        st.session_state.file_configs[st.session_state.current_file]['chart_settings'][i][key] = value
        save_state()


def upload_file():
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        file_name = uploaded_file.name
        file_content = uploaded_file.read()
        st.session_state.uploaded_files[file_name] = file_content
        if file_name not in st.session_state.file_configs:
            st.session_state.file_configs[file_name] = {'plots': []}
        st.success(f"File '{file_name}' uploaded successfully!")
        st.session_state.current_file = file_name
        st.session_state.show_upload = False

        save_state()
        st.rerun()


def on_file_select():
    selected_file = st.session_state.file_selector
    if selected_file != st.session_state.get('current_file'):
        if selected_file == "Add new file":
            st.session_state.show_upload = True
            st.session_state.current_file = None
        else:
            if st.session_state.get('current_file'):
                clear_normalized_dataset(st.session_state.current_file)
            st.session_state.current_file = selected_file
            st.session_state.show_upload = False

            # Load the ML state for the selected file
            if selected_file in st.session_state.ml_states:
                ml_state = st.session_state.ml_states[selected_file]
                st.session_state.uploaded_model = ml_state.get('uploaded_model')
                st.session_state.uploaded_model_filename = ml_state.get('uploaded_model_filename')
            else:
                st.session_state.uploaded_model = None
                st.session_state.uploaded_model_filename = None

        st.session_state.last_selected_file = selected_file
        save_state()


def choose_file():
    files = list(st.session_state.uploaded_files.keys()) + ["Add new file"]
    index = files.index(st.session_state.current_file) if st.session_state.current_file in files else 0

    selected_file = st.sidebar.selectbox(
        "Choose File",
        files,
        index=index,
        key="file_selector",
        on_change=on_file_select
    )

    return selected_file


def delete_file():
    if st.session_state.current_file and st.session_state.current_file in st.session_state.uploaded_files:
        if st.sidebar.button("Delete Current File"):
            st.session_state.confirm_delete = True

    if st.session_state.confirm_delete:
        st.sidebar.warning("Are you sure you want to delete this file?")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Yes"):
            delete_file_memory(st.session_state.current_file)
            st.session_state.current_file = None
            st.session_state.show_upload = True
            st.session_state.confirm_delete = False
            st.session_state.last_selected_file = None
            st.sidebar.success("File deleted successfully!")
            save_state()
            st.rerun()
        if col2.button("No"):
            st.session_state.confirm_delete = False
            st.rerun()


def get_column_types(data):
    return {
        'categorical': data.select_dtypes(include=['object', 'category']).columns,
        'numerical': data.select_dtypes(include=['number']).columns
    }


def initialize_chart_settings(file_name, categorical_columns, numerical_columns):
    if file_name not in st.session_state.file_configs:
        st.session_state.file_configs[file_name] = {'chart_settings': []}

    if 'chart_settings' not in st.session_state.file_configs[file_name]:
        st.session_state.file_configs[file_name]['chart_settings'] = []

    return st.session_state.file_configs[file_name]['chart_settings']


def create_new_chart_settings(categorical_columns, numerical_columns):
    return {
        'chart_type': 'Bar Chart',
        'x_axis': categorical_columns[0] if len(categorical_columns) > 0 else numerical_columns[0],
        'y_axis': numerical_columns[0],
        'z_axis': numerical_columns[1] if len(numerical_columns) > 1 else None,
        'show_error_bars': False,
        'show_std_dev': False,
        'width': 800,
        'height': 600
    }


def update_chart_settings_count(chart_settings, num_charts, categorical_columns, numerical_columns):
    while len(chart_settings) < num_charts:
        new_settings = create_new_chart_settings(categorical_columns, numerical_columns)
        chart_settings.append(new_settings)

        for key, value in new_settings.items():
            if f"{key}_{len(chart_settings) - 1}" not in st.session_state:
                st.session_state[f"{key}_{len(chart_settings) - 1}"] = value

    return chart_settings[:num_charts]


def create_chart_settings_ui(i, settings, categorical_columns, numerical_columns):
    with st.sidebar.expander(f"Chart {i + 1} Settings"):
        settings['chart_type'] = st.selectbox(
            f"Select Chart Type for Chart {i + 1}",
            ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Pie Chart', '3D Scatter Plot', 'Heatmap'],
            index=['Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Pie Chart', '3D Scatter Plot', 'Heatmap'].index(settings['chart_type']),
            key=f"chart_type_{i}",
            on_change=update_chart_settings,
            args=(i, 'chart_type')
        )
        settings['x_axis'] = st.selectbox(
            f"Select X-Axis Column for Chart {i + 1}",
            categorical_columns.union(numerical_columns),
            index=list(categorical_columns.union(numerical_columns)).index(settings['x_axis']),
            key=f"x_axis_{i}",
            on_change=update_chart_settings,
            args=(i, 'x_axis')
        )
        settings['y_axis'] = st.selectbox(
            f"Select Y-Axis Column for Chart {i + 1}",
            numerical_columns,
            index=list(numerical_columns).index(settings['y_axis']),
            key=f"y_axis_{i}",
            on_change=update_chart_settings,
            args=(i, 'y_axis')
        )
        if settings['chart_type'] in ['3D Scatter Plot', 'Heatmap']:
            settings['z_axis'] = st.selectbox(
                f"Select Z-Axis Column for Chart {i + 1}",
                numerical_columns,
                index=list(numerical_columns).index(settings['z_axis']) if settings['z_axis'] in numerical_columns else 0,
                key=f"z_axis_{i}",
                on_change=update_chart_settings,
                args=(i, 'z_axis')
            )
        else:
            settings['z_axis'] = st.selectbox(
                f"Select Z-Axis Column for Chart {i + 1}",
                numerical_columns,
                index=list(numerical_columns).index(settings['z_axis']) if settings[
                                                                               'z_axis'] in numerical_columns else 0,
                key=f"z_axis_{i}",
                on_change=update_chart_settings,
                args=(i, 'z_axis'),
                disabled=settings['chart_type'] not in ['3D Scatter Plot', 'Heatmap']
            )
        if settings['chart_type'] in ['Bar Chart', 'Line Chart']:
            settings['show_std_dev'] = st.checkbox(
                f"Show Standard Deviation for Chart {i + 1}",
                value=settings.get('show_std_dev', False),
                key=f"show_std_dev_{i}",
                on_change=update_chart_settings,
                args=(i, 'show_std_dev')
            )
        else:
            settings['show_std_dev'] = False
        settings['width'] = st.slider(
            f"Chart Width for Chart {i + 1}",
            min_value=400,
            max_value=1200,
            value=settings['width'],
            key=f"width_{i}",
            on_change=update_chart_settings,
            args=(i, 'width')
        )
        settings['height'] = st.slider(
            f"Chart Height for Chart {i + 1}",
            min_value=400,
            max_value=1200,
            value=settings['height'],
            key=f"height_{i}",
            on_change=update_chart_settings,
            args=(i, 'height')
        )


def create_chart(chart_type, data, x_col, y_col, z_col, title, width, height, show_std_dev):
    filtered_data = cached_process_data(data, x_col, y_col, z_col)

    if chart_type == 'Bar Chart':
        return create_bar_chart(filtered_data, x_col, y_col, title, width, height, show_std_dev)
    elif chart_type == 'Line Chart':
        return create_line_chart(filtered_data, x_col, y_col, title, width, height, show_std_dev)
    elif chart_type == 'Scatter Plot':
        return px.scatter(filtered_data, x=x_col, y=y_col, labels={x_col: x_col, y_col: y_col}, title=title,
                          width=width, height=height)
    elif chart_type == 'Histogram':
        return px.histogram(filtered_data, x=x_col, labels={x_col: x_col, 'count': 'Frequency'}, title=title,
                            width=width, height=height)
    elif chart_type == 'Pie Chart':
        fig = px.pie(filtered_data, names=x_col, values=y_col, title=title, width=width, height=height, hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    elif chart_type == '3D Scatter Plot':
        return px.scatter_3d(filtered_data, x=x_col, y=y_col, z=z_col,
                             labels={x_col: x_col, y_col: y_col, z_col: z_col}, title=title, width=width, height=height)
    elif chart_type == 'Heatmap':
        return px.density_heatmap(filtered_data, x=x_col, y=y_col, z=z_col,
                                  labels={x_col: x_col, y_col: y_col, z_col: z_col}, title=title, width=width,
                                  height=height)


def create_bar_chart(df, x_col, y_col, title, width, height, show_std_dev):
    if show_std_dev:
        summary = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        fig = px.bar(summary, x=x_col, y='mean', error_y='std', labels={x_col: x_col, 'mean': y_col}, title=title, width=width, height=height)
        fig.add_traces(px.line(summary, x=x_col, y='mean', error_y='std').update_traces(mode='lines').data)
    else:
        fig = px.bar(df, x=x_col, y=y_col, labels={x_col: x_col, y_col: y_col}, title=title, width=width, height=height)
    return fig


def create_line_chart(df, x_col, y_col, title, width, height, show_std_dev):
    if show_std_dev:
        summary = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        fig = px.line(summary, x=x_col, y='mean', error_y='std', labels={x_col: x_col, 'mean': y_col}, title=title, width=width, height=height)
        fig.add_traces(px.line(summary, x=x_col, y='mean').update_traces(line=dict(color='rgba(0,0,0,0.3)', dash='dash')).data)
    else:
        fig = px.line(df, x=x_col, y=y_col, labels={x_col: x_col, y_col: y_col}, title=title, width=width, height=height)
    return fig


def filter_data(data, x_axis, y_axis, z_axis=None):
    if z_axis:
        filtered_data = data[[x_axis, y_axis, z_axis]].dropna()
    else:
        filtered_data = data[[x_axis, y_axis]].dropna()

    if isinstance(data[x_axis].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(data[x_axis]):
        top_categories = filtered_data[x_axis].value_counts().head(10).index
        filtered_data = filtered_data[filtered_data[x_axis].isin(top_categories)]

    return filtered_data


def run_dashboard(data, file_name):
    # Check if data is a DataFrame, if not, try to convert it
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except:
            st.error("Unable to process the data. Please ensure it's in the correct format.")
            return

    # Now we can safely use data.columns
    data.columns = data.columns.str.strip()

    column_types = get_column_types(data)
    chart_settings = initialize_chart_settings(file_name, column_types['categorical'], column_types['numerical'])

    num_charts = st.sidebar.number_input('Number of Charts', min_value=1, max_value=20,
                                         value=len(chart_settings) if chart_settings else 1, step=1)

    st.session_state.charts_per_row = st.sidebar.slider('Charts per Row', min_value=1, max_value=5,
                                                        value=st.session_state.charts_per_row)

    chart_settings = update_chart_settings_count(chart_settings, num_charts,
                                                 column_types['categorical'], column_types['numerical'])

    for i, settings in enumerate(chart_settings):
        create_chart_settings_ui(i, settings, column_types['categorical'], column_types['numerical'])

    chart_cols = st.columns(st.session_state.charts_per_row)

    for i, settings in enumerate(chart_settings):
        col = chart_cols[i % st.session_state.charts_per_row]
        with col:
            fig = create_chart_wrapper(
                settings['chart_type'],
                data,
                settings['x_axis'],
                settings['y_axis'],
                settings['z_axis'],  # Add this line
                f'{settings["y_axis"]} by {settings["x_axis"]} - {settings["chart_type"]}',
                settings['width'],
                settings['height'],
                settings['show_std_dev']
            )
            st.plotly_chart(fig)

        if (i + 1) % st.session_state.charts_per_row == 0:
            st.markdown("---")

    st.session_state.file_configs[file_name]['chart_settings'] = chart_settings
    save_state()


def display_dashboard():
    initialize_memory()
    if st.session_state.current_file and st.session_state.current_file in st.session_state.uploaded_files:
        file_content = st.session_state.uploaded_files[st.session_state.current_file]

        # Check if file_content is already a DataFrame
        if isinstance(file_content, pd.DataFrame):
            data = file_content
        elif isinstance(file_content, bytes):
            # If it's bytes, assume it's a CSV and read it
            data = pd.read_csv(io.BytesIO(file_content), on_bad_lines='skip', low_memory=False)
        elif isinstance(file_content, str):
            # If it's a string, assume it's a CSV and read it
            data = pd.read_csv(io.StringIO(file_content), on_bad_lines='skip', low_memory=False)
        else:
            st.error(f"Unsupported data type: {type(file_content)}")
            return

        run_dashboard(data, st.session_state.current_file)
    elif not st.session_state.show_upload:
        st.info("Please choose a file from the sidebar to view its dashboard.")

