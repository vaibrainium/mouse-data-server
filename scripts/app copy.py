import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
import os
import re
import dotenv
dotenv.load_dotenv()

# TODO: Add LLM to summarize comments
from llm_summerize_local import summarize_session

PROCESSED_DATA_DIR_STR = os.getenv("PROCESSED_DATA_DIR")
if not PROCESSED_DATA_DIR_STR:
    raise ValueError("Environment variable PROCESSED_DATA_DIR must be set.")
PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR_STR)

# Constants
TEXTWIDTH = 5.0
FONTSIZE = 5.0
COLOR = ["#d11149", "#1a8fe3", "#1ccd6a", "#e6c229", "#6610f2", "#f17105", "#65e5f3", "#bd8ad5", "#b16b57"]


# Load Data
def load_data():
    """Load session info and analyzed data with error handling."""
    try:
        session_info = pd.read_csv(PROCESSED_DATA_DIR / "session_info.csv")
        session_info["date"] = pd.to_datetime(session_info["date"]).dt.date
        with open(PROCESSED_DATA_DIR / "analyzed_data.pkl", "rb") as f:
            analyzed_data = pickle.load(f)
        return session_info, analyzed_data
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def create_subplots():
    """Create the main subplot grid for accuracy and valid trials."""

    # Create the subplots first
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Accuracy vs Date",
            "Total Valid Trial vs Start Weight",
            "Sensory Noise vs Date",
            "Total Valid Trial vs Date",
        ],
        shared_xaxes=False,
        vertical_spacing=0.2,
    )
    return fig

def plot_accuracy_vs_date(fig, data, row, col):
    """Plot accuracy over time with improved error handling."""
    if data.empty:
        return

    threshold = 80
    right_block_color = "rgba(255, 0, 0, 0.5)"  # Red with transparency
    left_block_color = "rgba(0, 0, 255, 0.5)"  # Blue with transparency

    # Left accuracy trace
    left_data = data.dropna(subset=["left_accuracy"])
    if not left_data.empty:
        fig.add_trace(
            go.Scatter(
                x=left_data["date"],
                y=left_data["left_accuracy"].astype(int),
                mode="lines+markers",
                marker=dict(size=12, color=left_block_color),
                line=dict(color=left_block_color),
                hovertemplate="<b>Date</b>: %{x}<br><b>Left Accuracy</b>: %{y}%<extra></extra>",
                showlegend=False,
                name="Left Accuracy"
            ),
            row=row,
            col=col,
        )

    # Right accuracy trace
    right_data = data.dropna(subset=["right_accuracy"])
    if not right_data.empty:
        fig.add_trace(
            go.Scatter(
                x=right_data["date"],
                y=right_data["right_accuracy"].astype(int),
                mode="lines+markers",
                marker=dict(size=12, color=right_block_color),
                line=dict(color=right_block_color),
                hovertemplate="<b>Date</b>: %{x}<br><b>Right Accuracy</b>: %{y}%<extra></extra>",
                showlegend=False,
                name="Right Accuracy"
            ),
            row=row,
            col=col,
        )

    # Add threshold line if we have date data
    if not data.empty:
        fig.add_trace(
            go.Scatter(
                x=[min(data["date"]), max(data["date"])],
                y=[threshold, threshold],
                mode="lines",
                line=dict(color="black", dash="dash"),
                name=f"Threshold {threshold}%",
                showlegend=False,
            ),
            row=row, col=col,
        )

    # Update axes only once
    fig.update_xaxes(title_text="Date", tickformat="%Y-%m-%d", row=row, col=col)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 105],
                     zeroline=True, zerolinecolor="black",
                     zerolinewidth=2, mirror=True, row=row, col=col)

def plot_accuracy_vs_start_weight(fig, data, row, col):
    """Plot accuracy vs start weight with improved error handling."""
    clean_data = data.dropna(subset=["session_accuracy", "start_weight"])
    if clean_data.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=clean_data["start_weight"].astype(int),
            y=clean_data["session_accuracy"].astype(int),
            mode="markers",
            marker=dict(size=12),
            line=dict(color=COLOR[5]),
            hovertemplate="<b>Start Weight</b>: %{x}%<br><b>Accuracy</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title="% Start Weight", range=[80, 105], row=row, col=col)
    fig.update_yaxes(title="Accuracy (%)", range=[0, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def plot_total_valid_vs_date(fig, data, row, col):
    """Plot total valid trials over time with improved error handling."""
    clean_data = data.dropna(subset=["total_valid"])
    if clean_data.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=clean_data["date"],
            y=clean_data["total_valid"].astype(int),
            mode="lines+markers",
            marker=dict(size=12),
            line=dict(color=COLOR[5]),
            hovertemplate="<b>Date</b>: %{x}<br><b>Valid Trials</b>: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title="Date", tickformat="%Y-%m-%d", row=row, col=col)
    fig.update_yaxes(title="Total Valid Trials", zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def plot_sensory_noise_vs_date(fig, data, row, col):
    """Plot sensory noise over time with improved error handling."""
    clean_data = data.dropna(subset=["sensory_noise"])
    if clean_data.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=clean_data["date"],
            y=clean_data["sensory_noise"],
            mode="lines+markers",
            marker=dict(size=12),
            line=dict(color=COLOR[5]),
            hovertemplate="<b>Date</b>: %{x}%<br><b>Sensory Noise</b>: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title="Date", tickformat="%Y-%m-%d", row=row, col=col)
    fig.update_yaxes(title="Sensory Noise", zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def add_correction_blocks(fig, starts, ends, color, name, row, col):
    for idx in range(len(starts)):
        fig.add_trace(
            go.Scatter(
                x=[starts[idx], starts[idx], ends[idx], ends[idx]],
                y=[-1, 1, 1, -1],
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line=dict(width=0),
                name=name,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

def plot_summary_data(data):
    """Plot summary data."""
    data = data.sort_values(by="date", ascending=True)
    fig = create_subplots()
    plot_accuracy_vs_date(fig, data, row=1, col=1)
    plot_sensory_noise_vs_date(fig, data, row=2, col=1)
    plot_accuracy_vs_start_weight(fig, data, row=1, col=2)
    plot_total_valid_vs_date(fig, data, row=2, col=2)
    # Update the layout of the entire figure

    st.markdown(f"<h3 style='text-align: center; margin-top: 40px;'>Summary Data</h3>", unsafe_allow_html=True)
    fig.update_layout(
        title="",
        title_x=0.5,
        title_y=0.98,
        title_font=dict(size=16, family="Arial"),
        title_pad=dict(t=2),
        showlegend=True,
        height=900,
        width=600,
        annotations=[
            dict(text="Accuracy vs Date", x=0.22, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
            dict(text="Accuracy vs Start Weight", x=0.8, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
            dict(text="Sensory Noise vs Date", x=0.22, y=0.45, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
            dict(text="Valid Trials vs Date", x=0.8, y=0.45, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

def filter_sessions_by_date_range(start_date, end_date, session_info):
    """Filter session data based on date range."""
    return session_info[(session_info.date >= start_date) & (session_info.date <= end_date)]

def display_mouse_selection(filtered_sessions):
    """Display mouse selection dropdown."""
    mouse_options = [None] + list(np.sort(filtered_sessions.mouse_id.unique()))
    return st.selectbox("Select Mouse", options=mouse_options)

def plot_rolling_accuracy_vs_trial(fig, data, session_idx, row, col):
    fig.add_trace(
        go.Scatter(
            x=data["binned_trials"],
            y=data["binned_accuracies"],
            mode="lines",
            marker=dict(size=12),
            name=f"Session {session_idx+1}",
            line=dict(color=COLOR[session_idx]),
            hovertemplate="<b>Trial Number</b>: %{x}<br><b>Accuracy</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title="Trial Number", range=[0, len(data['binned_trials'])], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)
    fig.update_yaxes(title="Rolling Accuracy (%)", range=[-10, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def plot_accuracy_vs_coherence(fig, data, session_idx, row, col):
    fig.add_trace(
        go.Bar(
            x=data["coherences"],
            y=data["accuracy"] * 100,
            name=f"Session {session_idx+1}",
            marker=dict(color=COLOR[session_idx]),
            hovertemplate="<b>Coherence</b>: %{x}<br><b>Accuracy</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title="% Coherence", row=row, col=col)
    fig.update_yaxes(title="Accuracy (%)", range=[0, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def plot_all_trials_choices(fig, data, session_idx, row, col):
    fig.add_trace(
        go.Scatter(
            x=data["all_data_idx"],
            y=data["all_data_choice"] * 0.80,
            mode="markers",
            marker=dict(color="black", symbol="line-ns-open", size=8, line=dict(width=1.5)),
            name="Choices",
        ),
        row=row, col=col,
    )

def plot_all_trials_rolling_bias_and_threshold(fig, data, session_idx, row, col, plot_thresholds=True):
    fig.add_trace(
        go.Scatter(
            x=data["all_data_idx"],
            y=data["all_data_rolling_bias"],
            mode="lines",
            marker=dict(size=12),
            name=f"Session {session_idx+1}",
            line=dict(color=COLOR[session_idx]),
            hovertemplate="<b>Trial Number</b>: %{x}<br><b>Bias</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row, col=col,
    )
    if plot_thresholds:
        for threshold in [0.25, -0.25]:
            fig.add_trace(
                go.Scatter(
                    x=[0, max(data["all_data_idx"])],
                    y=[threshold, threshold],
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    name=f"Threshold {threshold}",
                ),
                    row=row, col=col,
            )

def plot_all_trials_active_block_bands(fig, data, session_idx, row, col):
    if (data["right_active_block_starts"]) or (data["left_active_block_starts"]):
        right_block_color = "rgba(255, 0, 0, 0.5)"  # Red with transparency
        left_block_color = "rgba(0, 0, 255, 0.5)"   # Blue with transparency

        add_correction_blocks(fig, data["right_active_block_starts"], data["right_active_block_ends"], right_block_color, "Right Active Block", row=1, col=3)
        add_correction_blocks(fig, data["left_active_block_starts"], data["left_active_block_ends"], left_block_color, "Left Active Block", row=1, col=3)

        # Legend traces (dummy scatter points for better legend formatting)
        for color, name in zip([right_block_color, left_block_color], ["Right Active Block", "Left Active Block"]):
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=color), name=name),
                row=row, col=col,
            )

def plot_all_trials_rolling_performance(fig, data, session_idx, row, col):
    plot_all_trials_choices(fig, data, session_idx, row, col)
    plot_all_trials_rolling_bias_and_threshold(fig, data, session_idx, row, col)
    plot_all_trials_active_block_bands(fig, data, session_idx, row, col)
    fig.update_xaxes(title="Trial Number", range=[0, len(data['all_data_idx'])], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)
    fig.update_yaxes(title="Rolling Bias", range=[-1.05, 1.05], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def plot_psych(fig, data, session_idx, row, col):
    fig.add_trace(
        go.Scatter(
            x=data["valid_psych_data"]["x_data"],
            y=data["valid_psych_data"]["y_data"],
            mode="markers",
            marker=dict(size=12),
            name=f"Session {session_idx+1}",
            line=dict(color=COLOR[session_idx]),
            hovertemplate="<b>Trial Number</b>: %{x}<br><b>Bias</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=data["valid_psych_data"]["x_hat"],
            y=data["valid_psych_data"]["y_hat"],
            mode="lines",
            marker=dict(size=12),
            name=f"Session {session_idx+1}",
            line=dict(color=COLOR[session_idx]),
            hovertemplate="<b>Trial Number</b>: %{x}<br><b>Bias</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row, col=col,
    )

    fig.update_yaxes(title="Rolling Bias", range=[-0.05, 1.05], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def format_comments(comments):
    """Format comments with HTML line breaks and bold section headers."""
    # remove lines that only contain numbers
    comments = re.sub(r'^\d+\s*$', '', comments, flags=re.MULTILINE)
    comments = comments.replace("\n", "<br>")
    # Bold text before a colon on any line (start of string or after <br>)
    comments = re.sub(r'(?:(?<=^)|(?<=<br>))([^:<]+?):', r'<b style="color: #444;">\1:</b>', comments)
    return comments

def add_observations(comment, unique_key):
    """Display formatted comments inside a styled Streamlit expander."""
    formatted = format_comments(comment)
    with st.expander("Show/Hide Notes"):
        st.markdown(f"""
        <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 50px;'>
            <h5 style='color: #333;'>Notes:</h5>
            <p style='font-size: 16px; color: #777;'>{formatted}</p>
        </div>
        """, unsafe_allow_html=True)

def build_title(sessions, identifier, date, mouse_id, session_data=None):
    """Build HTML-formatted session title with metadata."""
    title_parts = []
    if identifier == "date":
        title_parts.append(f"Date: {date}")
    elif identifier == "mouse_id":
        title_parts.append(f"Mouse: {mouse_id}")
    else:
        title_parts.append(f"Mouse: {mouse_id} &nbsp;&nbsp; Date: {date}")

    title_html = f"{' <br> '.join(title_parts)} <br>"

    for idx, metadata in sessions.iterrows():
        color = COLOR[idx % len(COLOR)]
        title_html += (
            f"<span style='color: {color}; font-size: 25px;'>"
            f"Session {idx + 1} - {metadata.rig_id.replace('_', ' ').title()}, Phase: {metadata.experiment.replace('_', ' ').title()}, "
        )
        # if metadata contains 'configuration_used' and it's not none:
        if 'configuration_used' in metadata and pd.notna(metadata.configuration_used):
            configuration = metadata.configuration_used.replace("_", "-").lower()
            title_html += f"Config: {configuration}, "

        # Format start_time to HH:MM AM/PM
        if pd.notna(metadata.start_time):
            if isinstance(metadata.start_time, str):
                # Try to parse string to datetime
                try:
                    start_time_dt = datetime.strptime(metadata.start_time, "%H:%M:%S")
                    formatted_time = start_time_dt.strftime("%I:%M %p")
                except ValueError:
                    try:
                        start_time_dt = datetime.strptime(metadata.start_time, "%H:%M")
                        formatted_time = start_time_dt.strftime("%I:%M %p")
                    except ValueError:
                        formatted_time = metadata.start_time  # fallback to original
            else:
                # Assume it's already a datetime object
                formatted_time = metadata.start_time.strftime("%I:%M %p")
        else:
            formatted_time = "N/A"

        if metadata.experiment in ["rt_directional_training", "rt_maintenance", "rt_test"]:
            # Format psychometric model coefficients for display
            # psych_coefs = session_data.get('valid_psych_model_coefs', session_data.get('all_psych_model_coefs', None))
            psych_coefs = session_data['valid_psych_data']["coefs_"]
            if psych_coefs:
                psych_display = f"Alpha: {psych_coefs['mean']:.2f}, Beta: {psych_coefs['var']:.2f}, Lapse: {psych_coefs['lapse_rate']:.2f}"
            else:
                psych_display = "No valid fits"

            title_html += f"Start Weight: {int(metadata.start_weight)}%,  Start Time: {formatted_time}<br>"
            title_html += "&nbsp;"*25
            title_html += f"Psych Fits: {psych_display}</span><br>"
        else:
            title_html += f"Start Weight: {int(metadata.start_weight)}%,  Start Time: {formatted_time}</span><br>"
    return title_html

def collect_comments(sessions):
    """Aggregate comments from all sessions with formatting."""
    return "<br>".join(
        f"Session {idx + 1}:<br>{metadata.comments}"
        for idx, metadata in sessions.iterrows()
    )

def plot_basic_data(sessions, analyzed_data, date, mouse_id=None, identifier=None):
    """Plot rolling bias and decisions for all trials."""
    fig = sp.make_subplots(
        rows=1, cols=1,
        subplot_titles=["All trials rolling performance"],
        shared_xaxes=True, vertical_spacing=0.15,
    )

    session_data = None
    x_len = 0
    for idx, metadata in sessions.iterrows():
        if metadata.total_valid < 10:
            continue

        session_data = analyzed_data.get(metadata.session_uuid)
        if session_data is None:
            continue

        x_len = max(x_len, len(session_data['all_data_idx']))
        plot_all_trials_rolling_bias_and_threshold(fig, session_data, session_idx=idx, row=1, col=1, plot_thresholds=False)
        plot_all_trials_choices(fig, session_data, session_idx=idx, row=1, col=1)

    if session_data is not None:
        fig.update_xaxes(title="Trial Number", range=[0, x_len], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True)
        fig.update_yaxes(title="Rolling Bias", range=[-1.05, 1.05], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True)

    st.markdown(f"<h3 style='text-align: left; margin-top: 30px; margin-bottom: -70px;'>{build_title(sessions, identifier, date, mouse_id, session_data)}</h3>", unsafe_allow_html=True)

    fig.update_layout(
        height=600, width=500, showlegend=True,
        annotations=[dict(text="Rolling Bias vs Trial Number", x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=18, color="black"))]
    )

    with st.columns([1, 2, 1])[1]:
        st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))

    add_observations(collect_comments(sessions), unique_key=f"comments_{uuid.uuid4()}")

def plot_rdk_data(sessions, analyzed_data, date, mouse_id=None, identifier=None):
    """Plot RDK session metrics including accuracy and performance."""
    fig = sp.make_subplots(
        rows=1, cols=3, column_widths=[0.25, 0.25, 0.5],
        subplot_titles=["Rolling Accuracy", "Accuracy vs Coherence", "All trials rolling performance"],
        shared_xaxes=True, vertical_spacing=0.15,
    )

    for idx, metadata in sessions.iterrows():
        if metadata.total_valid < 10:
            continue

        session_data = analyzed_data.get(metadata.session_uuid)
        if session_data is None:
            continue

        plot_rolling_accuracy_vs_trial(fig, session_data, session_idx=idx, row=1, col=1)
        plot_accuracy_vs_coherence(fig, session_data, session_idx=idx, row=1, col=2)
        plot_all_trials_rolling_performance(fig, session_data, session_idx=idx, row=1, col=3)

    st.markdown(f"<h3 style='text-align: left; margin-top: 30px; margin-bottom: -70px;'>{build_title(sessions, identifier, date, mouse_id, session_data)}</h3>", unsafe_allow_html=True)

    fig.update_layout(
        height=600, width=900, showlegend=True,
        annotations=[
            dict(text="Binned Session Accuracy", x=0.125, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black")),
            dict(text="Accuracy vs Coherence", x=0.375, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black")),
            dict(text="Rolling Bias vs Trial Number", x=0.8, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black")),
        ]
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
    add_observations(collect_comments(sessions), unique_key=f"comments_{uuid.uuid4()}")

if __name__ == "__main__":

    # Streamlit page configuration
    st.set_page_config(page_title="Mouse Training Data", layout="wide")
    st.markdown("<h1 style='text-align: center;'>Mouse Training Data</h1>", unsafe_allow_html=True)

    session_info, analyzed_data = load_data()

    # Inject custom CSS to increase tab font size
    st.markdown("""
        <style>
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
            font-size: 24px;
        }
        </style>
    """, unsafe_allow_html=True)
    # Define your tabs
    mouse_wise, day_wise = st.tabs(["🐭 Mouse Overview", "📅 Daily Session Details"])

    with mouse_wise:
          # Date Range Selection
        start_date = st.date_input("Start Date", value=None, min_value=session_info.date.min(), max_value=session_info.date.max())
        end_date = st.date_input("End Date", value=None, min_value=start_date, max_value=session_info.date.max())

          # Ensure valid date selection
        if start_date and end_date:
            if start_date > end_date:
                st.error("End date must be after start date!")
            else:
                filtered_sessions = filter_sessions_by_date_range(start_date, end_date, session_info)

                if not filtered_sessions.empty:
                    selected_mouse = display_mouse_selection(filtered_sessions)
                    mouse_sessions = filtered_sessions[filtered_sessions.mouse_id == selected_mouse].sort_values(by="date", ascending=False)
                    summary_text = None

                    # Plot summary if date range spans more than 1 day and mouse is selected
                    if (end_date - start_date).days > 1 and selected_mouse:
                        plot_summary_data(mouse_sessions)

                        # # add checkbox to show/hide comments
                        # if st.checkbox("Create Summary Analysis (EXPERIMENTAL)", value=False):
                        # 	if summary_text is None:
                        # 		# Summarize comments using LLM
                        # 		mouse_session_data = [analyzed_data[metadata.session_uuid] for metadata in mouse_sessions.itertuples()]
                        # 		summary_text = summarize_session(mouse_sessions, mouse_session_data)
                        # 	if summary_text:
                        # 		st.markdown(f"<h4>Summary of Comments</h4>", unsafe_allow_html=True)
                        # 		st.markdown(f"<div style='border:1px solid #ccc; padding:10px; border-radius:5px; background:#f4f4f4;'>{summary_text}</div>", unsafe_allow_html=True)
                        # 		st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

                    for idx_date, date in enumerate(mouse_sessions.date.unique()):
                        sessions = mouse_sessions[mouse_sessions.date == date].reset_index()
                        experiment_types = sessions.experiment.unique()

                        # Check if any RDK experiments are present
                        rdk_experiments = ["rt_directional_training", "rt_maintenance", "rt_test", "rt_dynamic_training"]
                        basic_experiments = ['reward_spout_stimulus_association', "reward_spout_association", "free_reward_training"]

                        if any(exp in rdk_experiments for exp in experiment_types):
                            plot_rdk_data(sessions, analyzed_data, date, identifier="date")
                        elif any(exp in basic_experiments for exp in experiment_types):
                            plot_basic_data(sessions, analyzed_data, date, identifier="date")
                        else:
                            st.warning(f"Unknown experiment type(s) for {date}: {experiment_types}")
                else:
                    st.warning("No data available for the selected date range.")


    with day_wise:
        # Show date input
        selected_date = st.date_input(
            "Select Date",
            value=st.session_state.get("selected_date", None),
            min_value=session_info.date.min(),
            max_value=session_info.date.max()
        )

        if selected_date is not None:
            filtered_sessions = filter_sessions_by_date_range(selected_date, selected_date, session_info)
            mouse_options = list(np.sort(filtered_sessions.mouse_id.unique()))
            if not filtered_sessions.empty:
                for mouse_idx, selected_mouse in enumerate(mouse_options):
                    sessions = filtered_sessions[(filtered_sessions.date == selected_date) & (filtered_sessions.mouse_id == selected_mouse)].reset_index()

                    if not sessions.empty:
                        experiment_types = sessions.experiment.unique()

                        # Check if any RDK experiments are present
                        rdk_experiments = ["rt_directional_training", "rt_maintenance", "rt_test", "rt_dynamic_training"]
                        basic_experiments = ["reward_spout_stimulus_association", "reward_spout_association", "free_reward_training"]

                        if any(exp in rdk_experiments for exp in experiment_types):
                            plot_rdk_data(sessions, analyzed_data, selected_date, selected_mouse, identifier="mouse_id")
                        elif any(exp in basic_experiments for exp in experiment_types):
                            plot_basic_data(sessions, analyzed_data, selected_date, selected_mouse, identifier="mouse_id")
                        else:
                            st.warning(f"Unknown experiment type(s) for {selected_mouse} on {selected_date}: {experiment_types}")
            else:
                st.warning("No data available for the selected date.")
