"""
Helper functions for the mouse training data analysis app.
Contains utility functions for data processing, visualization, and UI components.
"""

import re
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp


# Constants
COLOR = ["#d11149", "#1a8fe3", "#1ccd6a", "#e6c229", "#6610f2", "#f17105", "#65e5f3", "#bd8ad5", "#b16b57"]


def clamp(min_val, val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(val, max_val))


def filter_sessions_by_date_range(start_date, end_date, session_info):
    """Filter session data based on date range."""
    return session_info[(session_info.date >= start_date) & (session_info.date <= end_date)]


def format_comments(comments):
    """Format comments with HTML line breaks and bold section headers."""
    # Remove lines that only contain numbers
    comments = re.sub(r'^\d+\s*$', '', comments, flags=re.MULTILINE)
    comments = comments.replace("\n", "<br>")
    # Bold text before a colon on any line (start of string or after <br>)
    comments = re.sub(r'(?:(?<=^)|(?<=<br>))([^:<]+?):', r'<b style="color: #444;">\1:</b>', comments)
    return comments


def collect_comments(sessions):
    """Aggregate comments from all sessions with formatting."""
    return "<br>".join(
        f"Session {idx + 1}:<br>{metadata.comments}"
        for idx, metadata in sessions.iterrows()
    )


def add_observations(comment, unique_key):
    """Display formatted comments inside a styled Streamlit expander."""
    if not comment or comment.strip() == "":
        return

    formatted = format_comments(comment)
    with st.expander("📝 Session Notes & Observations", expanded=False):
        st.markdown(f"""
        <div style='
            border: 2px solid #e1e5e9;
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        '>
            <div style='
                color: #2c3e50;
                font-size: 16px;
                line-height: 1.6;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            '>
                {formatted}
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_mouse_selection(filtered_sessions):
    """Display mouse selection dropdown with enhanced styling."""
    mouse_options = [None] + list(np.sort(filtered_sessions.mouse_id.unique()))

    # Create columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_mouse = st.selectbox(
            "🐭 Select Mouse",
            options=mouse_options,
            format_func=lambda x: "Choose a mouse..." if x is None else f"Mouse {x}",
            help="Select a mouse to view detailed analysis"
        )

    with col2:
        if selected_mouse:
            mouse_session_count = len(filtered_sessions[filtered_sessions.mouse_id == selected_mouse])
            st.metric("Sessions", mouse_session_count)

    return selected_mouse


def format_start_time(start_time):
    """Format start_time to HH:MM AM/PM format."""
    if pd.notna(start_time):
        if isinstance(start_time, str):
            # Try to parse string to datetime
            try:
                start_time_dt = datetime.strptime(start_time, "%H:%M:%S")
                return start_time_dt.strftime("%I:%M %p")
            except ValueError:
                try:
                    start_time_dt = datetime.strptime(start_time, "%H:%M")
                    return start_time_dt.strftime("%I:%M %p")
                except ValueError:
                    return start_time  # fallback to original
        else:
            # Assume it's already a datetime object
            return start_time.strftime("%I:%M %p")
    else:
        return "N/A"


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
            f"Session {idx + 1} - {metadata.rig_id.replace('_', ' ').title()}, "
            f"Phase: {metadata.experiment.replace('_', ' ').title()}, "
        )

        # Add configuration if available
        if 'configuration_used' in metadata and pd.notna(metadata.configuration_used):
            configuration = metadata.configuration_used.replace("_", "-").lower()
            title_html += f"Config: {configuration}, "

        formatted_time = format_start_time(metadata.start_time)

        if metadata.experiment in ["rt_directional_training", "rt_maintenance", "rt_test"]:
            # Format psychometric model coefficients for display
            psych_display = "No valid fits"
            if session_data and 'valid_psych_data' in session_data:
                psych_coefs = session_data['valid_psych_data'].get("coefs_")
                if psych_coefs:
                    psych_display = f"Alpha: {psych_coefs['mean']:.2f}, Beta: {psych_coefs['var']:.2f}, Lapse: {psych_coefs['lapse_rate']:.2f}"

            title_html += f"Start Weight: {int(metadata.start_weight)}%,  Start Time: {formatted_time}<br>"
            title_html += "&nbsp;"*25
            title_html += f"Psych Fits: {psych_display}</span><br>"
        else:
            title_html += f"Start Weight: {int(metadata.start_weight)}%,  Start Time: {formatted_time}</span><br>"

    # Remove last <br> if it exists
    if title_html.endswith("<br>"):
        title_html = title_html[:-4]
    return title_html


def create_session_title_card(sessions, identifier, date, mouse_id, session_data=None):
    """Create a styled card with session title information."""
    title_content = build_title(sessions, identifier, date, mouse_id, session_data)

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0 20px 0;
        border-left: 5px solid #4a90e2;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    '>
        <div class='responsive-title' style='
            color: #2c3e50;
            margin: 0;
            font-weight: 600;
            line-height: 1.4;
        '>{title_content}</div>
    </div>
    """, unsafe_allow_html=True)


def add_correction_blocks(fig, starts, ends, color, name, row, col):
    """Add colored blocks to indicate correction periods."""
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


def create_responsive_plot_layout(height=600, title="", annotations=None):
    """Create a standard responsive plot layout."""
    layout_config = dict(
        height=height,
        showlegend=True,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=100, b=50),
    )

    if title:
        layout_config['title'] = title

    if annotations:
        layout_config['annotations'] = annotations

    return layout_config


def apply_responsive_css():
    """Apply responsive CSS styles to the Streamlit app."""
    st.markdown("""
        <style>
        /* Base responsive styles */
        @media (max-width: 768px) {
            .main > div {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
                padding-top: 0.5rem !important;
            }

            /* Make forms more compact on mobile */
            .stForm {
                padding: 1rem !important;
                margin-bottom: 1rem !important;
            }

            /* Reduce title font size on mobile */
            h1 {
                font-size: 2rem !important;
            }

            h3 {
                font-size: 1.2rem !important;
            }

            /* Make tabs more touch-friendly */
            button[data-baseweb="tab"] {
                min-height: 48px !important;
            }

            /* Adjust metrics for mobile */
            div[data-testid="metric-container"] {
                min-height: auto !important;
            }
        }

        /* Tab styling */
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
            font-size: clamp(18px, 4vw, 24px);
            font-weight: 600;
        }

        /* Responsive main container styling */
        .main > div {
            padding-left: clamp(0.5rem, 5vw, 2rem);
            padding-right: clamp(0.5rem, 5vw, 2rem);
            padding-top: 1rem;
        }

        /* Input controls styling */
        .stSelectbox > div > div {
            background-color: #f8f9fa;
            border-radius: 8px;
        }

        .stDateInput > div > div {
            background-color: #f8f9fa;
            border-radius: 8px;
        }

        /* Section spacing */
        div[data-testid="stVerticalBlock"] > div {
            gap: clamp(1rem, 3vw, 1.5rem);
        }

        /* Responsive form styling */
        .stForm {
            background-color: #ffffff;
            padding: clamp(1rem, 3vw, 1.5rem);
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: clamp(1rem, 3vw, 2rem);
        }

        /* Warning and error message styling */
        .stAlert {
            border-radius: 8px;
            margin: 1rem 0;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
        }

        /* Responsive plotly charts */
        .js-plotly-plot {
            width: 100% !important;
        }

        /* Mobile-friendly title styling */
        .responsive-title {
            font-size: clamp(1rem, 4vw, 1.4rem) !important;
            line-height: 1.3 !important;
        }

        /* Improve button touch targets on mobile */
        .stButton > button {
            min-height: 44px;
            padding: 0.5rem 1rem;
        }

        /* Make expanders more touch-friendly */
        .streamlit-expanderHeader {
            min-height: 48px;
            display: flex;
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)


def create_app_header():
    """Create the main application header with responsive styling."""
    st.markdown("""
    <div style='
        text-align: center;
        padding: clamp(1rem, 4vw, 2rem) clamp(0.5rem, 2vw, 1rem);
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        border-radius: 0 0 20px 20px;
    '>
        <h1 style='
            margin: 0;
            font-size: clamp(1.8rem, 6vw, 3rem);
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            line-height: 1.2;
        '>
            🐭 Mouse Training Data Analysis
        </h1>
        <p style='
            margin: 0.5rem 0 0 0;
            font-size: clamp(1rem, 3vw, 1.2rem);
            opacity: 0.9;
            line-height: 1.4;
        '>
            Comprehensive behavioral analysis and performance tracking
        </p>
    </div>
    """, unsafe_allow_html=True)


def create_date_range_form(session_info):
    """Create a responsive date range selection form."""
    with st.form("date_range_form", clear_on_submit=False):
        # Use responsive columns that stack on mobile
        col1, col2 = st.columns([1, 1])

        with col1:
            start_date = st.date_input(
                "📅 Start Date",
                value=None,
                min_value=session_info.date.min(),
                max_value=session_info.date.max(),
                help="Select the start date for analysis"
            )

        with col2:
            end_date = st.date_input(
                "📅 End Date",
                value=None,
                min_value=start_date if start_date else session_info.date.min(),
                max_value=session_info.date.max(),
                help="Select the end date for analysis"
            )

        # Submit button for form
        submitted = st.form_submit_button("🔍 Apply Date Range", type="primary", use_container_width=True)

    return start_date, end_date, submitted


def create_daily_date_selector(session_info):
    """Create a date selector for daily analysis with metrics."""
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_date = st.date_input(
            "Select Date for Analysis",
            value=st.session_state.get("selected_date", None),
            min_value=session_info.date.min(),
            max_value=session_info.date.max(),
            help="Choose a specific date to view all mouse sessions"
        )

    with col2:
        if selected_date:
            session_count = len(filter_sessions_by_date_range(selected_date, selected_date, session_info))
            st.metric("Available Sessions", session_count)

    return selected_date


def validate_date_range(start_date, end_date):
    """Validate the selected date range and show appropriate messages."""
    if not start_date or not end_date:
        st.info("📅 Please select both start and end dates to begin analysis.")
        return False

    if start_date > end_date:
        st.error("⚠️ End date must be after start date!")
        return False

    return True


def get_experiment_type_category(experiment_types):
    """Categorize experiment types into RDK or basic experiments."""
    rdk_experiments = ["rt_directional_training", "rt_maintenance", "rt_test", "rt_dynamic_training"]
    basic_experiments = ["reward_spout_stimulus_association", "reward_spout_association", "free_reward_training"]

    if any(exp in rdk_experiments for exp in experiment_types):
        return "rdk"
    elif any(exp in basic_experiments for exp in experiment_types):
        return "basic"
    else:
        return "unknown"
