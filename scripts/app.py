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

# Import helper and plotting functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.helpers import (
    filter_sessions_by_date_range, display_mouse_selection,
    create_session_title_card, add_observations, collect_comments,
    apply_responsive_css, create_app_header, create_date_range_form,
    create_daily_date_selector, validate_date_range, get_experiment_type_category
)
from src.utils.plot_functions import (
    plot_summary_data, plot_rdk_data, plot_basic_data
)

# TODO: Add LLM to summarize comments
from llm_summerize_local import summarize_session

PROCESSED_DATA_DIR_STR = os.getenv("PROCESSED_DATA_DIR")
if not PROCESSED_DATA_DIR_STR:
    raise ValueError("Environment variable PROCESSED_DATA_DIR must be set.")
PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR_STR)


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


if __name__ == "__main__":
    # Streamlit page configuration
    st.set_page_config(
        page_title="Mouse Training Data Analysis",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🐭"
    )

    # Apply responsive CSS and create header
    apply_responsive_css()
    create_app_header()

    # Load data
    session_info, analyzed_data = load_data()

    # Define your tabs
    mouse_wise, day_wise = st.tabs(["🐭 Mouse Overview", "📅 Daily Session Details"])

    with mouse_wise:
        st.markdown("### 📊 Data Analysis Configuration")

        # Create date range form
        start_date, end_date, submitted = create_date_range_form(session_info)

        # Validate and process date selection
        if validate_date_range(start_date, end_date):
            filtered_sessions = filter_sessions_by_date_range(start_date, end_date, session_info)

            if not filtered_sessions.empty:
                st.markdown("---")
                st.markdown("### 🐭 Mouse Selection")

                selected_mouse = display_mouse_selection(filtered_sessions)

                if selected_mouse:
                    mouse_sessions = filtered_sessions[filtered_sessions.mouse_id == selected_mouse].sort_values(by="date", ascending=False)

                    # Add some spacing before plots
                    st.markdown("---")
                    st.markdown("### 📈 Analysis Results")

                    # Plot summary if date range spans more than 1 day and mouse is selected
                    if (end_date - start_date).days > 1 and selected_mouse:
                        plot_summary_data(mouse_sessions)

                    for idx_date, date in enumerate(mouse_sessions.date.unique()):
                        sessions = mouse_sessions[mouse_sessions.date == date].reset_index()
                        experiment_types = sessions.experiment.unique()

                        # Determine experiment type category and plot accordingly
                        exp_category = get_experiment_type_category(experiment_types)

                        if exp_category == "rdk":
                            plot_rdk_data(sessions, analyzed_data, date, identifier="date")
                        elif exp_category == "basic":
                            plot_basic_data(sessions, analyzed_data, date, identifier="date")
                        else:
                            st.warning(f"⚠️ Unknown experiment type(s) for {date}: {experiment_types}")
                else:
                    st.info("🔍 Please select a mouse to view detailed analysis.")
            else:
                st.warning("📭 No data available for the selected date range.")

    with day_wise:
        st.markdown("### 📅 Daily Session Analysis")

        # Create daily date selector
        selected_date = create_daily_date_selector(session_info)

        if selected_date is not None:
            filtered_sessions = filter_sessions_by_date_range(selected_date, selected_date, session_info)

            if not filtered_sessions.empty:
                mouse_options = list(np.sort(filtered_sessions.mouse_id.unique()))

                st.markdown("---")
                st.markdown(f"### 🐭 Sessions for {selected_date.strftime('%B %d, %Y')}")

                for mouse_idx, selected_mouse in enumerate(mouse_options):
                    sessions = filtered_sessions[
                        (filtered_sessions.date == selected_date) &
                        (filtered_sessions.mouse_id == selected_mouse)
                    ].reset_index()

                    if not sessions.empty:
                        # Add a subtle separator between mice
                        if mouse_idx > 0:
                            st.markdown("---")

                        experiment_types = sessions.experiment.unique()

                        # Determine experiment type category and plot accordingly
                        exp_category = get_experiment_type_category(experiment_types)

                        if exp_category == "rdk":
                            plot_rdk_data(sessions, analyzed_data, selected_date, selected_mouse, identifier="mouse_id")
                        elif exp_category == "basic":
                            plot_basic_data(sessions, analyzed_data, selected_date, selected_mouse, identifier="mouse_id")
                        else:
                            st.warning(f"⚠️ Unknown experiment type(s) for {selected_mouse} on {selected_date}: {experiment_types}")
            else:
                st.warning("📭 No data available for the selected date.")
        else:
            st.info("📅 Please select a date to view daily session details.")
