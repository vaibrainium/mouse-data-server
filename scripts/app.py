import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

# def login():
#     # Create a simple login form
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")

#     if st.button("Login"):
#         if username == "" and password == "password123":
#             st.success("Logged in successfully")
#             return True
#         else:
#             st.error("Invalid username or password")
#             return False
#     return False

# if login():
#     st.write("Welcome to the Streamlit app!")



# Constants
DATA_DIR = Path("~/mouse-data-server/data").expanduser()
TEXTWIDTH = 5.0
FONTSIZE = 5.0
COLOR = ["#d11149", "#1a8fe3", "#1ccd6a", "#e6c229", "#6610f2", "#f17105", "#65e5f3", "#bd8ad5", "#b16b57"]

# Streamlit page configuration
st.set_page_config(page_title="Mouse Training Data", layout="wide")
st.markdown("<h1 style='text-align: center;'>Mouse Training Data</h1>", unsafe_allow_html=True)


# Load Data
def load_data():
    """Load session info and analyzed data."""
    session_info = pd.read_csv(DATA_DIR / "session_info.csv")
    session_info["date"] = pd.to_datetime(session_info["date"]).dt.date
    with open(DATA_DIR / "analyzed_data.pkl", "rb") as f:
        analyzed_data = pickle.load(f)
    return session_info, analyzed_data


session_info, analyzed_data = load_data()


def create_subplots():
    """Create the main subplot grid for accuracy and valid trials."""
    return sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Accuracy vs Date",
            "Accuracy vs Start Weight",
            "Total Valid Trial vs Date",
            "Total Valid Trial vs Start Weight",
        ],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )


def plot_accuracy_vs_date(fig, data):
    """Plot accuracy over time."""
    data = data.sort_values(by="date", ascending=True)
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["session_accuracy"].astype(int),
            mode="lines+markers",
            marker=dict(size=12),
            line=dict(color=COLOR[5]),
            hovertemplate="<b>Date</b>: %{x}<br><b>Accuracy</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


def plot_accuracy_vs_start_weight(fig, data):
    """Plot accuracy vs start weight."""
    data = data.sort_values(by="date", ascending=True)
    fig.add_trace(
        go.Scatter(
            x=data["start_weight"].astype(int),
            y=data["session_accuracy"].astype(int),
            mode="markers",
            marker=dict(size=12),
            line=dict(color=COLOR[5]),
            hovertemplate="<b>Start Weight</b>: %{x}%<br><b>Accuracy</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )


def plot_total_valid_vs_date(fig, data):
    """Plot total valid trials over time."""
    data = data.sort_values(by="date", ascending=True)
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["total_valid"].astype(int),
            mode="lines+markers",
            marker=dict(size=12),
            line=dict(color=COLOR[5]),
            hovertemplate="<b>Date</b>: %{x}<br><b>Valid Trials</b>: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )


def plot_total_valid_vs_start_weight(fig, data):
    """Plot total valid trials vs start weight."""
    data = data.sort_values(by="date", ascending=True)
    fig.add_trace(
        go.Scatter(
            x=data["start_weight"].astype(int),
            y=data["total_valid"].astype(int),
            mode="markers",
            marker=dict(size=12),
            line=dict(color=COLOR[5]),
            hovertemplate="<b>Start Weight</b>: %{x}%<br><b>Valid Trials</b>: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )


def plot_summary_data(data):
    """Plot summary data."""
    fig = create_subplots()
    plot_accuracy_vs_date(fig, data)
    plot_accuracy_vs_start_weight(fig, data)
    plot_total_valid_vs_date(fig, data)
    plot_total_valid_vs_start_weight(fig, data)

    fig.update_layout(
        title="Summary Data",
        title_x=0.5,
        title_y=0.98,
        title_font=dict(size=16, family="Arial"),
        title_pad=dict(t=2),
        showlegend=True,
        height=900,
        width=600,
        yaxis=dict(title="Accuracy (%)", range=[0, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
        yaxis2=dict(title="Accuracy (%)", range=[0, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
        yaxis3=dict(title="Total Valid Trials", zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
        yaxis4=dict(title="Total Valid Trials", zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
        xaxis=dict(title="Date"),
        xaxis2=dict(title="% Baseline Weight"),
        xaxis3=dict(title="Date", tickformat="%Y-%m-%d"),
        xaxis4=dict(title="% Baseline Weight"),
    )
    st.plotly_chart(fig)


def filter_sessions_by_date_range(start_date, end_date, session_info):
    """Filter session data based on date range."""
    return session_info[(session_info.date >= start_date) & (session_info.date <= end_date)]


def display_mouse_selection(filtered_sessions):
    """Display mouse selection dropdown."""
    mouse_options = [None] + list(np.sort(filtered_sessions.mouse_id.unique()))
    return st.selectbox("Select Mouse", options=mouse_options)


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

            # Plot summary if date range spans more than 1 day and mouse is selected
            if (end_date - start_date).days > 1 and selected_mouse:
                plot_summary_data(mouse_sessions)
                st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

            for idx_date, date in enumerate(mouse_sessions.date.unique()):
                sessions = mouse_sessions[mouse_sessions.date == date].reset_index()

                # Skip sessions with low valid trials
                if (sessions["total_valid"] < 20).all():
                    continue

                # Create subplot for individual session analysis
                fig = sp.make_subplots(
                    rows=1,
                    cols=4,
                    subplot_titles=[
                        "Rolling Accuracy",
                        "Accuracy vs Coherence",
                        "Rolling Bias vs Trial Number",
                        "Reaction Time vs Coherence",
                    ],
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                )

                title = f"Date: {date} <br>"
                start_weights, experiments = [], []

                # Loop through sessions and add traces for each
                for idx, metadata in sessions.iterrows():
                    if metadata.total_valid < 10:
                        continue
                    session_data = analyzed_data[metadata["index"]]
                    start_weights.append(int(metadata.start_weight))
                    experiments.append(metadata.experiment.replace("_", " ").title())

                    title += f"Session {idx+1}: {metadata.experiment.replace('_', ' ').title()}, Start Weight: {int(metadata.start_weight)}%<br>"

                    # Add subplots for individual session data
                    fig.add_trace(
                        go.Scatter(
                            x=session_data["binned_trials"],
                            y=session_data["binned_accuracies"],
                            mode="lines",
                            marker=dict(size=12),
                            name=f"Session {idx+1}",
                            line=dict(color=COLOR[idx]),
                            hovertemplate="<b>Trial Number</b>: %{x}<br><b>Accuracy</b>: %{y}%<extra></extra>",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Bar(
                            x=session_data["coherences"],
                            y=session_data["accuracy"] * 100,
                            name=f"Session {idx+1}",
                            marker=dict(color=COLOR[idx]),
                            hovertemplate="<b>Coherence</b>: %{x}<br><b>Accuracy</b>: %{y}%<extra></extra>",
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=session_data["rolling_trials"],
                            y=session_data["rolling_bias"],
                            mode="lines",
                            marker=dict(size=12),
                            name=f"Session {idx+1}",
                            line=dict(color=COLOR[idx]),
                            hovertemplate="<b>Trial Number</b>: %{x}<br><b>Bias</b>: %{y}%<extra></extra>",
                            showlegend=False,
                        ),
                        row=1,
                        col=3,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[0, max(session_data["rolling_trials"])],
                            y=[0.25, 0.25],
                            mode="lines",
                            line=dict(color="black", dash="dash"),
                            name="Threshold 0.25",
                        ),
                        row=1,
                        col=3,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[0, max(session_data["rolling_trials"])],
                            y=[-0.25, -0.25],
                            mode="lines",
                            line=dict(color="black", dash="dash"),
                            name="Threshold -0.25",
                        ),
                        row=1,
                        col=3,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=session_data["coherences"],
                            y=session_data["reaction_time_mean"],
                            mode="markers",
                            marker=dict(size=12),
                            error_y=dict(
                                type="data",
                                array=session_data["reaction_time_sd"],
                                visible=True,
                            ),
                            name=f"Session {idx+1}",
                            line=dict(color=COLOR[idx]),
                            hovertemplate="<b>Coherence</b>: %{x}<br><b>Median Reaction Time</b>: %{y}s<extra></extra>",
                        ),
                        row=1,
                        col=4,
                    )

                fig.update_layout(
                    title=title,
                    title_x=0,
                    title_y=0.98,
                    title_font=dict(size=16, family="Arial"),
                    title_pad=dict(t=2),
                    showlegend=True,
                    height=600,
                    width=900,

                    yaxis=dict(title="Rolling Accuracy (%)", range=[-10, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
                    yaxis2=dict(title="Accuracy (%)", range=[0, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
                    yaxis3=dict(title="Rolling Bias", range=[-1.05, 1.05], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
                    yaxis4=dict(title="Reaction Time (s)", range=[0, 5], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),

                    xaxis=dict(title="Trial Number", range=[0, len(session_data['binned_trials'])], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
                    xaxis2=dict(title="% Coherence"),
                    xaxis3=dict(title="Trial Number", range=[0, len(session_data['rolling_trials'])], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True),
                    xaxis4=dict(title="% Coherence"),

                    annotations=[
                        dict(text="Binned Session Accuracy", x=0.125, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
                        dict(text="Accuracy vs Coherence", x=0.375, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
                        dict(text="Rolling Bias vs Trial Number", x=0.625, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"), ),
                        dict(text="Reaction Time vs Coherence", x=0.875, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
                    ]
                )

                st.plotly_chart(fig)

        else:
            st.warning("No data available for the selected date range.")
