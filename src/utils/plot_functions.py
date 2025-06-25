"""
Plotting functions for the mouse training data analysis app.
Contains all visualization functions for different types of plots.
"""

import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

from .helpers import (
    COLOR, add_correction_blocks, create_responsive_plot_layout,
    create_session_title_card, add_observations, collect_comments
)


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
            marker=dict(size=12, color=COLOR[5]),
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
            marker=dict(size=12, color=COLOR[5]),
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
            marker=dict(size=12, color=COLOR[5]),
            hovertemplate="<b>Date</b>: %{x}%<br><b>Sensory Noise</b>: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title="Date", tickformat="%Y-%m-%d", row=row, col=col)
    fig.update_yaxes(title="Sensory Noise", zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)


def plot_rolling_accuracy_vs_trial(fig, data, session_idx, row, col):
    """Plot rolling accuracy vs trial number."""
    fig.add_trace(
        go.Scatter(
            x=data["binned_trials"],
            y=data["binned_accuracies"],
            mode="lines",
            marker=dict(size=12),
            name=f"Session {session_idx+1}",
            line=dict(color=COLOR[session_idx % len(COLOR)]),
            hovertemplate="<b>Trial Number</b>: %{x}<br><b>Accuracy</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title="Trial Number", range=[0, len(data['binned_trials'])], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)
    fig.update_yaxes(title="Rolling Accuracy (%)", range=[-10, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)


def plot_accuracy_vs_coherence(fig, data, session_idx, row, col):
    """Plot accuracy vs coherence as bar chart."""
    fig.add_trace(
        go.Bar(
            x=data["coherences"],
            y=data["accuracy"] * 100,
            name=f"Session {session_idx+1}",
            marker=dict(color=COLOR[session_idx % len(COLOR)]),
            hovertemplate="<b>Coherence</b>: %{x}<br><b>Accuracy</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title="% Coherence", row=row, col=col)
    fig.update_yaxes(title="Accuracy (%)", range=[0, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)


def plot_all_trials_choices(fig, data, session_idx, row, col):
    """Plot individual trial choices."""
    fig.add_trace(
        go.Scatter(
            x=data["all_data_idx"],
            y=data["all_data_choice"] * 0.80,
            mode="markers",
            marker=dict(color="black", symbol="line-ns-open", size=8, line=dict(width=1.5)),
            name="Choices",
            showlegend=False,
        ),
        row=row, col=col,
    )


def plot_all_trials_rolling_bias_and_threshold(fig, data, session_idx, row, col, plot_thresholds=True):
    """Plot rolling bias with optional threshold lines."""
    fig.add_trace(
        go.Scatter(
            x=data["all_data_idx"],
            y=data["all_data_rolling_bias"],
            mode="lines",
            name=f"Session {session_idx+1}",
            line=dict(color=COLOR[session_idx % len(COLOR)]),
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
                    showlegend=False,
                ),
                row=row, col=col,
            )


def plot_all_trials_active_block_bands(fig, data, session_idx, row, col):
    """Plot active block bands if available."""
    if (data["right_active_block_starts"]) or (data["left_active_block_starts"]):
        right_block_color = "rgba(255, 0, 0, 0.5)"  # Red with transparency
        left_block_color = "rgba(0, 0, 255, 0.5)"   # Blue with transparency

        add_correction_blocks(fig, data["right_active_block_starts"], data["right_active_block_ends"], right_block_color, "Right Active Block", row, col)
        add_correction_blocks(fig, data["left_active_block_starts"], data["left_active_block_ends"], left_block_color, "Left Active Block", row, col)

        # Legend traces (dummy scatter points for better legend formatting)
        for color, name in zip([right_block_color, left_block_color], ["Right Active Block", "Left Active Block"]):
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=color), name=name, showlegend=False),
                row=row, col=col,
            )


def plot_all_trials_rolling_performance(fig, data, session_idx, row, col):
    """Plot complete rolling performance including choices, bias, and active blocks."""
    plot_all_trials_choices(fig, data, session_idx, row, col)
    plot_all_trials_rolling_bias_and_threshold(fig, data, session_idx, row, col)
    plot_all_trials_active_block_bands(fig, data, session_idx, row, col)
    fig.update_xaxes(title="Trial Number", range=[0, len(data['all_data_idx'])], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)
    fig.update_yaxes(title="Rolling Bias", range=[-1.05, 1.05], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)


def plot_psych(fig, data, session_idx, row, col):
    """Plot psychometric function data and fit."""
    # Data points
    fig.add_trace(
        go.Scatter(
            x=data["valid_psych_data"]["x_data"],
            y=data["valid_psych_data"]["y_data"],
            mode="markers",
            marker=dict(size=12, color=COLOR[session_idx % len(COLOR)]),
            name=f"Session {session_idx+1}",
            hovertemplate="<b>Coherence</b>: %{x}<br><b>% Choice</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row, col=col,
    )

    # Fitted line
    fig.add_trace(
        go.Scatter(
            x=data["valid_psych_data"]["x_hat"],
            y=data["valid_psych_data"]["y_hat"],
            mode="lines",
            line=dict(color=COLOR[session_idx % len(COLOR)]),
            name=f"Session {session_idx+1}",
            hovertemplate="<b>Coherence</b>: %{x}<br><b>% Choice</b>: %{y}%<extra></extra>",
            showlegend=False,
        ),
        row=row, col=col,
    )

    fig.update_yaxes(title="Proportion of Right Choices", range=[-0.05, 1.05], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)
    fig.update_xaxes(title="% Coherence", range=[-110, 110], zeroline=False, mirror=True, row=row, col=col)


def plot_summary_data(data):
    """Plot summary data with responsive layout."""
    data = data.sort_values(by="date", ascending=True)

    # Create responsive subplots
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Accuracy vs Date",
            "Accuracy vs Start Weight",
            "Sensory Noise vs Date",
            "Valid Trials vs Date",
        ],
        shared_xaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    plot_accuracy_vs_date(fig, data, row=1, col=1)
    plot_accuracy_vs_start_weight(fig, data, row=1, col=2)
    plot_sensory_noise_vs_date(fig, data, row=2, col=1)
    plot_total_valid_vs_date(fig, data, row=2, col=2)

    st.markdown("### 📊 Summary Dashboard")

    layout_config = create_responsive_plot_layout(
        height=600,
        annotations=[
            dict(text="Accuracy vs Date", x=0.22, y=1.06, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="black", family="Arial")),
            dict(text="Accuracy vs Start Weight", x=0.78, y=1.06, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="black", family="Arial")),
            dict(text="Sensory Noise vs Date", x=0.22, y=0.45, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="black", family="Arial")),
            dict(text="Valid Trials vs Date", x=0.78, y=0.45, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="black", family="Arial")),
        ]
    )

    fig.update_layout(**layout_config)
    st.plotly_chart(fig, use_container_width=True)


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

    # Create session title card
    create_session_title_card(sessions, identifier, date, mouse_id, session_data)

    # Configure plot layout
    layout_config = create_responsive_plot_layout(
        height=600,
        annotations=[dict(
            text="Rolling Bias vs Trial Number",
            x=0.5, y=1.08,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="black", family="Arial"),
            xanchor='center'
        )]
    )

    fig.update_layout(**layout_config)
    st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
    add_observations(collect_comments(sessions), unique_key=f"comments_{uuid.uuid4()}")


def plot_rdk_data(sessions, analyzed_data, date, mouse_id=None, identifier=None):
    """Plot RDK session metrics including accuracy and performance."""
    # Use responsive layout that adapts to screen size
    fig = sp.make_subplots(
        rows=1, cols=4,
        column_widths=[0.25, 0.25, 0.3, 0.2],
        subplot_titles=["Rolling Accuracy", "Accuracy vs Coherence", "Rolling Performance", "Psychometric"],
        shared_xaxes=False,
        horizontal_spacing=0.08,
    )

    session_data = None
    for idx, metadata in sessions.iterrows():
        if metadata.total_valid < 10:
            continue

        session_data = analyzed_data.get(metadata.session_uuid)
        if session_data is None:
            continue

        plot_rolling_accuracy_vs_trial(fig, session_data, session_idx=idx, row=1, col=1)
        plot_accuracy_vs_coherence(fig, session_data, session_idx=idx, row=1, col=2)
        plot_all_trials_rolling_performance(fig, session_data, session_idx=idx, row=1, col=3)
        plot_psych(fig, session_data, session_idx=idx, row=1, col=4)

    # Create session title card
    create_session_title_card(sessions, identifier, date, mouse_id, session_data)

    # Configure plot layout
    layout_config = create_responsive_plot_layout(
        height=600,
        annotations=[
            dict(text="Session Accuracy", x=0.125, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="black", family="Arial")),
            dict(text="Accuracy vs Coherence", x=0.375, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="black", family="Arial")),
            dict(text="Rolling Bias", x=0.65, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="black", family="Arial")),
            dict(text="Psychometric", x=0.9, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="black", family="Arial")),
        ]
    )

    fig.update_layout(**layout_config)
    st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
    add_observations(collect_comments(sessions), unique_key=f"comments_{uuid.uuid4()}")
