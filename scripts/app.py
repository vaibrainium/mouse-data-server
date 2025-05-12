import pickle
from pathlib import Path
import numpy as np
import pandas as pd
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

PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR"))

# Constants
TEXTWIDTH = 5.0
FONTSIZE = 5.0
COLOR = ["#d11149", "#1a8fe3", "#1ccd6a", "#e6c229", "#6610f2", "#f17105", "#65e5f3", "#bd8ad5", "#b16b57"]


# Load Data
def load_data():
	"""Load session info and analyzed data."""
	session_info = pd.read_csv(PROCESSED_DATA_DIR / "session_info.csv")
	session_info["date"] = pd.to_datetime(session_info["date"]).dt.date
	with open(PROCESSED_DATA_DIR / "analyzed_data.pkl", "rb") as f:
		analyzed_data = pickle.load(f)
	return session_info, analyzed_data

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
	"""Plot accuracy over time."""
	threshold = 80
	right_block_color = "rgba(255, 0, 0, 0.5)"  # Red with transparency
	left_block_color = "rgba(0, 0, 255, 0.5)"  # Blue with transparency

	# Left accuracy trace
	fig.add_trace(
		go.Scatter(
			x=data["date"],
			y = data.dropna(subset=["left_accuracy"])["left_accuracy"].astype(int),
			mode="lines+markers",
			marker=dict(size=12, color=left_block_color),
			line=dict(color=left_block_color),
			hovertemplate="<b>Date</b>: %{x}<br><b>Left Accuracy</b>: %{y}%<extra></extra>",
			showlegend=False,
		),
		row=row,
		col=col,
	)

	# Right accuracy trace
	fig.add_trace(
		go.Scatter(
			x=data["date"],
			y = data.dropna(subset=["right_accuracy"])["right_accuracy"].astype(int),
			mode="lines+markers",
			marker=dict(size=12, color=right_block_color),
			line=dict(color=right_block_color),
			hovertemplate="<b>Date</b>: %{x}<br><b>Right Accuracy</b>: %{y}%<extra></extra>",
			showlegend=False,
		),
		row=row,
		col=col,
	)

	fig.add_trace(
		go.Scatter(
			x=[min(data["date"]), max(data["date"])],
			y=[threshold, threshold],
			mode="lines",
			line=dict(color="black", dash="dash"),
			name=f"Threshold {threshold}",
		),
			row=row, col=col,
	)

	# Update axes only once
	fig.update_xaxes(title_text="Date", tickformat="%Y-%m-%d", row=row, col=col)
	fig.update_yaxes(title_text="Accuracy (%)", range=[0, 105],
					 zeroline=True, zerolinecolor="black",
					 zerolinewidth=2, mirror=True, row=row, col=col)

def plot_accuracy_vs_start_weight(fig, data, row, col):
	"""Plot accuracy vs start weight."""
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
		row=row,
		col=col,
	)
	fig.update_xaxes(title="% Start Weight", range=[80, 105], row=row, col=col)
	fig.update_yaxes(title="Accuracy (%)", range=[0, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def plot_total_valid_vs_date(fig, data, row, col):
	"""Plot total valid trials over time."""
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
		row=row,
		col=col,
	)
	fig.update_xaxes(title="Date", tickformat="%Y-%m-%d", row=row, col=col)
	fig.update_yaxes(title="Total Valid Trials", zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def plot_sensory_noise_vs_date(fig, data, row, col):
	"""Plot total valid trials vs start weight."""
	fig.add_trace(
		go.Scatter(
			x=data["date"],
			y=data["sensory_noise"],
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
	fig.update_xaxes(title="Trial Number", range=[0, len(session_data['binned_trials'])], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)
	fig.update_yaxes(title="Rolling Accuracy (%)", range=[-10, 105], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def plot_accuracy_vs_coherence(fig, data, session_idx, row, col):
	fig.add_trace(
		go.Bar(
			x=data["coherences"],
			y=data["accuracy"] * 100,
			name=f"Session {idx+1}",
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

def plot_all_trials_rolling_bias_and_threshold(fig, data, session_idx, row, col):
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
	fig.update_xaxes(title="Trial Number", range=[0, len(session_data['all_data_idx'])], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)
	fig.update_yaxes(title="Rolling Bias", range=[-1.05, 1.05], zeroline=True, zerolinecolor="black", zerolinewidth=2, mirror=True, row=row, col=col)

def add_observations(comment, unique_key):
	# Replace newlines with <br> for HTML formatting
	comment = comment.replace("\n", "<br>")
	# Highlight words between newline and ':'
	comment = re.sub(r'(<br>)(.*?):', r'\1<b style="color: #444;">\2:</b>', comment)

	# Use st.expander for toggling visibility
	with st.expander("Show/Hide Notes"):
		st.markdown(f"""
		<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 50px;'>
			<h5 style='color: #333;'>Notes:</h5>
			<p style='font-size: 16px; color: #777;'>{comment}</p>
		</div>
		""", unsafe_allow_html=True)


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

                        # add checkbox to show/hide comments
						if st.checkbox("Create Summary Analysis (EXPERIMENTAL)", value=False):
							if summary_text is None:
								# Summarize comments using LLM
								mouse_session_data = [analyzed_data[metadata.session_uuid] for metadata in mouse_sessions.itertuples()]
								summary_text = summarize_session(mouse_sessions, mouse_session_data)
							if summary_text:
								st.markdown(f"<h4>Summary of Comments</h4>", unsafe_allow_html=True)
								st.markdown(f"<div style='border:1px solid #ccc; padding:10px; border-radius:5px; background:#f4f4f4;'>{summary_text}</div>", unsafe_allow_html=True)
								st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

					for idx_date, date in enumerate(mouse_sessions.date.unique()):
						sessions = mouse_sessions[mouse_sessions.date == date].reset_index()

						# Skip sessions with low valid trials
						if (sessions["total_valid"] < 20).all():
							continue

						# Create subplot for individual session analysis
						fig = sp.make_subplots(
							rows=1,
							cols=3,
							column_widths=[0.25, 0.25, 0.5],
							subplot_titles=[
								"Rolling Accuracy",
								"Accuracy vs Coherence",
								"All trials rolling performance"
							],
							shared_xaxes=True,
							vertical_spacing=0.15,
						)

						title = f"Date: {date} <br>"
						start_weights, experiments = [], []
						comments = ""

						# Loop through sessions and add traces for each
						for idx, metadata in sessions.iterrows():
							if metadata.total_valid < 10:
								continue
							session_data = analyzed_data[metadata.session_uuid]
							start_weights.append(int(metadata.start_weight))
							experiments.append(metadata.experiment.replace("_", " ").title())

							color = COLOR[idx % len(COLOR)]  # Cycle colors if needed
							title += (
								f"<span style='color: {color}; font-size: 25px;'>"
								f"Session {idx+1}: {metadata.experiment.replace('_', ' ').title()}, "
								f"Start Weight: {int(metadata.start_weight)}%</span><br>"
							)

							plot_rolling_accuracy_vs_trial(fig, session_data, session_idx=idx, row=1, col=1)
							plot_accuracy_vs_coherence(fig, session_data, session_idx=idx, row=1, col=2)
							plot_all_trials_rolling_performance(fig, session_data, session_idx=idx, row=1, col=3)

							# if this is last row of the session, add comments
							if idx == len(sessions) - 1:
								comments += f"Session {idx+1}: \n {metadata.comments}"
							else:
								comments += f"Session {idx+1}: \n {metadata.comments} <br>"

						st.markdown(f"<h3 style='text-align: left; margin-top: 30px; margin-bottom: -70px;'>{title}</h3>", unsafe_allow_html=True)
						fig.update_layout(
							title="",
							title_x=0,
							title_y=0.98,
							title_font=dict(size=16, family="Arial"),
							title_pad=dict(t=0),
							showlegend=True,
							height=600,
							width=900,
							annotations=[
								dict(text="Binned Session Accuracy", x=0.125, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
								dict(text="Accuracy vs Coherence", x=0.375, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
								dict(text="Rolling Bias vs Trial Number", x=0.8, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"), ),
							]
						)
						st.plotly_chart(fig, use_container_width=True, key=f"session_plot_{idx_date}")
						add_observations(comments, unique_key=f"comments_{idx_date}")  # Add observations for the session




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
					# Skip sessions with low valid trials
					if (sessions["total_valid"] < 20).all():
						continue

					# Create subplot for individual session analysis
					fig = sp.make_subplots(
						rows=1,
						cols=3,
						column_widths=[0.25, 0.25, 0.5],
						subplot_titles=[
							"Rolling Accuracy",
							"Accuracy vs Coherence",
							"All trials rolling performance"
						],
						shared_xaxes=True,
						vertical_spacing=0.15,
					)

					title = f"Subject ID: {selected_mouse} <br>"
					start_weights, experiments = [], []
					comments = ""

					# Loop through sessions and add traces for each
					for idx, metadata in sessions.iterrows():
						if metadata.total_valid < 10:
							continue
						session_data = analyzed_data[metadata.session_uuid]
						start_weights.append(int(metadata.start_weight))
						experiments.append(metadata.experiment.replace("_", " ").title())
						start_time = datetime.strptime(metadata.time, "%H:%M:%S").time().strftime('%I:%M %p')
						color = COLOR[idx % len(COLOR)]  # Cycle colors if needed
						title += (
							f"<span style='color: {color}; font-size: 25px;'>"
							f"Session {idx+1}: {metadata.experiment.replace('_', ' ').title()}, "
							f"Start Weight: {int(metadata.start_weight)}%, "
							f"Start Time: {start_time} </span><br>"
						)

						plot_rolling_accuracy_vs_trial(fig, session_data, session_idx=idx, row=1, col=1)
						plot_accuracy_vs_coherence(fig, session_data, session_idx=idx, row=1, col=2)
						plot_all_trials_rolling_performance(fig, session_data, session_idx=idx, row=1, col=3)

						# if this is last row of the session, add comments
						if idx == len(sessions) - 1:
							comments += f"Session {idx+1}: \n {metadata.comments}"
						else:
							comments += f"Session {idx+1}: \n {metadata.comments} <br>"

					st.markdown(f"<h3 style='text-align: left; margin-top: 30px; margin-bottom: -70px;'>{title}</h3>", unsafe_allow_html=True)
					fig.update_layout(
						title="",
						title_x=0,
						title_y=0.98,
						title_font=dict(size=16, family="Arial"),
						title_pad=dict(t=0),
						showlegend=True,
						height=600,
						width=900,
						annotations=[
							dict(text="Binned Session Accuracy", x=0.125, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
							dict(text="Accuracy vs Coherence", x=0.375, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"),),
							dict(text="Rolling Bias vs Trial Number", x=0.8, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="black"), ),
						]
					)
					st.plotly_chart(fig, use_container_width=True, key=f"date_plot_{mouse_idx}")
					add_observations(comments, unique_key=f"comments_{mouse_idx}")  # Add observations for the session
