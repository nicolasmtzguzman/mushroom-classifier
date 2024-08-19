import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def add_sidebar():
  st.sidebar.header("Mushroom Specimen Measurements")

  data = pd.read_csv("data/mushroom_cleaned.csv")

  slider_labels = [
        ("Cap Diameter", "cap-diameter"),
        ("Cap Shape", "cap-shape"),
        ("Gill Attachment", "gill-attachment"),
        ("Gill Color", "gill-color"),
        ("Stem Height", "stem-height"),
        ("Stem Width", "stem-width"),
        ("Stem Color", "stem-color"),
        ("Season", "season"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
  
  return input_dict


def get_scaled_values(input_dict):
  data = pd.read_csv("data/mushroom_cleaned.csv")

  X = data.drop(['class'], axis=1)

  scaled_dict = {}

  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value

  return scaled_dict


def get_radar_chart(input_data):
  input_data = get_scaled_values(input_data)
  
  categories = ['Cap Diameter', 'Cap Shape', 'Gill Attachment', 'Gill Color', 
                'Stem Height', 'Stem Width', 
                'Stem Color', 'Season']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['cap-diameter'], input_data['cap-shape'], input_data['gill-attachment'],
          input_data['gill-color'], input_data['stem-height'], input_data['stem-width'],
          input_data['stem-color'], input_data['season']
        ],
        theta=categories,
        fill='toself',
        name='Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  scaler = pickle.load(open("model/scaler.pkl", "rb"))

  input_array = np.array(list(input_data.values())).reshape(1,-1)
  input_array_scaled = scaler.transform(input_array)
  prediction = model.predict(input_array_scaled)

  st.subheader("Mushroom Specimen Prediction")
  st.write("The mushroom is:")

  if prediction[0] == 0:
    st.write("Edible")
  else:
    st.write("Poisonous")
  
  st.write("Probability of being Edible: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being Poisonous: ", model.predict_proba(input_array_scaled)[0][1])




def main():
  # Set main page
  st.set_page_config(
    page_title="Mushroom Classifier",
    page_icon=":mushroom:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  input_data = add_sidebar()

  with st.container():
    st.title("Mushroom Classifier")
    st.write("This app uses a machine learning model to predict whether a mushroom is poisonous or edible based on the measurements it receives. You can update these measurements using the sliders in the sidebar. Values for the variables can be seen at https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset")
  
  col1, col2 = st.columns([4,1])

  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)
  

  
  

if __name__ == '__main__':
  main()