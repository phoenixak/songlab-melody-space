import streamlit as st
import requests
import numpy as np
import os
import base64
from streamlit import session_state as st_state
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt


def get_api_url_and_bearer_token():
  """
  Tries to get API_URL and BEARER_TOKEN from environment variables.
  If not found, sets them to default values and throws an error.
  """
  try:
    API_URL = os.environ["API_URL"]
  except KeyError:
    st.error("API_URL environment variable is not set.")
    st.stop()

  try:
    BEARER_TOKEN = os.environ["BEARER_TOKEN"]
  except KeyError:
    st.error("BEARER_TOKEN environment variable is not set.")
    st.stop()

  return API_URL, BEARER_TOKEN


def initialize_session_state():
  """
  Initializes session state variables for audio data and user inputs.
  """
  if 'audio' not in st_state:
    st_state.audio = None
  if 'augmented_audio' not in st_state:
    st_state.augmented_audio = None
  if 'vocal_audio' not in st_state:
    st_state.vocal_audio = None


def create_headers(bearer_token):
  """
  Creates headers for API requests with Bearer token authorization.
  """
  return {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json"
  }


def upload_and_get_file_bytes():
  """
  Uploads a music file and returns its bytes if uploaded, otherwise None.
  """
  uploaded_file = st.file_uploader("Upload Music File", type=["mp3", "wav", "ogg", "flac", "aac"])
  if uploaded_file:
    return uploaded_file.read()
  else:
    return None


def generate_audio(api_url, headers, prompt, duration, audio_bytes=None):
  """
  Generates audio based on user prompt, duration and optional uploaded audio.
  """
  payload = {"inputs": {"prompt": prompt, "duration": duration}}
  if audio_bytes:
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    payload["inputs"]["track"] = audio_base64
  st.text("Generating audio...")
  response = requests.post(api_url, headers=headers, json=payload)
  generated_audio = np.array(response.json()[0]['generated_audio'], dtype=np.float32)
  sample_rate = response.json()[0]['sample_rate']
  st.audio(generated_audio, format="audio/wav", sample_rate=sample_rate, start_time=0)
  return generated_audio, sample_rate


def mix_vocals(audio, vocal_audio, sample_rate):
  """
  Mixes uploaded vocal audio with the generated audio.
  """
  vocal_audio, _ = librosa.load(vocal_audio, sr=sample_rate, mono=False)
  vocal_audio = librosa.util.fix_length(vocal_audio, len(audio))
  return (audio + vocal_audio) / 2


def apply_volume_balance(audio, balance):
  """
  Applies volume balance to the audio.
  """
  return audio * 10 ** (balance / 20)


def apply_compression(audio, threshold, ratio, knee, max_gain):
  """
  Applies simple soft-knee compression to the audio.
  """
  def compress(x):
    over = np.maximum(x - threshold, 0)
    gain = over / (over + knee) * (1 - (1 / ratio)) + 1
    gain = np.maximum(gain, 1 - max_gain)
    return x * gain
  return compress(audio)
