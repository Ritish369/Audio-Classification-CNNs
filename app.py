import numpy as np
import streamlit as st
from tensorflow.keras.models import model_from_json
from st_audiorec import st_audiorec
import io
import librosa
import scipy
from skimage.transform import resize
import tensorflow as tf
import tensorflow_io as tfio
import warnings


warnings.filterwarnings("ignore")

st.title("Audio Classification using CNNs")


@st.cache_resource
def load_model(model_path_with_model, model_name):
    # Change made of \\ due to file not found error
    with open(f"{model_path_with_model}/{model_name}.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(f"{model_path_with_model}/{model_name}.weights.h5")
    return model


models_path = "vastai/working/models"

waveform_gender = load_model(models_path, "waveform_gender_model")
waveform_digit = load_model(models_path, "waveform_num_model")
spectrogram_gender = load_model(models_path, "spectrogram_gender_model")
spectrogram_digit = load_model(models_path, "spectrogram_num_model")

st.write("Models loaded successfully.")

st.write("Kindly record your voice by clicking on the below button:")

try:
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.success("Your voice has been successfully recorded.")

        status = st.empty()
        status.write("Predictions are being made, kindly be patient.")

        audio_file_like_obj = io.BytesIO(wav_audio_data)
        audio_recording, sampling_rate = librosa.load(
            audio_file_like_obj, sr=8000, mono=True
        )

        audio = np.empty((8000,), dtype=np.float32)
        if len(audio_recording) < 8000:
            audio = np.pad(audio_recording, (0, 8000 - len(audio_recording)))
        elif len(audio_recording) > 8000:
            audio = audio_recording[:8000]

        st.write(
            "This recording will be an audio input for the models to perform the classification task of speaker's gender and the digit spoken."
            " This recognition is based on the two ways an audio signal can be represented in."
        )

        st.write("Waveform based predictions are being made and thus, are as follows:")
        predicted_gender = np.argmax(
            waveform_gender.predict(audio.reshape(1, 8000, 1)), axis=1
        )
        predicted_digit = np.argmax(
            waveform_digit.predict(audio.reshape(1, 8000, 1)), axis=1
        )

        gender = "INF"
        if predicted_gender == 0:
            gender = "Male"
        elif predicted_gender == 1:
            gender = "Female"

        st.write(f"Gender is {gender}.")
        st.write(f"Spoken digit is  {predicted_digit[0]}.")

        st.write(
            "Spectrogram based predictions are being made and thus, are as follows:"
        )

        f, t, Zxx = scipy.signal.stft(
            audio, 8000, nperseg=455, noverlap=393, window="hann"
        )

        Zxx_mag = np.abs(Zxx)
        spect_final = np.atleast_3d(Zxx_mag)

        spect_mel = tfio.audio.melscale(
            spect_final, rate=8000, mels=128, fmin=0, fmax=4000
        )

        spect_log_mel = tf.math.log(spect_mel + 1e-6)

        spect_dB_scaled = tfio.audio.dbscale(spect_log_mel, top_db=80)

        resized = resize(spect_dB_scaled.numpy(), (128, 128, 1), preserve_range=True)

        resized = np.clip(resized, -80, 0)
        resized = (resized + 80) / 80.0

        resized_spect = tf.reshape(
            tf.convert_to_tensor(resized, dtype=tf.float32), (1, 128, 128, 1)
        )

        spect_predicted_gender = np.argmax(
            spectrogram_gender.predict(resized_spect),
            axis=1,
        )
        spect_predicted_digit = np.argmax(
            spectrogram_digit.predict(resized_spect),
            axis=1,
        )

        sgender = "INF"
        if spect_predicted_gender == 0:
            sgender = "Male"
        elif spect_predicted_gender == 1:
            sgender = "Female"

        st.write(f"Gender is {sgender}.")
        st.write(f"Spoken digit is  {spect_predicted_digit[0]}.")

        status.empty()

    # else:
    #     st.error("Failed to record your audio. Kindly record again. THANKS!")

except Exception as e:
    print(f"Error occurred: {e}")
