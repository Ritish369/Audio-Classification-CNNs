import numpy as np
from tensorflow.keras.models import model_from_json
import librosa
import scipy
from skimage.transform import resize
import tensorflow as tf
import tensorflow_io as tfio
import warnings
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")


def load_model(model_path_with_model, model_name):
    with open(f"{model_path_with_model}/{model_name}.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(f"{model_path_with_model}/{model_name}.weights.h5")
    return model


def audio_load(filename):
    au, sr = librosa.load(filename, sr=8000)
    audio = np.empty((8000,), dtype=np.float32)
    if len(au) < 8000:
        audio = np.pad(au, (0, 8000 - len(au)))
    elif len(au) > 8000:
        audio = au[:8000]
    print("Audio processed.")
    return audio


def spectrogram_generator(audio_in):
    f, t, zxx = scipy.signal.stft(
        audio_in, 8000, nperseg=455, noverlap=393, window="hann"
    )
    zxx_spect = np.abs(zxx)
    spectrogram = np.atleast_3d(zxx_spect)
    mel_spect = tfio.audio.melscale(spectrogram, rate=8000, mels=128, fmin=0, fmax=4000)
    log_mel_spect = tf.math.log(mel_spect + 1e-6)
    dbscale_spect = tfio.audio.dbscale(log_mel_spect, top_db=80)
    resized_spect = resize(dbscale_spect.numpy(), (128, 128, 1), preserve_range=True)
    resized = tf.convert_to_tensor(resized_spect, dtype=tf.float32)
    final_spect = tf.reshape(resized, (1, 128, 128, 1))
    print("Spectrogram generated.")
    return final_spect


def prediction(audio_r, spect, wgmodel, wdmodel, sgmodel, sdmodel):
    predicted_gender = np.argmax(wgmodel.predict(audio_r.reshape(1, 8000, 1)), axis=1)
    predicted_digit = np.argmax(wdmodel.predict(audio_r.reshape(1, 8000, 1)), axis=1)

    spect_predicted_gender = np.argmax(sgmodel.predict(spect), axis=1)
    spect_predicted_digit = np.argmax(sdmodel.predict(spect), axis=1)
    print("Predictions made.")
    return (
        predicted_gender,
        predicted_digit,
        spect_predicted_gender,
        spect_predicted_digit,
    )


if __name__ == "__main__":
    models_path = (
        "C:\\Users\\ritis\\Desktop\\Audio-Classification-CNNs\\vastai\\working\\models"
    )

    waveform_gender = load_model(models_path, "waveform_gender_model")
    waveform_digit = load_model(models_path, "waveform_num_model")
    spectrogram_gender = load_model(models_path, "spectrogram_gender_model")
    spectrogram_digit = load_model(models_path, "spectrogram_num_model")

    audio_mobile = audio_load("Zero-1.wav")

    spect_mobile = spectrogram_generator(audio_mobile)

    wgender, wdigit, sgender, sdigit = prediction(
        audio_mobile,
        spect_mobile,
        waveform_gender,
        waveform_digit,
        spectrogram_gender,
        spectrogram_digit,
    )

    print(f"Zero: {wgender}, {wdigit}, {sgender}, {sdigit}")
