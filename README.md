# Audio-Classification-CNNs
An Audio Classification task with two types of inputs to the CNN models for intended work using Tensorflow.

Benchmark is in the below mentioned release:

https://github.com/Ritish369/Audio-Classification-CNNs/releases/tag/v0.0.1-alpha

**DEMO LINK**:

https://curious-audio.onrender.com (Broken because of OOM limits by Render; give it a try.)

https://huggingface.co/spaces/ritish369/curious-audio

**Datasets link**:

https://www.kaggle.com/datasets/ritisheditor/audiowaveform-dataset

https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist

Think of this as an open source project and Consider contributing to this repository to the fullest since there are still issues present in it.

**Some of these issues are:**

Perfectly timed recordings for the models to make predictions -- isn't a second long recording a tight constraint ? Can this recording time be increased as a buffer for better working ?

Is there any way to stop the recording automatically after the decided period of time ?

How to make the training and validation data closest to the real world audio scenarios using methouds like Data Augmentation ? Since the test data in this project is quite understood.

**streamlit-webrtc was tried to be studied for integration in this project. Is there any way to do so ? Could not succeed from my end.**
 (https://github.com/whitphx/streamlit-webrtc)

 Are there any problems related to Spectrogram-based models ? Specifically, to the spectrogram generation from the real-time website user  interface audio recording ?

 Because spectrogram-based models seem to be saturated as give the same output everytime. Maybe model saturation or not ?

 Therefore, it is highly encouraged to discuss and study this project for its betterment. Leave comments. It would help.
