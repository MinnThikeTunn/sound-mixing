# Realtime_PyAudio_FFT
<p align="center">
  <img src="./assets/teaser.gif">
</p>

### A simple package to do realtime audio analysis in native Python, using PyAudio and Numpy to extract and visualize FFT features from a live audio stream.

[**Demo Video**](https://youtu.be/FnP2bkzU4oo)

**The basic pipeline:**
* Starts a stream_reader that pulls live audio data from any source using PyAudio (soundcard, microphone, ...)
* Reads data from this stream many times per second (eg 1000 updates per second) and stores that data in a fifo buffer
* When triggered by `.get_audio_features()`, the stream_analyzer, applies a Fast-Fourier-Transform to the most recent audio window in the buffer
* When `visualize` is enabled, the visualizer displays these FFT features in realtime using a PyGame GUI (I made two display modes: 2D and 3D)

**Requirements:**

`pip install -r requirements.txt`

You also might have to:

`sudo apt install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0` (on Ubuntu)
or 

`brew install portaudio` (on Mac)

* Other platforms like Mac/Windows should work if PyGame can find your display and Python finds your audio card (these can be tricky with [WSL](https://research.wmz.ninja/articles/2017/11/setting-up-wsl-with-graphics-and-audio.html))

Alternatively to pyaudio, you can try [sounddevice](https://python-sounddevice.readthedocs.io/en/0.3.15/installation.html) which sometimes is more compatible with Windows/Mac
* just run `pip install sounddevice`
* The code to switch between the two sound interfaces is in the `__init__` function of the Stream_Analyzer class

**Usage:**

just run `python run_FFT_analyzer.py` and play a sound on your machine!
* I have personally learned **A LOT** about sound by watching [this realtime visualization](https://www.youtube.com/watch?v=FnP2bkzU4oo) while listening to music
* You can run the stream_analyzer in headless mode and use the FFT features in any Python Application that requires live musical features

![Teaser image](./assets/usage.png)

**ToDo:**
* Implement realtime beat detection / melody extraction on top of FFT features (eg using Harmonic/Percussive decomposition)
* The pygame.transform operations sometimes cause weird visual artifacts (boxes) for some resolution settings --> fix??
* Remove the matplotlib dependency since it's only needed for the colormap of the vis..
* Slow bars decay speed currently depends on how often `.get_audio_features()` is called --> fix
# sound-mixing
