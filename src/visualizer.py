import numpy as np
import time, sys, math
import pygame
from collections import deque
from src.utils import Button
from matplotlib import cm
from pydub import AudioSegment
import librosa
import sounddevice as sd
import wave
import scipy.signal as signal



CHUNK = 1024 * 2
RATE = 44100
CHANNELS = 1
AMPLITUDE_LIMIT = 4096
BUTTON_COLOR = (50, 50, 150)
HOVER_COLOR = (70, 70, 200)
TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (20, 20, 30)

class Spectrum_Visualizer:
    """
    The Spectrum_Visualizer visualizes spectral FFT data using a simple PyGame GUI
    """
    def __init__(self, ear):
        self.plot_audio_history = True
        self.ear = ear

        self.HEIGHT  = self.ear.height
        window_ratio = self.ear.window_ratio

        self.HEIGHT = round(self.HEIGHT)
        self.WIDTH  = round(window_ratio*self.HEIGHT)
        self.y_ext = [round(0.05*self.HEIGHT), self.HEIGHT]
        self.cm = cm.plasma
        #self.cm = cm.inferno

        self.toggle_history_mode()

        self.add_slow_bars = 1
        self.add_fast_bars = 1
        self.slow_bar_thickness = max(0.00002*self.HEIGHT, 1.25 / self.ear.n_frequency_bins)
        self.tag_every_n_bins = max(1,round(5 * (self.ear.n_frequency_bins / 51))) # Occasionally display Hz tags on the x-axis

        self.fast_bar_colors = [list((255*np.array(self.cm(i))[:3]).astype(int)) for i in np.linspace(0,255,self.ear.n_frequency_bins).astype(int)]
        self.slow_bar_colors = [list(np.clip((255*3.5*np.array(self.cm(i))[:3]).astype(int) , 0, 255)) for i in np.linspace(0,255,self.ear.n_frequency_bins).astype(int)]
        self.fast_bar_colors = self.fast_bar_colors[::-1]
        self.slow_bar_colors = self.slow_bar_colors[::-1]

        self.slow_features = [0]*self.ear.n_frequency_bins
        self.frequency_bin_max_energies  = np.zeros(self.ear.n_frequency_bins)
        self.frequency_bin_energies = self.ear.frequency_bin_energies
        self.bin_text_tags, self.bin_rectangles = [], []

        #Fixed init params:
        self.start_time = None
        self.vis_steps  = 0
        self.fps_interval = 10
        self.fps = 0
        self._is_running = False
        self.recording = False
        

    def record_audio(self, filename="recorded_audio.wav", duration=30):

        # self.recording = False
        #             sd.wait()  # Ensure recording is completed
        #             self.audio_data = self.extract_white_noise(self.audio_data.flatten())  # Apply noise reduction
        #             with wave.open("recorded_audio.wav", 'wb') as wavefile:
        #                 wavefile.setnchannels(CHANNELS)
        #                 wavefile.setsampwidth(2)
        #                 wavefile.setframerate(RATE)
        #                 wavefile.writeframes(self.audio_data.tobytes())
        #             

        audio_data = sd.rec(int(RATE * duration), samplerate=RATE, channels=CHANNELS, dtype=np.int16)

        sd.wait()  # Wait for recording to complete
        audio_data = audio_data.flatten()
        with wave.open("recorded_audio.wav", 'wb') as wavefile:
            wavefile.setnchannels(CHANNELS)
            wavefile.setsampwidth(2)
            wavefile.setframerate(RATE)
            wavefile.writeframes(audio_data.tobytes())
        print("Recording saved as recorded_audio.wav with noise reduction")

    def slow_down(self, sound, factor):
        return sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * factor)}).set_frame_rate(sound.frame_rate)
    def extract_white_noise(self, audio_data, fs=44100):
        """Applies a high-pass filter to remove white noise."""
        if len(audio_data) <= 15:
            print("Warning: Audio data is too short for noise reduction. Skipping filter.")
        return audio_data  # Return original data instead of raising an error.

        # Design a high-pass Butterworth filter with a normalized cutoff frequency
        b, a = signal.butter(2, 0.1 / (fs / 2), 'high')
        return signal.filtfilt(b, a, audio_data, padlen=len(audio_data) - 1)

    def toggle_history_mode(self):

        if self.plot_audio_history:
            self.bg_color           = 10    #Background color
            self.decay_speed        = 0.10  #Vertical decay of slow bars
            self.inter_bar_distance = 0
            self.avg_energy_height  = 0.1125
            self.alpha_multiplier   = 0.995
            self.move_fraction      = 0.0099
            self.shrink_f           = 0.994

        else:
            self.bg_color           = 60
            self.decay_speed        = 0.06
            self.inter_bar_distance = int(0.2*self.WIDTH / self.ear.n_frequency_bins)
            self.avg_energy_height  = 0.225

        self.bar_width = (self.WIDTH / self.ear.n_frequency_bins) - self.inter_bar_distance

        #Configure the bars:
        self.slow_bars, self.fast_bars, self.bar_x_positions = [],[],[]
        for i in range(self.ear.n_frequency_bins):
            x = int(i* self.WIDTH / self.ear.n_frequency_bins)
            fast_bar = [int(x), int(self.y_ext[0]), math.ceil(self.bar_width), None]
            slow_bar = [int(x), None, math.ceil(self.bar_width), None]
            self.bar_x_positions.append(x)
            self.fast_bars.append(fast_bar)
            self.slow_bars.append(slow_bar)
      
  
    def start(self):
        print("Starting spectrum visualizer...")
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.screen.fill((self.bg_color,self.bg_color,self.bg_color))

        if self.plot_audio_history:
            self.screen.set_alpha(255)
            self.prev_screen = self.screen

        pygame.display.set_caption('Spectrum Analyzer -- (FFT-Peak: %05d Hz)' %self.ear.strongest_frequency)
        self.bin_font = pygame.font.Font('freesansbold.ttf', round(0.025*self.HEIGHT))
        self.fps_font = pygame.font.Font('freesansbold.ttf', round(0.05*self.HEIGHT))

        for i in range(self.ear.n_frequency_bins):
            if i == 0 or i == (self.ear.n_frequency_bins - 1):
                continue
            if i % self.tag_every_n_bins == 0:
                f_centre = self.ear.frequency_bin_centres[i]
                text = self.bin_font.render('%d Hz' %f_centre, True, (255, 255, 255) , (self.bg_color, self.bg_color, self.bg_color))
                textRect = text.get_rect()
                x = i*(self.WIDTH / self.ear.n_frequency_bins) + (self.bar_width - textRect.x)/2
                y = 0.98*self.HEIGHT
                textRect.center = (int(x),int(y))
                self.bin_text_tags.append(text)
                self.bin_rectangles.append(textRect)

        self._is_running = True

        #Interactive components:
        self.button_height = round(0.05*self.HEIGHT)
        self.history_button  = Button(text="Toggle 2D/3D Mode", right=self.WIDTH, top=0, width=round(0.12*self.WIDTH), height=self.button_height)
        self.slow_bar_button = Button(text="Toggle Slow Bars", right=self.WIDTH, top=self.history_button.height, width=round(0.12*self.WIDTH), height=self.button_height)
        self.record_button = Button(text="Record", right=self.WIDTH, top=(self.slow_bar_button.height + self.history_button.height), width=round(0.12*self.WIDTH), height=self.button_height)
        self.record2_button = Button(text="Overlay Record", right=self.WIDTH, top=(self.slow_bar_button.height + self.history_button.height + self.record_button.height), width=round(0.12*self.WIDTH), height=self.button_height)
        self.record3_button = Button(text="Reverse Record", right=self.WIDTH, top=(self.slow_bar_button.height + self.history_button.height + self.record_button.height + self.record2_button.height), width=round(0.12*self.WIDTH), height=self.button_height)
        self.record4_button = Button(text="Combine Record", right=self.WIDTH, top=(self.slow_bar_button.height + self.history_button.height + self.record_button.height + self.record2_button.height + self.record3_button.height), width=round(0.12*self.WIDTH), height=self.button_height)
        self.record5_button = Button(text="Fade In/Fade Out", right=self.WIDTH, top=(self.slow_bar_button.height + self.history_button.height + self.record_button.height + self.record2_button.height + self.record3_button.height + self.record4_button.height), width=round(0.12*self.WIDTH), height=self.button_height)
        self.record6_button = Button(text="Speed Up", right=self.WIDTH, top=(self.slow_bar_button.height + self.history_button.height + self.record_button.height + self.record2_button.height + self.record3_button.height + self.record4_button.height + self.record5_button.height), width=round(0.12*self.WIDTH), height=self.button_height)
        self.record7_button = Button(text="Slow Down", right=self.WIDTH, top=(self.slow_bar_button.height + self.history_button.height + self.record_button.height + self.record2_button.height + self.record3_button.height + self.record4_button.height + self.record5_button.height + self.record5_button.height), width=round(0.12*self.WIDTH), height=self.button_height)

    def stop(self):
        print("Stopping spectrum visualizer...")
        del self.fps_font
        del self.bin_font
        del self.screen
        del self.prev_screen
        pygame.quit()
        self._is_running = False

    def toggle_display(self):
        '''
        This function can be triggered to turn on/off the display
        '''
        if self._is_running: self.stop()
        else: self.start()

    def update(self):
        for event in pygame.event.get():
            if self.history_button.click():
                self.plot_audio_history = not self.plot_audio_history
                self.toggle_history_mode()
            if self.slow_bar_button.click():
                self.add_slow_bars = not self.add_slow_bars
                self.slow_features = [0]*self.ear.n_frequency_bins
            if self.record_button.click():
                if not self.recording:
                    self.recording = True
                    print("Recording started...")
                    self.audio_data = sd.rec(int(30 * RATE), samplerate=RATE, channels=CHANNELS, dtype='int16')
                else:
                    self.recording = False
                    sd.wait()  # Ensure recording is completed
                    self.audio_data = self.extract_white_noise(self.audio_data.flatten())  # Apply noise reduction
                    with wave.open("recorded_audio.wav", 'wb') as wavefile:
                        wavefile.setnchannels(CHANNELS)
                        wavefile.setsampwidth(2)
                        wavefile.setframerate(RATE)
                        wavefile.writeframes(self.audio_data.tobytes())
                    print("Recording saved as recorded_audio.wav with noise reduction")
            if self.record2_button.click():
                sound1 = AudioSegment.from_file("recorded_audio.wav", format="wav")
                sound2 = AudioSegment.from_file("background.wav", format="wav")

                needed_duration = 30_000  # 30 seconds in milliseconds
                while len(sound2) < needed_duration:
                    sound2 += sound2  # Repeat until it's at least 30s

# Trim to exactly 30s
                sound2 = sound2[:needed_duration]

                sound2 = sound2 - 10

# sound1 6 dB louder
                sound1 = sound1 + 6

# Overlay sound2 over sound1 at position 0  (use louder instead of sound1 to use the louder version)
                overlay = sound1.overlay(sound2, position=0)


# simple export
                file_handle = overlay.export("overlay.wav", format="wav")

            if self.record3_button.click():
                sound1 = AudioSegment.from_file("recorded_audio.wav", format="wav")
                reversed_audio = sound1.reverse()    

                find_handle = reversed_audio.export("reverse.wav", format="wav")

            if self.record4_button.click():
                sound1 = AudioSegment.from_file("recorded_audio.wav", format="wav")
                sound2 = AudioSegment.from_file("intro.wav", format="wav")

# sound1 6 dB louder
                louder = sound1 + 6


# sound1, with sound2 appended (use louder instead of sound1 to append the louder version)
                combined = sound2 + louder + sound2

# simple export
                file_handle = combined.export("combine.wav", format="wav")
            if self.record5_button.click():
                sound = AudioSegment.from_file("recorded_audio.wav", format="wav")
                faded = sound.fade_in(2000).fade_out(2000)  # 2s fade
                file_handle = faded.export("fadein.wav", format="wav")

            if self.record6_button.click():
                sound = AudioSegment.from_file("recorded_audio.wav", format="wav")
                faster = sound.speedup(playback_speed=1.5)
  # 2s fade
                file_handle = faster.export("fast.wav", format="wav")
            
            if self.record7_button.click():
                sound = AudioSegment.from_file("recorded_audio.wav", format="wav")
                slower_sound = self.slow_down(sound, 0.5) 
                slower_sound = slower_sound + 10

# Export the slowed audio
                slower_sound.export("slow.mp3", format="mp3")
  # 2s fade
               

            


            


        if np.min(self.ear.bin_mean_values) > 0:
            self.frequency_bin_energies = self.avg_energy_height * self.ear.frequency_bin_energies / self.ear.bin_mean_values

        if self.plot_audio_history:
            new_w, new_h = int((2+self.shrink_f)/3*self.WIDTH), int(self.shrink_f*self.HEIGHT)
            #new_w, new_h = int(self.shrink_f*self.WIDTH), int(self.shrink_f*self.HEIGHT)

            horizontal_pixel_difference = self.WIDTH - new_w
            prev_screen = pygame.transform.scale(self.prev_screen, (new_w, new_h))

        self.screen.fill((self.bg_color,self.bg_color,self.bg_color))

        if self.plot_audio_history:
            new_pos = int(self.move_fraction*self.WIDTH - (0.0133*self.WIDTH)), int(self.move_fraction*self.HEIGHT)
            self.screen.blit(pygame.transform.rotate(prev_screen, 180), new_pos)

        if self.start_time is None:
           self.start_time = time.time()

        self.vis_steps += 1

        if self.vis_steps%self.fps_interval == 0:
            self.fps = self.fps_interval / (time.time()-self.start_time)
            self.start_time = time.time()

        self.text = self.fps_font.render('Fps: %.1f' %(self.fps), True, (255, 255, 255) , (self.bg_color, self.bg_color, self.bg_color))
        self.textRect = self.text.get_rect()
        self.textRect.x, self.textRect.y = round(0.015*self.WIDTH), round(0.03*self.HEIGHT)
        pygame.display.set_caption('Spectrum Analyzer -- (FFT-Peak: %05d Hz)' %self.ear.strongest_frequency)

        self.plot_bars()

        #Draw text tags:
        self.screen.blit(self.text, self.textRect)
        if len(self.bin_text_tags) > 0:
            cnt = 0
            for i in range(self.ear.n_frequency_bins):
                if i == 0 or i == (self.ear.n_frequency_bins - 1):
                    continue
                if i % self.tag_every_n_bins == 0:
                    self.screen.blit(self.bin_text_tags[cnt], self.bin_rectangles[cnt])
                    cnt += 1

        self.history_button.draw(self.screen)
        self.slow_bar_button.draw(self.screen)
        self.record_button.draw(self.screen)
        self.record2_button.draw(self.screen)
        self.record3_button.draw(self.screen)
        self.record4_button.draw(self.screen)
        self.record5_button.draw(self.screen)
        self.record6_button.draw(self.screen)
        self.record7_button.draw(self.screen)

        pygame.display.flip()


    def plot_bars(self):
        bars, slow_bars, new_slow_features = [], [], []
        local_height = self.y_ext[1] - self.y_ext[0]
        feature_values = self.frequency_bin_energies[::-1]

        for i in range(len(self.frequency_bin_energies)):
            feature_value = feature_values[i] * local_height

            self.fast_bars[i][3] = int(feature_value)

            if self.plot_audio_history:
                self.fast_bars[i][3] = int(feature_value + 0.02*self.HEIGHT)

            if self.add_slow_bars:
                self.decay = min(0.99, 1 - max(0,self.decay_speed * 60 / self.ear.fft_fps))
                slow_feature_value = max(self.slow_features[i]*self.decay, feature_value)
                new_slow_features.append(slow_feature_value)
                self.slow_bars[i][1] = int(self.fast_bars[i][1] + slow_feature_value)
                self.slow_bars[i][3] = int(self.slow_bar_thickness * local_height)

        if self.add_fast_bars:
            for i, fast_bar in enumerate(self.fast_bars):
                pygame.draw.rect(self.screen,self.fast_bar_colors[i],fast_bar,0)

        if self.plot_audio_history:
                self.prev_screen = self.screen.copy().convert_alpha()
                self.prev_screen = pygame.transform.rotate(self.prev_screen, 180)
                self.prev_screen.set_alpha(self.prev_screen.get_alpha()*self.alpha_multiplier)

        if self.add_slow_bars:
            for i, slow_bar in enumerate(self.slow_bars):
                pygame.draw.rect(self.screen,self.slow_bar_colors[i],slow_bar,0)

        self.slow_features = new_slow_features

        #Draw everything:
        self.screen.blit(pygame.transform.rotate(self.screen, 180), (0, 0))

