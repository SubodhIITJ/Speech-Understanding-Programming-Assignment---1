# -*- coding: utf-8 -*-
"""M23CSA531_PA1"""

from google.colab import drive
drive.mount('/content/drive')

"""## **Question 1: Text-To-Speech (TTS)**

First, we install the necessary dependencies.
In addition to torchaudio, DeepPhonemizer is required to perform phoneme-based encoding.
"""

import torch
import torchaudio
import IPython
import matplotlib.pyplot as plt

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# pip3 install deep_phonemizer

"""Setting the Seed and cuda."""

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)
print(torchaudio.__version__)
print(device)

"""Text Processing - Character-based encoding

The pre-trained Tacotron2 model is designed to work with a specific set of symbols (letters, punctuation, and special characters). These symbols must be converted into numeric IDs that the model can process. While libraries like torchaudio provide this functionality, manually implementing the encoding process helps you understand the underlying mechanics.

We start by defining the set of symbols _-!'(),.:;? abcdefghijklmnopqrstuvwxyz. Next, each character in the input text is mapped to the index of its corresponding symbol in the table, while any characters not present in the table are ignored.
"""

symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
look_up = {s: i for i, s in enumerate(symbols)}
symbols = set(symbols)


def text_to_sequence(text):
    text = text.lower()
    return [look_up[s] for s in text if s in symbols]


text = "The implementation and execution of a Tacotron2-based Text-to-Speech (TTS) pipeline using PyTorch and torchaudio."
print(text_to_sequence(text))

"""As noted above, the symbol table and indices must align with the requirements of the pretrained Tacotron2 model. Fortunately, torchaudio includes this transformation alongside the pretrained model. We can create and use this transformation as shown below."""

processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()

#text = "Hello world! Text to speech!"
processed, lengths = processor(text)

print(processed)
print(lengths)

"""
Note: The output from our manual encoding matches the result from torchaudio's text_processor, confirming that we correctly re-implemented the functionality of the library. The text_processor accepts either a single text or a list of texts as input. When a list of texts is provided, the returned lengths variable indicates the valid length of each processed token in the output batch."""

print([processor.tokens[i] for i in processed[0, : lengths[0]]])

"""Phoneme-based encoding

Similar to the case of character-based encoding, the encoding process is expected to match what a pretrained Tacotron2 model is trained on. torchaudio has an interface to create the process.
"""

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()

#text = "Hello world! Text to speech!"
with torch.inference_mode():
    processed, lengths = processor(text)

print(processed)
print(lengths)

"""Notice that the encoded values are different from the example of character-based encoding."""

print([processor.tokens[i] for i in processed[0, : lengths[0]]])

"""Spectrogram Generation

Tacotron2 is the model we use to generate spectrogram from the encoded text. torchaudio.pipelines.Tacotron2TTSBundle bundles the matching models and processors together so that it is easy to create the pipeline.
"""

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)

#text = "Hello world! Text to speech!"

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, _, _ = tacotron2.infer(processed, lengths)


_ = plt.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")

"""Note that Tacotron2.infer method perfoms multinomial sampling, therefore, the process of generating the spectrogram incurs randomness."""

def plot():
    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        with torch.inference_mode():
            spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        print(spec[0].shape)
        ax[i].imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")


plot()

"""Waveform Generation

Generate the waveform from the spectrogram using a vocoder - GriffinLim, WaveRNN and Waveglow.

WaveRNN Vocoder
"""

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

#text = "Hello world! Text to speech!"

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)

def plot(waveforms, spec, sample_rate):
    waveforms = waveforms.cpu().detach()

    fig, [ax1, ax2] = plt.subplots(2, 1)
    ax1.plot(waveforms[0])
    ax1.set_xlim(0, waveforms.size(-1))
    ax1.grid(True)
    ax2.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
    return IPython.display.Audio(waveforms[0:1], rate=sample_rate)


plot(waveforms, spec, vocoder.sample_rate)

"""Griffin-Lim Vocoder"""

bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
waveforms, lengths = vocoder(spec, spec_lengths)

plot(waveforms, spec, vocoder.sample_rate)

"""Waveglow Vocoder

Waveglow is a vocoder published by Nvidia. The pretrained weights are published on Torch Hub. One can instantiate the model using torch.hub module.
"""

waveglow = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub",
    "nvidia_waveglow",
    model_math="fp32",
    pretrained=False,
)
checkpoint = torch.hub.load_state_dict_from_url(
    "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",  # noqa: E501
    progress=False,
    map_location=device,
)
state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

waveglow.load_state_dict(state_dict)
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to(device)
waveglow.eval()

with torch.no_grad():
    waveforms = waveglow.infer(spec)

plot(waveforms, spec, 22050)

import numpy as np

def calculate_mos(ratings):
    mos_scores = [np.mean(sample_ratings) for sample_ratings in ratings]
    return mos_scores
ratings = [
    [4, 5, 3, 5, 4],
    [5, 5, 4, 4, 5],
    [3, 3, 4, 2, 3],
]
mos_scores = calculate_mos(ratings)

# Print MOS for each audio sample
for i, mos in enumerate(mos_scores, 1):
    print(f"waveforms{i}: MOS = {mos:.2f}")

"""# **Question 2: Task A (Windowing Techniques with UrbanSound8K Dataset)**

- Understand and implement the following windowing techniques:
a. Hann Window
b. Hamming Window
c. Rectangular Window
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import os
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Load an audio file
waveform, sample_rate = torchaudio.load("/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/UrbanSound8K/UrbanSound8K/audio/fold1/101415-3-0-2.wav")

# Define STFT parameters
n_fft = 1024
hop_length = 512

# Define windows
windows = {
    "hann": torch.hann_window(n_fft),
    "hamming": torch.hamming_window(n_fft),
    "rectangular": torch.ones(n_fft),
}

# Generate and plot spectrograms
def generate_spectrogram(waveform, sample_rate, window_name, window):
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    spectrogram = torch.abs(stft)
    spectrogram_db = AmplitudeToDB()(spectrogram)  # Convert to dB scale

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram_db[0].numpy(), origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram ({window_name.title()} Window)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()

# Generate and visualize spectrograms for all windows
for window_name, window in windows.items():
    generate_spectrogram(waveform, sample_rate, window_name, window)

from torchaudio.transforms import MelSpectrogram

# Define Mel Spectrogram transform
mel_transform = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=40)

# Apply Mel Spectrogram
mel_spec = mel_transform(waveform)
mel_spec_db = AmplitudeToDB()(mel_spec)  # Convert to dB

# Plot Mel Spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(mel_spec_db[0].numpy(), origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.show()

# Define dataset path and Load metadata
dataset_path = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/UrbanSound8K/UrbanSound8K"
metadata_file = os.path.join(dataset_path, "metadata/UrbanSound8K.csv")
metadata = pd.read_csv(metadata_file)

# Construct file paths and labels
file_paths = [
    os.path.join(dataset_path, "audio", f"fold{row['fold']}", row['slice_file_name'])
    for _, row in metadata.iterrows()
]
labels = metadata["classID"].tolist()

# Helper function to pad or truncate spectrograms
def pad_or_truncate(spec, max_length=400):
    """
    Pads or truncates the spectrogram to the specified max length.
    Args:
        spec: Input spectrogram (Tensor).
        max_length: Desired fixed length for the spectrogram.
    Returns:
        Padded or truncated spectrogram.
    """
    current_length = spec.shape[-1]
    if current_length > max_length:
        return spec[..., :max_length]
    elif current_length < max_length:
        padding = max_length - current_length
        return F.pad(spec, (0, padding), mode='constant', value=0)
    return spec

# Define the UrbanSound dataset class
import torchaudio
class UrbanSoundDataset(Dataset):
    def __init__(self, file_paths, labels, transform, max_length=400):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply the transformation (e.g., MelSpectrogram)
        features = self.transform(waveform)
        features = pad_or_truncate(features, max_length=self.max_length)
        label = self.labels[idx]
        return features, label

# Define a simple CNN
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Update the Linear layer's input size based on the output shape after convolution
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 100, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        print(f"Shape after convolution: {x.shape}")  # Debugging print statement
        x = self.fc(x)
        return x


# Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 3
n_fft = 1024
hop_length = 512
n_mels = 40

# Create dataset and dataloaders
transform = MelSpectrogram(sample_rate=22050, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
dataset = UrbanSoundDataset(file_paths, labels, transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = AudioClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for features, labels in train_loader:
    print(f"Features shape before model: {features.shape}")
    outputs = model(features)

    # Compute the loss
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch Loss: {loss.item():.4f}")

print(f"Number of audio files: {len(file_paths)}")
print(f"Sample file path: {file_paths[0]}")
print(f"Sample label: {labels[0]}")

"""# **Question 2: Task B (Spectrograms Analysis)**

Select 4 songs from 4 different genres and compare their spectrograms. Analyze the
spectrograms and provide a detailed comparative analysis based on your observations
and speech understanding.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Audio files
audio_files = {
    "Classical": "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Songs/classical_music.mp3",
    "Rock": "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Songs/rock_music.mp3",
    "Jazz": "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Songs/jazz_music.mp3",
    "Electronic": "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Songs/electronic_music.mp3"
}

# Function to plot spectrogram
def plot_spectrogram(audio_file, genre, ax):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)

        # Compute the Short-Time Fourier Transform (STFT)
        stft = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
        spectrogram = librosa.amplitude_to_db(abs(stft), ref=np.max)

        # Display the spectrogram
        img = librosa.display.specshow(
            spectrogram, sr=sr, hop_length=512,
            x_axis='time', y_axis='log', ax=ax, cmap='magma'
        )
        ax.set_title(f"{genre} Spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        return img
    except Exception as e:
        ax.set_title(f"{genre} - Error")
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', fontsize=10, color='red')
        ax.axis("off")
        return None

# Plot spectrograms for all genres
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
img = None

for ax, (genre, file) in zip(axs.flatten(), audio_files.items()):
    img = plot_spectrogram(file, genre, ax) or img  # Use last successful img

if img:
    cbar = fig.colorbar(img, ax=axs, orientation='vertical', shrink=0.6, aspect=10)
    cbar.set_label('Amplitude (dB)')

plt.tight_layout()
plt.show()