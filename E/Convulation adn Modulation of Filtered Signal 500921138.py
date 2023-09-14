# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:16:39 2023

@author: 62813
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Generate a sample signal
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector from 0 to 1 second
f1 = 5  # Frequency of the first sine wave (Hz)
f2 = 50  # Frequency of the second sine wave (Hz)
signal_input = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.random.randn(len(t))

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, signal_input)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.grid(True)

# Apply a low-pass Butterworth filter
nyquist_freq = 0.5 * fs
cutoff_freq = 10  # Cutoff frequency of the filter (Hz)
order = 4  # Filter order
b, a = signal.butter(order, cutoff_freq / nyquist_freq, btype='low')
filtered_signal = signal.lfilter(b, a, signal_input)

# Plot the filtered signal
plt.subplot(3, 1, 2)
plt.plot(t, filtered_signal)
plt.title('Filtered Signal (Low-pass Butterworth)')
plt.xlabel('Time (s)')
plt.grid(True)

# Compute and plot the FFT of the filtered signal
plt.subplot(3, 1, 3)
fft_result = np.fft.fft(filtered_signal)
frequencies = np.fft.fftfreq(len(t), 1/fs)
magnitude_spectrum = np.abs(fft_result)
plt.plot(frequencies[:len(frequencies)//2], magnitude_spectrum[:len(frequencies)//2])
plt.title('FFT of Filtered Signal')
plt.xlabel('Frequency (Hz)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Perform convolution
kernel = np.array([1, 2, 3, 2, 1])  # Example kernel
convolved_signal = np.convolve(filtered_signal, kernel, mode='same')

# Plot the convolved signal
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.plot(t, convolved_signal, label='Convolved Signal')
plt.title('Convolution of Filtered Signal')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)

# Perform modulation
carrier_frequency = 100  # Modulation carrier frequency (Hz)
modulated_signal = np.cos(2 * np.pi * carrier_frequency * t) * filtered_signal

# Plot the modulated signal
plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.plot(t, modulated_signal, label='Modulated Signal')
plt.title('Modulation of Filtered Signal')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
