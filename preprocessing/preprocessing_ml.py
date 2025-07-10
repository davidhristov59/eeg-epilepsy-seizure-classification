import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
import logging
from scipy import stats
from scipy.signal import welch, argrelextrema
import pyeeg
from typing import Optional, List, Tuple, Dict

@dataclass
class EEGFeatures:
    temporal: Dict[str, float] # Time-domain features (mean,variance, rms)
    spectral: Dict[str, float] # Frequency-domain features (spectral entropy, band powers)
    nonlinear: Dict[str, float] # Complexity/chaos features (Hjorth parameters, fractal dimensions)

class SignalProcessor:
    def __init__(self, sampling_rate : int = 256):
        self.fs = sampling_rate
        self.freq_bands = [1, 5, 10, 15, 20, 25] #edges

        # self.freq_bands = {
        #     'delta' : (1, 5), # 1-5Hz
        #     'theta' : (5,10), # 5-10Hz
        #     'alpha' : (10-15), # 10-15Hz
        #     'low-beta' : (15-20), # 15-20Hz
        #     'high-beta' : (20-25), # 20-25Hz
        #     'gamma': (25, self.fs // 2) #Nyquist limit (128)
        # }

    def extract_temporal_features(self, signal: np.ndarray) -> Dict[str, float]:
        return {
            "mean" : float(np.mean(signal)),
            "variance" : float(np.var(signal)),
            "skewness":  float(stats.skew(signal)),
            "kurtosis" : float(stats.kurtosis(signal)),
            "rms": float(np.sqrt(np.mean(np.square(signal)))), #Root Mean Square
            "zero_crossings": float(np.sum(np.diff(np.signbit(signal).astype(int)) != 0)),
            "peak_amp": np.max(np.abs(signal)),
            "peak_count": len(argrelextrema(signal, np.greater)[0]),
        }

    def extract_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:

        # Compute the Power Spectral Density (PSD) using Welch's method
        freqs, psd = welch(signal, fs=self.fs)

        # Compute band power and power ratio for defined frequency bands using PyEEG
        power, power_ratio = pyeeg.bin_power(signal, self.freq_bands, self.fs)

        return {
            "total_power": np.sum(psd), #total power across all frequencies
            "median_freq": freqs[np.where(np.cumsum(psd) >= np.sum(psd) / 2)[0][0]],
            "peak_freq": freqs[np.argmax(psd)],
            "spectral_entropy": pyeeg.spectral_entropy(
                signal, self.freq_bands, self.fs
            ),
                # band power ratios for each frequency band
            **{
                f"band_power_{band}": ratio
                for band, ratio in zip(self.freq_bands[:-1], power_ratio)
            },
        }

    def extract_nonlinear_features(self, signal: np.ndarray) -> Dict[str, float]:

        # Hjorth parameters: activity, mobility, complexity
        activity, mobility, complexity = pyeeg.hjorth(signal)

        # Higuchi Fractal Dimension (HFD)
        hfd = pyeeg.hfd(signal, Kmax=5)

        # Petrosian Fractal Dimension (PFD)
        pfd = pyeeg.pfd(signal)

        # Hurst exponent
        hurst = pyeeg.hurst(signal)

        return {
            "hfd" : hfd,
            "pfd" : pfd,
            "hurst" : hurst,
            "hjorth_activity" : activity,
            "hjorth_mobility" : mobility,
            "hjorth_complexity" : complexity
        }


    def extract_all_features(self, signal: np.ndarray) -> EEGFeatures:
        return EEGFeatures(
            temporal = self.extract_temporal_features(signal),
            spectral = self.extract_spectral_features(signal),
            nonlinear = self.extract_nonlinear_features(signal),
        )
