import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
import logging
from pandas import DataFrame
from scipy import stats
from scipy.signal import welch, argrelextrema
import pyeeg
from typing import Optional, List, Tuple, Dict
from mne import io

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


class EEGProcessor:
    def __init__(self, processor : SignalProcessor, epoch_length : int = 10, step_size : int = 1):
        """
        :param processor: extracts features from each window
        :param epoch_length: the length of each window (in seconds, where the default is 10)
        :param step_size: sliding window, we move the window by 1 second (allows for overlapping epochs)

        This setup enables the class to segment continuous EEG data into manageable,
        labeled chunks for feature extraction and machine learning.
        """
        self.processor = processor
        self.epoch_length = epoch_length
        self.step_size = step_size
        self.logger = logging.getLogger(__name__)


    def load_and_filter(self, file_path:str) -> io.Raw:
        """
        load raw EEG data (edf format) from a specified file and filters
        Applies bandpass filter 0.25–25 Hz, useful for capturing most EEG activity while removing noise.
        """
        raw = io.read_raw_edf(file_path, preload=True)
        raw.filter(l_freq=0.25, h_freq=25)
        return raw


    def process_epoch(
            self,
            raw: io.Raw,
            start_time: float,
            seizure_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict:
        start, stop = raw.time_as_index([start_time, start_time + self.epoch_length])
        data = raw[:, start:stop][0]

        features = {"start_time": start_time}

        for idx, channel in enumerate(raw.ch_names):
            channel_features = self.processor.extract_all_features(data[idx])
            features.update(
                {
                    f"{channel}_{key}": value
                    for feature_type in vars(channel_features).values()
                    for key, value in feature_type.items()
                }
            )

        if seizure_intervals:
            features["seizure"] = any(
                start_time > start
                and start_time < end
                or start_time + self.epoch_length > start
                and start_time + self.epoch_length < end
                for start, end in seizure_intervals
            )
        else:
            features["seizure"] = 0

        return features


    def process_recording(self,
                          file_path: str,
                          seizure_intervals: Optional[List[Tuple[float, float]]] = None, ) -> pd.DataFrame:

        raw = self.load_and_filter(file_path)

        start_time = 0
        epochs = []
        while start_time <= raw.times[-1] - self.epoch_length: # divides the full EEG signal into sliding windows (epochs)
            self.logger.info(f"Processing epoch starting at {start_time}s")
            epoch_features = self.process_epoch(raw, start_time, seizure_intervals)
            epochs.append(epoch_features)
            start_time += self.step_size

        return pd.DataFrame(epochs)


def main():
    logging.basicConfig(level=logging.INFO)

    data_dir = 'dataset'
    output_dir = 'processed_data'

    seizure_info = {
        "chb01_03": [[2996, 3036]],
        "chb01_04": [[1467, 1494]],
        "chb01_15": [[1732, 1772]],
        "chb01_16": [[1015, 1066]],
        "chb01_18": [[1720, 1810]],
        "chb01_21": [[327, 420]],
        "chb01_26": [[1862, 1963]],
        "chb02_16": [[130, 212]],
        "chb02_16+": [[2972, 3053]],
        "chb02_19": [[3369, 3378]],
        "chb05_06": [[417, 532]],
        "chb05_13": [[1086, 1196]],
        "chb05_16": [[2317, 2413]],
        "chb05_17": [[2451, 2571]],
        "chb05_22": [[2348, 2465]],
    }

    signal_processor = SignalProcessor()
    eeg_processor = EEGProcessor(signal_processor)

    # core loop that processes EEG .edf files and saves preprocessed .csv feature files
    for filename in os.listdir(data_dir):
        if filename.endswith(".edf"):
            file_path = os.path.join(data_dir, filename)
            recording_id = os.path.splitext(filename)[0]

            seizure_intervals = seizure_info.get(recording_id)

            features_df = eeg_processor.process_recording(file_path, seizure_intervals)

            output_path = os.path.join(output_dir, f"{recording_id}.csv")
            features_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()