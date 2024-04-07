import numpy as np
import sound_file as sf
import librosa
import pyworld
from text.parser import g2p

from pitch import (extract_smoothed_continuous_f0, extract_smoothed_f0, 
                   hz_to_cent_based_c4, extract_vibrato_likelihood, 
                   extract_vibrato_parameters, interp1d)

class FileDataSource(object):
    """File data source interface.

    Users are expected to implement custum data source for your own data.
    All file data sources must implement this interface.
    """

    def collect_files(self):
        """Collect data source files

        Returns:
            List or tuple of list: List of files, or tuple of list if you need
            multiple files to collect features.
        """
        raise NotImplementedError

    def collect_features(self, *args):
        """Collect features given path(s).

        Args:
            args: File path or tuple of file paths

        Returns:
            2darray: ``T x D`` features represented by 2d array.
        """
        raise NotImplementedError
    
class TextDataSource(FileDataSource):
    def __init__(
        self
    ): ...
    
    def collect_features(self, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        phones, ids = g2p(text)
        
        return phones, ids
        

class WORLDAcousticSoure(FileDataSource):
    def __init__(
        self,
        wav_root,
        f0_extractor="harvest",
        f0_floor=150,
        f0_ceil=700,
        frame_period=5,
        vibrato_mode="none",  # diff, sine
        sample_rate=48000,
        d4c_threshold=0.85,
        trajectory_smoothing=False,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_f0=True,
        trajectory_smoothing_cutoff_f0=20,
        decouple_pitch_mode=False,
    ):
        self.wav_root = wav_root
        self.f0_extractor = f0_extractor
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.frame_period = frame_period
        self.vibrato_mode = vibrato_mode
        self.sample_rate = sample_rate
        self.d4c_threshold = d4c_threshold
        self.trajectory_smoothing = trajectory_smoothing
        self.trajectory_smoothing_cutoff = trajectory_smoothing_cutoff
        self.trajectory_smoothing_f0 = trajectory_smoothing_f0
        self.trajectory_smoothing_cutoff_f0 = trajectory_smoothing_cutoff_f0
        
    def collect_features(self, wav_path):
        if self.f0_floor is not None:
            min_f0 = self.f0_floor
        if self.f0_ceil is not None:
            max_f0 = self.f0_ceil
        
        min_f0 = min(min_f0, 500)
        
        x, fs = sf.read(wav_path)
        assert np.max(x) <= 1.0
        assert x.dtype == np.float64
        
        frame_shift_samples = int(self.frame_period * 0.001 * fs)
        num_frames = len(x) // frame_shift_samples
        
        if fs != self.sample_rate:
            x = librosa.resample(
                x, orig_sr=fs, target_sr=self.sample_rate, res_type=self.res_type
            )
            fs = self.sample_rate
            
        if self.f0_extractor == "harvest":
            f0, timeaxis = pyworld.harvest(
                x, fs, frame_period=self.frame_period, f0_floor=min_f0, f0_ceil=max_f0
            )
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        elif self.f0_extractor == "dio":
            f0, timeaxis = pyworld.dio(
                x, fs, frame_period=self.frame_period, f0_floor=min_f0, f0_ceil=max_f0
            )
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        else:
            raise ValueError(f"unknown f0 extractor: {self.f0_extractor}")
        
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs, threshold=self.d4c_threshold)
        
        if np.isnan(aperiodicity).any():
            print(wav_path)
            print(min_f0, max_f0, aperiodicity.shape, fs)
            print(np.isnan(aperiodicity).sum())
            print(aperiodicity)
            raise RuntimeError("Aperiodicity has NaN")
        
        sr_f0 = int(1 / (self.frame_period * 0.001))
        
        lf0 = f0[:, np.newaxis].copy()
        nonzero_indices = np.nonzero(lf0)
        lf0[nonzero_indices] = np.log(f0[:, np.newaxis][nonzero_indices])
        if self.f0_extractor == "harvest":
            vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
        else:
            vuv = (lf0 != 0).astype(np.float32)
            
        lf0 = interp1d(lf0, kind="slinear")
        
        if self.trajectory_smoothing_f0:
            lf0 = extract_smoothed_continuous_f0(
                lf0, sr_f0, cutoff=self.trajectory_smoothing_cutoff_f0
            )
            
        if self.vibrato_mode == "sine":
            win_length = 64
            n_fft = 256
            threshold = 0.12
            
            if self.f0_extractor == "harvest":
                # NOTE: harvest is not supported here since the current implemented algorithm
                # relies on v/uv flags to find vibrato sections.
                # We use DIO since it provides more accurate v/uv detection in my experience.
                _f0, _timeaxis = pyworld.dio(
                    x,
                    fs,
                    frame_period=self.frame_period,
                    f0_floor=min_f0,
                    f0_ceil=max_f0,
                )
                _f0 = pyworld.stonemask(x, _f0, _timeaxis, fs)
                f0_smooth = extract_smoothed_f0(_f0, sr_f0, cutoff=8)
            else:
                f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=8)
                
            f0_smooth_cent = hz_to_cent_based_c4(f0_smooth)
            vibrato_likelihood = extract_vibrato_likelihood(
                f0_smooth_cent, sr_f0, win_length=win_length, n_fft=n_fft
            )
            vib_flags, m_a, m_f = extract_vibrato_parameters(
                f0_smooth_cent, vibrato_likelihood, sr_f0, threshold=threshold
            )
            m_a = interp1d(m_a, kind="linear")
            m_f = interp1d(m_f, kind="linear")
            vib = np.stack([m_a, m_f], axis=1)
            vib_flags = vib_flags[:, np.newaxis]
        elif self.vibrato_mode == "diff":
            # NOTE: vibrato is known to have 3 ~ 8 Hz range (in general)
            # remove higher frequency than 3 to separate vibrato from the original F0
            f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=3)
            assert len(f0.shape) == 1 and len(f0_smooth.shape) == 1
            vib = (f0 - f0_smooth)[:, np.newaxis]
            vib_flags = None
        elif self.vibrato_mode == "none":
            vib, vib_flags = None, None
        else:
            raise RuntimeError("Unknown vibrato mode: {}".format(self.vibrato_mode))
        
        lf0 = lf0[:num_frames]
        vuv = vuv[:num_frames]
        vib = vib[:num_frames] if vib is not None else None
        vib_flags = vib_flags[:num_frames] if vib_flags is not None else None

        # Align waveform and features
        wave = x.astype(np.float32)
        
        frame_shift_int = int(fs * self.frame_period / 1000)
        T = int(lf0.shape[0] * frame_shift_int)
        if len(wave) < T:
            if T - len(wave) > int(fs * (self.frame_period * 0.001)):
                print("Length mismatch", T, len(wave), T - len(wave))
                raise RuntimeError("Unaligned data")
            else:
                pass
            wave = np.pad(wave, (0, T - len(wave)))
        assert wave.shape[0] >= T
        wave = wave[:T]
        
        return lf0, vuv, vib, vib_flags, wave
