import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import os
class TorgoDataset(Dataset):
    def __init__(self, metadata_df, target_sample_rate=16000, max_duration=3.0):
        self.metadata = metadata_df
        self.target_sample_rate = target_sample_rate
        self.max_length = int(target_sample_rate * max_duration)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    def __len__(self):
        return len(self.metadata)
    def _pad_or_truncate(self, waveform):
        if waveform.shape[1] > self.max_length:
            return waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, padding))
        return waveform
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = row['file_path']
        label = row['severity_label']
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            waveform = self._pad_or_truncate(waveform)
            mel_spec = self.mel_spectrogram(waveform)
            mel_spec = self.amplitude_to_db(mel_spec)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return mel_spec, label_tensor
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            dummy_spec = torch.zeros((1, 64, self.max_length // 512 + 1))
            return dummy_spec, torch.tensor(label, dtype=torch.long)
if __name__ == "__main__":
    if os.path.exists("data/processed/torgo_metadata.csv"):
        df = pd.read_csv("data/processed/torgo_metadata.csv")
        dataset = TorgoDataset(df)
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            mel_spec, label = dataset[0]
            print(f"Mel-spectrogram shape: {mel_spec.shape}, Label: {label}")
    else:
        print("Metadata not found. Please run parse_torgo.py first.")
