import os
import torch
import torchaudio
import argparse
from src.models.baseline import DysarthriaCRNN
def predict(audio_path, model_path="models/best_model.pth"):
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Train the model first.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = DysarthriaCRNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_sample_rate = 16000
    max_length = target_sample_rate * 3

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_mels=64,
        n_fft=1024,
        hop_length=512
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    elif waveform.shape[1] < max_length:
        padding = max_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    mel_spec = mel_spectrogram(waveform)
    mel_spec = amplitude_to_db(mel_spec)
    mel_spec = mel_spec.unsqueeze(0).to(device)

    labels = ["Normal", "Mild", "Moderate", "Severe"]

    with torch.no_grad():
        output = model(mel_spec)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted_idx = torch.max(probs, dim=1)

    predicted_label = labels[predicted_idx.item()]
    print(f"Audio: {audio_path}")
    print(f"Predicted Severity: {predicted_label}")
    print(f"Confidence: {conf.item()*100:.2f}%")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict dysarthria severity from an audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the .wav audio file")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to trained model")

    args = parser.parse_args()
    predict(args.audio_path, args.model)
