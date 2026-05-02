import torch
import torch.nn as nn
class DysarthriaCRNN(nn.Module):
    def __init__(self, num_classes=4):
        super(DysarthriaCRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(input_size=128 * 8, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(128 * 2, num_classes)
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
if __name__ == "__main__":
    model = DysarthriaCRNN()
    dummy_input = torch.randn(4, 1, 64, 94)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
