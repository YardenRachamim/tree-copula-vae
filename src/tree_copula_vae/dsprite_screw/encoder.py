import torch.nn as nn


class StrongEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=256):
        super().__init__()

        # שדרוג: 4 שכבות במקום 3, עם BatchNorm ו-LeakyReLU ליציבות
        self.cnn = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4 (שכבה נוספת להפשטה טובה יותר)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Flatten: 256 channels * 4 * 4 = 4096
        self.flatten_dim = 256 * 4 * 4
        # Linear Head
        self.fc = nn.Linear(self.flatten_dim, hidden_dim)

    def forward(self, x):
        h = self.cnn(x)
        h = h.flatten(1)
        return self.fc(h)


