import torch.nn as nn

class MNISTEncoderBackbone(nn.Module):
    """
    Input: Image [B, 1, 28, 28]
    Output: Feature Vector [B, hidden_dim]
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            # nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            # nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(), # 64 * 7 * 7 = 3136
            
            # דחיסה ראשונית לוקטור ייצוג כללי
            nn.Linear(3136, hidden_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x)

