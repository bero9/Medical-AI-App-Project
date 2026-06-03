import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) Residual blocks (محسّنة)
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1), dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.downsample = None
        if stride != (1, 1) or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity)
        return out


# -----------------------------------------
# 2) Optional: Spatial Transformer (STN)
# لتصحيح الميلان/المنظور تلقائياً قبل الـCNN
# -----------------------------------------
class STNRectifier(nn.Module):
    """
    STN خفيف لتقويم النص المائل/المشوه (مفيد جدًا لصور الكاميرا).
    يمكن تعطيله بسهولة من CRNN.
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(True),
            nn.Linear(32, 6)
        )
        # initialize as identity transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0, 0,1,0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x).view(x.size(0), -1)  # [B,64]
        theta = self.fc_loc(xs).view(-1, 2, 3)         # [B,2,3]
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


# -----------------------------------------
# 3) Sequence module: BiLSTM + projection
# (CTC-friendly)
# -----------------------------------------
class BiLSTM(nn.Module):
    def __init__(self, in_size, hidden, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            in_size, hidden, num_layers=layers,
            bidirectional=True, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm(hidden * 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out)
        return out


# -----------------------------------------
# 4) Upgraded CRNN
# -----------------------------------------
class StrongCRNN(nn.Module):
    def __init__(
        self,
        img_height=32,
        num_channels=1,
        num_classes=120,     # vocab + blank
        lstm_hidden=256,
        lstm_layers=2,
        use_stn=True,
        cnn_dropout=0.1
    ):
        super().__init__()

        self.use_stn = use_stn
        if use_stn:
            self.stn = STNRectifier(in_ch=num_channels)

        # CNN Backbone أقوى (أقرب لـ ResNet OCR)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.MaxPool2d(2, 2),               # H/2

            ResidualBlock(64, 128, stride=(2,1), dropout=cnn_dropout),   # H/4
            ResidualBlock(128, 128, stride=(1,1), dropout=cnn_dropout),

            ResidualBlock(128, 256, stride=(2,1), dropout=cnn_dropout),  # H/8
            ResidualBlock(256, 256, stride=(1,1), dropout=cnn_dropout),

            ResidualBlock(256, 512, stride=(2,1), dropout=cnn_dropout),  # H/16
            ResidualBlock(512, 512, stride=(1,1), dropout=cnn_dropout),

            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),

            nn.MaxPool2d((2,1), (2,1)),      # H/32, W ثابت تقريبًا
        )

        # حساب LSTM input size تلقائي
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, img_height, 100)
            feat = self.cnn(dummy)
            self.lstm_input = feat.size(1) * feat.size(2)  # C * H'

        self.seq = BiLSTM(self.lstm_input, lstm_hidden, layers=lstm_layers, dropout=0.2)

        # CTC Head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden * 2, num_classes)
        )

        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x, labels=None, label_lengths=None):
        # 1) STN تصحيح
        if self.use_stn:
            x = self.stn(x)

        # 2) CNN features
        x = self.cnn(x)  # [B, C, H', W']

        # 3) To sequence: time = W'
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        x = x.view(B, W, C * H)                 # [B, W, C*H]

        # 4) BiLSTM
        x = self.seq(x)                         # [B, W, 2H]

        # 5) logits per timestep
        logits = self.head(x)                   # [B, W, classes]
        log_probs = F.log_softmax(logits, dim=2)
        log_probs = log_probs.permute(1, 0, 2)  # [T, B, classes]

        if labels is not None:
            input_lengths = torch.full((B,), log_probs.size(0), dtype=torch.long, device=log_probs.device)
            loss = self.ctc_loss(log_probs, labels, input_lengths, label_lengths)
            return log_probs, loss

        return log_probs, None
