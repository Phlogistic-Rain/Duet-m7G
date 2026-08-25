"""Convolutional encoder used by the control architecture.

Three Conv1d-ReLU-MaxPool-Dropout stages over the ENAC encoding of the
sequence, optionally with batch normalisation after each convolution."""
import torch
from torch import nn


class Convolution(nn.Module):
    def __init__(self, seq_length, use_bn=False):
        """With use_bn=True a BatchNorm1d is inserted after every Conv1d, between the
        convolution and the ReLU. The convolutions stay at index 0 of each
        stage, so layers can still be addressed positionally."""
        super(Convolution, self).__init__()
        self.seq_length = seq_length - 1

        def _block(cin, cout):
            layers = [nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=5, stride=1, padding=2)]
            if use_bn:
                layers.append(nn.BatchNorm1d(cout))
            layers += [nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2)]
            return nn.Sequential(*layers)

        self.conv1 = _block(4, 64)
        self.conv2 = _block(64, 64)
        self.conv3 = _block(64, 64)
        self.fc = nn.Sequential(nn.Linear(64 * (self.seq_length // 8), 512),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )

    def forward(self, x, return_map=False):
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if return_map:
            return x
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Convolution(seq_length=501)

    batch_size = 2
    seq_len = 500
    input_channels = 4

    x = torch.randn(batch_size, seq_len, input_channels)

    output = model(x)

    print(output.shape)
