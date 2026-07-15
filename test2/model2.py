import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """CNN Encoder designed for 128x400 math expression images.

    Extracts spatial features and flattens them into a 2D feature grid sequence
    preserving vertical and horizontal spatial relationships.
    """

    def __init__(self, out_channels=512):
        super(CNNEncoder, self).__init__()

        # Conv Block 1: 1x128x400 -> 64x64x200
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: 64x64x200 -> 128x32x100
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: 128x32x100 -> 256x16x50
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 4: 256x16x50 -> 512x8x50 (Asymmetric pool to keep width)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Conv Block 5: 512x8x50 -> out_channels x 4 x 50
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.relu(self.bn5(self.conv5(x)))
        features = self.pool5(x)  # Shape: [B, out_channels, 4, 50]

        # Reshape to 1D sequence of visual features: [B, 200, out_channels]
        batch_size, channels, h, w = features.size()
        features = features.view(batch_size, channels, h * w)
        features = features.permute(0, 2, 1)

        return features


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) Attention Module."""

    def __init__(self, encoder_dim, decoder_dim, attn_dim=256):
        super(BahdanauAttention, self).__init__()
        self.W_encoder = nn.Linear(encoder_dim, attn_dim)
        self.W_decoder = nn.Linear(decoder_dim, attn_dim)
        self.v_attn = nn.Linear(attn_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        enc_proj = self.W_encoder(encoder_outputs)  # [B, num_regions, attn_dim]
        dec_proj = self.W_decoder(decoder_hidden).unsqueeze(1)  # [B, 1, attn_dim]

        # Score computation: e_i = v^T * tanh(W_e * h_i + W_d * s_t)
        scores = self.v_attn(torch.tanh(enc_proj + dec_proj))  # [B, num_regions, 1]
        scores = scores.squeeze(2)  # [B, num_regions]

        attention_weights = F.softmax(scores, dim=1)  # [B, num_regions]

        # Spatial context vector calculation
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        )  # [B, 1, encoder_dim]
        context_vector = context_vector.squeeze(1)  # [B, encoder_dim]

        return context_vector, attention_weights


class RNNDecoder(nn.Module):
    """LSTM-based Text Decoder working with dynamic spatial Attention."""

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        encoder_dim=512,
        decoder_dim=512,
        attn_dim=256,
    ):
        super(RNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(
            encoder_dim, decoder_dim, attn_dim=attn_dim
        )

        self.lstm_cell = nn.LSTMCell(
            input_size=embed_dim + encoder_dim, hidden_size=decoder_dim
        )
        self.out_proj = nn.Linear(decoder_dim + encoder_dim, vocab_size)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def init_hidden_states(self, encoder_outputs):
        mean_encoder_features = encoder_outputs.mean(dim=1)
        h0 = torch.tanh(self.init_h(mean_encoder_features))
        c0 = torch.tanh(self.init_c(mean_encoder_features))
        return h0, c0

    def forward_step(self, prev_token, encoder_outputs, h, c):
        embedded = self.embedding(prev_token)
        context, attn_weights = self.attention(encoder_outputs, h)
        lstm_input = torch.cat([embedded, context], dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))

        output_feat = torch.cat([h, context], dim=1)
        logits = self.out_proj(output_feat)

        return logits, h, c, attn_weights


class HMER_Model(nn.Module):
    """End-to-End Handwritten Mathematical Expression Recognition System."""

    def __init__(
        self,
        vocab_size,
        sos_idx,
        eos_idx,
        embed_dim=256,
        encoder_dim=512,
        decoder_dim=512,
        attn_dim=256,
    ):
        super(HMER_Model, self).__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.encoder = CNNEncoder(out_channels=encoder_dim)
        self.decoder = RNNDecoder(
            vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attn_dim=attn_dim,
        )

    def forward(self, images, targets=None, max_len=150, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)

        # 1. Extract 2D features
        encoder_outputs = self.encoder(images)

        # 2. Initialize LSTM Decoder State
        h, c = self.decoder.init_hidden_states(encoder_outputs)

        # Setup variables
        target_len = targets.size(1) if targets is not None else max_len
        all_predicted_logits = []
        all_attentions = []

        curr_token = torch.full(
            (batch_size,), self.sos_idx, dtype=torch.long, device=images.device
        )

        for t in range(1, target_len):
            logits, h, c, attn_weights = self.decoder.forward_step(
                curr_token, encoder_outputs, h, c
            )

            all_predicted_logits.append(logits.unsqueeze(1))
            all_attentions.append(attn_weights.unsqueeze(1))

            if targets is not None and random.random() < teacher_forcing_ratio:
                curr_token = targets[:, t]
            else:
                curr_token = logits.argmax(dim=-1)

        all_predicted_logits = torch.cat(all_predicted_logits, dim=1)
        all_attentions = torch.cat(all_attentions, dim=1)

        return all_predicted_logits, all_attentions