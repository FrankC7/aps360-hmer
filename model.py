import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding2D(nn.Module):
    """Static 2D Sinusoidal Positional Encoding for HMER.

    Encodes vertical (row) and horizontal (col) dimensions directly without
    relying on learnable parameters, ensuring zero noisy initialization.
    """

    def __init__(self, d_model, height=8, width=50):
        super(SinusoidalPositionalEncoding2D, self).__init__()
        assert (
            d_model % 4 == 0
        ), "d_model must be a multiple of 4 for 2D Sinusoidal PE"
        self.d_model = d_model

        d_row = d_model // 2
        d_col = d_model // 2

        pe = torch.zeros(d_model, height, width)

        # 1. Build row (vertical) sinusoidal encodings
        position_y = torch.arange(0, height, dtype=torch.float).unsqueeze(
            1
        )  # [H, 1]
        div_term_y = torch.exp(
            torch.arange(0, d_row, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_row)
        )  # [d_row // 2]
        pe_y_sin = torch.sin(position_y * div_term_y)  # [H, d_row // 2]
        pe_y_cos = torch.cos(position_y * div_term_y)  # [H, d_row // 2]

        for w_idx in range(width):
            pe[0:d_row:2, :, w_idx] = pe_y_sin.t()
            pe[1:d_row:2, :, w_idx] = pe_y_cos.t()

        # 2. Build column (horizontal) sinusoidal encodings
        position_x = torch.arange(0, width, dtype=torch.float).unsqueeze(
            1
        )  # [W, 1]
        div_term_x = torch.exp(
            torch.arange(0, d_col, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_col)
        )  # [d_col // 2]
        pe_x_sin = torch.sin(position_x * div_term_x)  # [W, d_col // 2]
        pe_x_cos = torch.cos(position_x * div_term_x)  # [W, d_col // 2]

        for h_idx in range(height):
            pe[d_row : d_model : 2, h_idx, :] = pe_x_sin.t()
            pe[d_row + 1 : d_model : 2, h_idx, :] = pe_x_cos.t()

        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: [1, Channel, H, W]

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        return x + self.pe[:, :, :h, :w].to(x.device)


class ResNetBlock(nn.Module):
    """Residual Bottleneck Block to promote healthy gradient flow."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CNNEncoder(nn.Module):
    """ResNet-based Encoder designed for 128x400 math images.

    Preserves structural height resolution (8 px) to avoid blending small
    symbols like decimals, commas, and inequalities.
    """

    def __init__(self, out_channels=512):
        super(CNNEncoder, self).__init__()

        self.conv_init = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_init = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Downsample Block 1: 128x400 -> 64x200
        self.layer1 = ResNetBlock(64, 128, stride=2)
        # Downsample Block 2: 64x200 -> 32x100
        self.layer2 = ResNetBlock(128, 256, stride=2)
        # Downsample Block 3: 32x100 -> 16x50
        self.layer3 = ResNetBlock(256, 256, stride=2)
        # Downsample Block 4: 16x50 -> 8x50
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                256,
                512,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer5 = ResNetBlock(512, out_channels)

        # Height is now preserved at 8 (vs. the previous 4)
        self.pos_encoder = SinusoidalPositionalEncoding2D(
            out_channels, height=8, width=50
        )

    def forward(self, x):
        x = self.relu(self.bn_init(self.conv_init(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.layer5(x)  # Shape: [B, out_channels, 8, 50]

        features = self.pos_encoder(features)

        batch_size, channels, h, w = features.size()
        flat_features = (
            features.view(batch_size, channels, h * w).permute(0, 2, 1)
        )  # Shape: [B, H*W, C]

        return flat_features, h, w


class CoverageAttention(nn.Module):
    """Additive Attention utilizing Layer-Normalized Cumulative Coverage to

    prevent scale-explosion and tanh input saturation.
    """

    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        attn_dim=256,
        cov_channels=128,
        kernel_size=5,
    ):
        super(CoverageAttention, self).__init__()
        self.W_encoder = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attn_dim, bias=False)

        # Spatial convolution over cumulative scans history
        self.conv_coverage = nn.Conv2d(
            1,
            cov_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.W_coverage = nn.Linear(cov_channels, attn_dim, bias=False)

        # Layer normalization prevents tanh saturation as coverage values aggregate over time
        self.norm_coverage = nn.LayerNorm(attn_dim)
        self.v_attn = nn.Linear(attn_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden, coverage):
        B, seq_len, _ = encoder_outputs.size()

        enc_proj = self.W_encoder(encoder_outputs)  # [B, seq_len, attn_dim]
        dec_proj = self.W_decoder(decoder_hidden).unsqueeze(
            1
        )  # [B, 1, attn_dim]

        # Spatial 2D extraction over parsed sequence history
        cov_feature = self.conv_coverage(coverage)  # [B, cov_channels, H, W]
        cov_feature = (
            cov_feature.view(B, cov_feature.size(1), -1).permute(0, 2, 1)
        )  # [B, seq_len, cov_channels]
        cov_proj = self.W_coverage(cov_feature)  # [B, seq_len, attn_dim]

        # Normalize cumulative magnitude bounds
        cov_proj = self.norm_coverage(cov_proj)

        # Scale-controlled alignment scoring
        scores = self.v_attn(
            torch.tanh(enc_proj + dec_proj + cov_proj)
        ).squeeze(
            2
        )  # [B, seq_len]
        attention_weights = F.softmax(scores, dim=1)  # [B, seq_len]

        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)

        return context_vector, attention_weights


class RNNDecoder(nn.Module):
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
        self.attention = CoverageAttention(
            encoder_dim, decoder_dim, attn_dim=attn_dim
        )

        self.lstm_cell = nn.LSTMCell(
            input_size=embed_dim + encoder_dim, hidden_size=decoder_dim
        )

        self.dropout = nn.Dropout(p=0.3)
        self.out_proj = nn.Linear(decoder_dim + encoder_dim, vocab_size)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def init_hidden_states(self, encoder_outputs):
        mean_encoder_features = encoder_outputs.mean(dim=1)
        h0 = torch.tanh(self.init_h(mean_encoder_features))
        c0 = torch.tanh(self.init_c(mean_encoder_features))
        return h0, c0

    def forward_step(
        self, prev_token, encoder_outputs, h, c, coverage, height, width
    ):
        embedded = self.embedding(prev_token)
        context, attn_weights = self.attention(encoder_outputs, h, coverage)

        # Update historical representation with the current attention maps
        coverage = coverage + attn_weights.view(
            attn_weights.size(0), 1, height, width
        )

        lstm_input = torch.cat([embedded, context], dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))

        output_feat = torch.cat([h, context], dim=1)
        output_feat = self.dropout(output_feat)
        logits = self.out_proj(output_feat)

        return logits, h, c, attn_weights, coverage


class HMER_Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        sos_idx,
        eos_idx,
        pad_idx=0,
        embed_dim=256,
        encoder_dim=512,
        decoder_dim=512,
        attn_dim=256,
    ):
        super(HMER_Model, self).__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.encoder = CNNEncoder(out_channels=encoder_dim)
        self.decoder = RNNDecoder(
            vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attn_dim=attn_dim,
        )

    def forward(
        self, images, targets=None, max_len=150, teacher_forcing_ratio=0.5
    ):
        batch_size = images.size(0)
        encoder_outputs, h_feat, w_feat = self.encoder(images)
        h, c = self.decoder.init_hidden_states(encoder_outputs)

        target_len = targets.size(1) if targets is not None else max_len
        all_predicted_logits = []
        all_attentions = []

        coverage = torch.zeros(
            batch_size, 1, h_feat, w_feat, device=images.device
        )
        curr_token = torch.full(
            (batch_size,), self.sos_idx, dtype=torch.long, device=images.device
        )

        for t in range(1, target_len):
            logits, h, c, attn_weights, coverage = self.decoder.forward_step(
                curr_token, encoder_outputs, h, c, coverage, h_feat, w_feat
            )

            all_predicted_logits.append(logits.unsqueeze(1))
            all_attentions.append(attn_weights.unsqueeze(1))

            if targets is not None and random.random() < teacher_forcing_ratio:
                gt_token = targets[:, t]
                is_pad = gt_token == self.pad_idx
                curr_token = torch.where(
                    is_pad, logits.argmax(dim=-1), gt_token
                )
            else:
                curr_token = logits.argmax(dim=-1)

        all_predicted_logits = torch.cat(all_predicted_logits, dim=1)
        all_attentions = torch.cat(all_attentions, dim=1)

        return all_predicted_logits, all_attentions

    def beam_search(self, image, beam_size=3, max_len=150):
        """Standard Beam Search Decoder for high-accuracy inference testing."""
        self.eval()
        with torch.no_grad():
            batch_size = image.size(0)
            assert (
                batch_size == 1
            ), "Beam search is optimized for inference with batch size 1"

            encoder_outputs, h_feat, w_feat = self.encoder(image)
            h, c = self.decoder.init_hidden_states(encoder_outputs)

            coverage = torch.zeros(
                batch_size, 1, h_feat, w_feat, device=image.device
            )

            # (cumulative_log_prob, sequence, hidden_state_h, hidden_state_c, coverage_mask)
            beams = [(0.0, [self.sos_idx], h, c, coverage)]

            for t in range(1, max_len):
                candidates = []
                for log_prob, seq, h_state, c_state, cov in beams:
                    if seq[-1] == self.eos_idx:
                        candidates.append((log_prob, seq, h_state, c_state, cov))
                        continue

                    curr_token = torch.tensor(
                        [seq[-1]], dtype=torch.long, device=image.device
                    )
                    (
                        logits,
                        next_h,
                        next_c,
                        _,
                        next_cov,
                    ) = self.decoder.forward_step(
                        curr_token,
                        encoder_outputs,
                        h_state,
                        c_state,
                        cov,
                        h_feat,
                        w_feat,
                    )

                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                    top_probs, top_indices = log_probs.topk(beam_size)

                    for p, idx in zip(top_probs, top_indices):
                        candidates.append(
                            (
                                log_prob + p.item(),
                                seq + [idx.item()],
                                next_h,
                                next_c,
                                next_cov,
                            )
                        )

                # Keep top sorted candidate matching beams
                beams = sorted(candidates, key=lambda x: x[0], reverse=True)[
                    :beam_size
                ]

                if all(b[1][-1] == self.eos_idx for b in beams):
                    break

            best_beam = beams[0][1]
            return best_beam[1:-1]  # Strip SOS and EOS indexes