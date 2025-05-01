import torch
import torch.nn as nn
import torch.nn.functional as F

from hflayers import Hopfield


class RNNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        head_dim = d_model // n_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable decay (γ) per head
        self.gamma = nn.Parameter(torch.ones(n_heads, head_dim))

        # Gated MLP (SwiGLU-style)
        self.gate = nn.Linear(d_model, 2 * d_model, bias=False)
        self.down = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("init_state", torch.zeros(n_heads, head_dim, head_dim))

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [d_model] tensor for the current frame.
        Returns: [d_model] updated tensor.
        """
        # --- 1) WKV-style attention update ---
        x_norm = self.ln1(x)

        # Project and split into heads: [heads, head_dim]
        q = self.q_proj(x_norm).view(-1, self.gamma.size(1))
        k = self.k_proj(x_norm).view(-1, self.gamma.size(1))
        v = self.v_proj(x_norm).view(-1, self.gamma.size(1))

        # Update S in-place: S ← γ * S + k ⊗ vᵀ
        #   γ: [heads, head_dim] → [heads, head_dim, 1]
        γ = self.gamma.unsqueeze(-1)
        #   k: [heads, head_dim] → [heads, head_dim, 1]
        k_h = k.unsqueeze(-1)
        #   v: [heads, head_dim] → [heads, 1, head_dim]
        v_h = v.unsqueeze(1)
        state = state + γ  # decay
        state = state + (k_h @ v_h)  # outer product

        # Compute attention‐like output: o = S ⋅ q
        #   q: [heads, head_dim] → [heads, head_dim, 1]
        q_h = q.unsqueeze(-1)
        o = (state @ q_h).squeeze(-1)  # [heads, head_dim]

        attn_out = o.view(-1)  # back to [d_model]

        # --- 2) Gated MLP ---
        x2 = self.ln2(x + attn_out)
        gate, mlp_in = self.gate(x2).chunk(2, dim=-1)
        mlp_out = F.silu(mlp_in) * torch.sigmoid(gate)
        mlp_out = self.down(mlp_out)

        # Residual sum
        return x + attn_out + mlp_out, state


class GestureToMIDIConfig:
    def __init__(
        self,
        latent_dim: int,
        video_channels: int,
        n_video_heads: int,
        audio_channels: int,
        n_audio_heads: int,
        notes: list[int],
        velocities: list[int],
    ):
        self.latent_dim = latent_dim
        self.video_channels = video_channels
        self.n_video_heads = n_video_heads
        self.audio_channels = audio_channels
        self.n_audio_heads = n_audio_heads
        self.notes = notes
        self.velocities = velocities


class GestureToMIDI(nn.Module):
    def __init__(self, config: GestureToMIDIConfig):
        self.notes = config.notes
        self.velocities = config.velocities
        self.video_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=config.video_channels,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
            ),
            nn.ReLU(),
            nn.Conv3d(
                config.video_channels,
                config.video_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.video_proj = nn.Linear(config.video_channels * 2, config.latent_dim)

        self.video_core = RNNBlock(
            d_model=config.latent_dim, n_heads=config.n_video_heads
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=config.audio_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=config.audio_channels,
                out_channels=config.audio_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.audio_proj = nn.Linear(config.audio_channels * 2, config.latent_dim)

        self.audio_core = RNNBlock(
            d_model=config.latent_dim, n_heads=config.n_audio_heads
        )

        self.video_predictor = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

        self.hopfield = Hopfield(input_size=config.latent_dim)

        self.note_head = nn.Linear(config.latent_dim, len(config.notes))

        self.velocity_head = nn.Linear(config.latent_dim, len(config.velocities))

    def forward(
        self,
        audio_input: torch.Tensor,
        video_input: torch.Tensor,
        audio_state: torch.Tensor,
        video_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_features = self.audio_proj(self.audio_encoder(audio_input))
        audio_latent, audio_state = self.audio_core(audio_features, audio_state)

        video_features = self.video_proj(self.video_encoder(video_input))
        video_latent, video_state = self.video_core(video_features, video_state)

        video_pred = self.video_predictor(audio_latent)

        latent = video_latent + self.hopfield(video_latent)

        note_logits: torch.Tensor = self.note_head(latent)
        velo_logits: torch.Tensor = self.note_head(latent)

        return note_logits, velo_logits, audio_state, video_state
