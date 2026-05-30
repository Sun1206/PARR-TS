import torch
import torch.nn as nn
import torch.nn.functional as F


class PARRPreprocessor(nn.Module):
    """Predictability-aware input processor for frozen forecasting backbones.

    The module computes patch-level descriptors used by PARR-style experiments:
    spectral entropy, period drift, local roughness, and channel profile drift.
    It can also softly replace low-predictability regions with a stable window
    baseline for intervention-style ablations.
    """

    def __init__(self, args):
        super().__init__()
        self.patch_len = int(getattr(args, "parr_patch_len", 16))
        self.alpha_s = float(getattr(args, "parr_alpha_s", 1.0))
        self.alpha_d = float(getattr(args, "parr_alpha_d", 0.5))
        self.alpha_e = float(getattr(args, "parr_alpha_e", 1.0))
        self.alpha_g = float(getattr(args, "parr_alpha_g", 0.5))
        self.min_keep = float(getattr(args, "parr_min_keep", 0.35))
        self.dropout = float(getattr(args, "parr_dropout", 0.0))
        self.replace_strength = float(getattr(args, "parr_replace_strength", 1.0))
        self.eps = 1e-6

    def forward(self, x, training=False):
        """Return processed input, sample weights, and detached diagnostics.

        Args:
            x: Tensor with shape [B, T, C].
            training: Whether to enable adaptive dropout.
        """
        if self.patch_len <= 1 or x.size(1) < self.patch_len:
            weight = x.new_ones(x.size(0))
            return x, weight, {"score": weight.detach()}

        score = self.score(x).detach()
        token_score = self._expand_patch_score(score, x.size(1))
        baseline = x.mean(dim=1, keepdim=True)
        gate = token_score.unsqueeze(-1)

        if self.replace_strength > 0:
            mix = 1.0 - self.replace_strength * (1.0 - gate)
            mix = mix.clamp(0.0, 1.0)
            x = mix * x + (1.0 - mix) * baseline

        if training and self.dropout > 0:
            keep_prob = (self.min_keep + (1.0 - self.min_keep) * token_score).clamp(
                self.min_keep, 1.0
            )
            keep_prob = (1.0 - self.dropout) + self.dropout * keep_prob
            mask = torch.bernoulli(keep_prob).unsqueeze(-1)
            x = mask * x + (1.0 - mask) * baseline

        weight = score.mean(dim=1).clamp(self.min_keep, 1.0)
        stats = {
            "score": weight.detach(),
            "patch_score": score.detach(),
        }
        return x, weight, stats

    def score(self, x):
        patches, _ = self._patchify(x)
        if patches is None:
            return x.new_ones(x.size(0), 1)

        spectral_entropy = self._spectral_entropy(patches)
        period_drift = self._period_drift(patches)
        smooth_residual = self._smooth_residual(patches)
        channel_drift = self._channel_profile_drift(patches)

        raw = (
            self.alpha_s * spectral_entropy
            + self.alpha_d * period_drift
            + self.alpha_e * smooth_residual
            + self.alpha_g * channel_drift
        )
        return torch.exp(-raw).clamp(self.min_keep, 1.0)

    def components(self, x):
        """Return the four patch-level PARR components as a dictionary."""
        patches, _ = self._patchify(x)
        if patches is None:
            zeros = x.new_zeros(x.size(0), 1)
            return {
                "spectral_entropy": zeros,
                "period_drift": zeros,
                "smooth_residual": zeros,
                "channel_profile_drift": zeros,
            }
        return {
            "spectral_entropy": self._spectral_entropy(patches),
            "period_drift": self._period_drift(patches),
            "smooth_residual": self._smooth_residual(patches),
            "channel_profile_drift": self._channel_profile_drift(patches),
        }

    def _patchify(self, x):
        bsz, length, _ = x.shape
        patch_len = min(self.patch_len, length)
        num_patches = length // patch_len
        valid_len = num_patches * patch_len
        if num_patches < 1:
            return None, 0
        patches = x[:, :valid_len, :].unfold(dimension=1, size=patch_len, step=patch_len)
        return patches.permute(0, 1, 3, 2).contiguous(), valid_len

    def _expand_patch_score(self, score, length):
        token_score = score.repeat_interleave(self.patch_len, dim=1)
        if token_score.size(1) < length:
            pad = token_score[:, -1:].expand(-1, length - token_score.size(1))
            token_score = torch.cat([token_score, pad], dim=1)
        return token_score[:, :length]

    def _spectral_entropy(self, patches):
        freq = torch.fft.rfft(patches, dim=2)
        energy = freq.abs().pow(2).mean(dim=-1)
        prob = energy / (energy.sum(dim=-1, keepdim=True) + self.eps)
        entropy = -(prob * (prob + self.eps).log()).sum(dim=-1)
        norm = torch.log(torch.tensor(prob.size(-1), device=patches.device, dtype=patches.dtype))
        return entropy / (norm + self.eps)

    def _period_drift(self, patches):
        freq = torch.fft.rfft(patches, dim=2).abs().pow(2).mean(dim=-1)
        if freq.size(-1) <= 2:
            return patches.new_zeros(patches.size(0), patches.size(1))
        non_dc = freq[:, :, 1:]
        dominant = non_dc.argmax(dim=-1).float() + 1.0
        ref = dominant.mean(dim=1, keepdim=True)
        return ((dominant - ref).abs() / (ref.abs() + self.eps)).clamp(0.0, 1.0)

    def _smooth_residual(self, patches):
        bsz, num_patches, patch_len, channels = patches.shape
        if patch_len < 3:
            return patches.new_zeros(bsz, num_patches)
        series = patches.permute(0, 1, 3, 2).reshape(bsz * num_patches * channels, 1, patch_len)
        smooth = F.avg_pool1d(series, kernel_size=3, stride=1, padding=1)
        residual = (series - smooth).pow(2).mean(dim=(1, 2))
        scale = series.pow(2).mean(dim=(1, 2)) + self.eps
        residual = (residual / scale).reshape(bsz, num_patches, channels).mean(dim=-1)
        return residual.clamp(0.0, 1.0)

    def _channel_profile_drift(self, patches):
        patch_profile = patches.mean(dim=2)
        ref_profile = patch_profile.mean(dim=1, keepdim=True)
        patch_norm = F.normalize(patch_profile, dim=-1)
        ref_norm = F.normalize(ref_profile, dim=-1)
        drift = 1.0 - (patch_norm * ref_norm).sum(dim=-1)
        return drift.clamp(0.0, 1.0)


# Backward-compatible aliases for older notes/scripts with inconsistent names.
PARRPreprocesson = PARRPreprocessor
PARRPreprocessing = PARRPreprocessor
