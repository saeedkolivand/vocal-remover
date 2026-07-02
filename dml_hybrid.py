"""Run Mel-Band RoFormer on a DirectML GPU (any Windows GPU — NVIDIA/AMD/Intel).

DirectML fatally aborts on complex tensors ("Invalid or unsupported data type
ComplexFloat"), which is why naively enabling it crashes at the model's STFT (step 0).
But RoFormer's heavy part — the band-split + transformer stack — is real-valued (real
rotary embeddings, real attention), and on a GPU it runs ~1-2 orders of magnitude faster
than CPU. So we split the forward: the complex STFT / iSTFT / mask application stay on
CPU; the real transformer runs on the GPU. Two small tensor transfers per chunk; the
transformer speedup dwarfs them.

Inference-only. app.py calls enable() only when the active device is DirectML.
audio-separator already handles device placement (use_directml=True); this only swaps
the model's forward so the complex ops never touch DirectML.
"""
import torch
from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as _mbr
import rotary_embedding_torch.rotary_embedding_torch as _rot

# Reuse the model module's own einops helpers so behaviour matches the stock forward.
_rearrange, _repeat = _mbr.rearrange, _mbr.repeat
_pack, _unpack = _mbr.pack, _mbr.unpack
_pack_one, _unpack_one = _mbr.pack_one, _mbr.unpack_one

_CPU = torch.device("cpu")


def _hybrid_forward(self, raw_audio, target=None, return_loss_breakdown=False):
    # Mirrors MelBandRoformer.forward's inference path, but pins the complex ops to CPU
    # and the transformer to the model's (DirectML) device.
    assert target is None, "dml_hybrid is inference-only"
    gpu = next(self.parameters()).device  # DirectML; falls back to CPU if unmoved

    # ---- CPU: STFT (complex) ----
    raw_audio = raw_audio.to(_CPU)
    if raw_audio.ndim == 2:
        raw_audio = _rearrange(raw_audio, "b t -> b 1 t")
    batch, channels, raw_audio_length = raw_audio.shape
    istft_length = raw_audio_length if self.match_input_audio_length else None
    assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
        "stereo flag must match the input channel count"
    raw_audio, packed_shape = _pack_one(raw_audio, "* t")
    stft_window = self.stft_window_fn().to(_CPU)
    stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
    stft_repr = torch.view_as_real(stft_repr)
    stft_repr = _unpack_one(stft_repr, packed_shape, "* f t c")
    stft_repr = _rearrange(stft_repr, "b s f t c -> b (f s) t c")
    batch_arange = torch.arange(batch, device=_CPU)[..., None]
    x = stft_repr[batch_arange, self.freq_indices.to(_CPU)]
    x = _rearrange(x, "b f t c -> b t (f c)")

    # ---- GPU: band split + transformer + mask estimation (all real-valued) ----
    x = x.to(gpu)
    x = self.band_split(x)
    for time_transformer, freq_transformer in self.layers:
        x = _rearrange(x, "b t f d -> b f t d")
        x, ps = _pack([x], "* t d")
        x = time_transformer(x)
        (x,) = _unpack(x, ps, "* t d")
        x = _rearrange(x, "b f t d -> b t f d")
        x, ps = _pack([x], "* f d")
        x = freq_transformer(x)
        (x,) = _unpack(x, ps, "* f d")
    masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
    masks = _rearrange(masks, "b n t (f c) -> b n f t c", c=2)
    masks = masks.to(_CPU)

    # ---- CPU: complex mask application + iSTFT ----
    stft_repr = _rearrange(stft_repr, "b f t c -> b 1 f t c")
    stft_repr = torch.view_as_complex(stft_repr)
    masks = torch.view_as_complex(masks)
    masks = masks.type(stft_repr.dtype)
    scatter_indices = _repeat(self.freq_indices.to(_CPU), "f -> b n f t",
                              b=batch, n=self.num_stems, t=stft_repr.shape[-1])
    stft_repr_expanded_stems = _repeat(stft_repr, "b 1 ... -> b n ...", n=self.num_stems)
    masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)
    denom = _repeat(self.num_bands_per_freq.to(_CPU), "f -> (f r) 1", r=channels)
    masks_averaged = masks_summed / denom.clamp(min=1e-8)
    stft_repr = stft_repr * masks_averaged
    stft_repr = _rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
    recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window,
                              return_complex=False, length=istft_length)
    recon_audio = _rearrange(recon_audio, "(b n s) t -> b n s t",
                             b=batch, s=self.audio_channels, n=self.num_stems)
    if self.num_stems == 1:
        recon_audio = _rearrange(recon_audio, "b 1 s t -> b s t")
    return recon_audio


def _safe_apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    # Faithful copy of rotary_embedding_torch.apply_rotary_emb, except it drops zero-width
    # slices before torch.cat. With start_index=0 and full-width rotation, t_left/t_right
    # are empty, and DirectML aborts on cat with an empty tensor ("The parameter is
    # incorrect") where CPU/CUDA handle it fine. Filtering empties is a no-op elsewhere.
    dtype = t.dtype
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], \
        f"feature dimension {t.shape[-1]} too small to rotate {rot_dim} positions"
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    t_transformed = (t_middle * freqs.cos() * scale) + (_rot.rotate_half(t_middle) * freqs.sin() * scale)
    parts = [p for p in (t_left, t_transformed, t_right) if p.shape[-1] > 0]
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    return out.type(dtype)


_original_forward = None
_original_rotary = None


def enable():
    """Route Mel-Band RoFormer through the CPU/GPU hybrid and DirectML-safe rotary. Idempotent."""
    global _original_forward, _original_rotary
    if _original_forward is None:
        _original_forward = _mbr.MelBandRoformer.forward
        _mbr.MelBandRoformer.forward = _hybrid_forward
    if _original_rotary is None:
        _original_rotary = _rot.apply_rotary_emb
        _rot.apply_rotary_emb = _safe_apply_rotary_emb


def disable():
    global _original_forward, _original_rotary
    if _original_forward is not None:
        _mbr.MelBandRoformer.forward = _original_forward
        _original_forward = None
    if _original_rotary is not None:
        _rot.apply_rotary_emb = _original_rotary
        _original_rotary = None
