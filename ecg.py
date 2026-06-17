import torch

# ── Lead index constants ──────────────────────────────────────────────────────
_I   = 0
_II  = 1
_III = 2
_AVR = 3
_AVL = 4
_AVF = 5
_V1  = 6
_V2  = 7
_V3  = 8
_V4  = 9
_V5  = 10
_V6  = 11


class ECGCorruptions:
    """One class, six static corruption methods. 
    
    Compatible with 2D tensors, 3D batches, and 4D channel-packed tensors.
    """

    # ── 1. RA / LA swap ──────────────────────────────────────────────────────
    @staticmethod
    def ra_la_swap(ecg: torch.Tensor) -> torch.Tensor:
        """RA ↔ LA electrode reversal."""
        out = ecg.clone()
        out[..., _I, :]   = -ecg[..., _I, :]
        out[..., _II, :]  =  ecg[..., _III, :]
        out[..., _III, :] =  ecg[..., _II, :]
        out[..., _AVR, :] = -ecg[..., _AVL, :]
        out[..., _AVL, :] = -ecg[..., _AVR, :]
        out[..., _AVF, :] =  ecg[..., _AVF, :]   # unchanged
        return out

    # ── 2. RA / LL swap ──────────────────────────────────────────────────────
    @staticmethod
    def ra_ll_swap(ecg: torch.Tensor) -> torch.Tensor:
        """RA ↔ LL electrode reversal."""
        out = ecg.clone()
        out[..., _I, :]   = -ecg[..., _III, :]
        out[..., _II, :]  = -ecg[..., _II, :]
        out[..., _III, :] = -ecg[..., _I, :]
        out[..., _AVR, :] =  ecg[..., _AVF, :]
        out[..., _AVL, :] =  ecg[..., _AVL, :]   # unchanged
        out[..., _AVF, :] =  ecg[..., _AVR, :]
        return out

    # ── 3. LA / LL swap ──────────────────────────────────────────────────────
    @staticmethod
    def la_ll_swap(ecg: torch.Tensor) -> torch.Tensor:
        """LA ↔ LL electrode reversal."""
        out = ecg.clone()
        out[..., _I, :]   =  ecg[..., _II, :]
        out[..., _II, :]  =  ecg[..., _I, :]
        out[..., _III, :] = -ecg[..., _III, :]
        out[..., _AVR, :] =  ecg[..., _AVR, :]   # unchanged
        out[..., _AVL, :] =  ecg[..., _AVF, :]
        out[..., _AVF, :] =  ecg[..., _AVL, :]
        return out

    # ── 4. V1 / V2 swap ──────────────────────────────────────────────────────
    @staticmethod
    def v1_v2_swap(ecg: torch.Tensor) -> torch.Tensor:
        """V1 ↔ V2 chest lead swap."""
        out = ecg.clone()
        out[..., _V1, :] = ecg[..., _V2, :]
        out[..., _V2, :] = ecg[..., _V1, :]
        return out

    # ── 5. V2 / V3 swap ──────────────────────────────────────────────────────
    @staticmethod
    def v2_v3_swap(ecg: torch.Tensor) -> torch.Tensor:
        """V2 ↔ V3 chest lead swap."""
        out = ecg.clone()
        out[..., _V2, :] = ecg[..., _V3, :]
        out[..., _V3, :] = ecg[..., _V2, :]
        return out

    # ── 6. Single-lead polarity inversion ────────────────────────────────────
    @staticmethod
    def single_lead_polarity_inversion(
        ecg: torch.Tensor,
        lead_idx: int = 0,
    ) -> torch.Tensor:
        """Multiply one lead by -1 across all dimensions."""
        # Check against the second-to-last dimension (the leads axis)
        if lead_idx < 0 or lead_idx >= ecg.shape[-2]:
            raise IndexError(
                f"lead_idx {lead_idx} out of range for ECG with "
                f"{ecg.shape[-2]} leads (valid: 0-{ecg.shape[-2]-1})."
            )
        out = ecg.clone()
        out[..., lead_idx, :] = -ecg[..., lead_idx, :]
        return out