"""
Acquisition-error simulation.

Corruptions are applied to the full 12-lead tensor using fixed global lead
indices before lead-subset extraction.
"""

import torch

# Canonical PTB-XL lead order. Index positions are fixed and global.
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = {name: i for i, name in enumerate(LEAD_ORDER)}

I, II, III = LEAD_IDX["I"], LEAD_IDX["II"], LEAD_IDX["III"]
AVR, AVL, AVF = LEAD_IDX["aVR"], LEAD_IDX["aVL"], LEAD_IDX["aVF"]
V1, V2, V3 = LEAD_IDX["V1"], LEAD_IDX["V2"], LEAD_IDX["V3"]

# Leads whose signal is altered by each corruption. Used to restrict the
# candidate pool in the corruption-specific selection of Experiment 5.
CORRUPTION_AFFECTED_LEADS = {
    "ra_la_reversal": ["I", "II", "III", "aVR", "aVL"],
    "ra_ll_reversal": ["I", "II", "III", "aVR", "aVF"],
    "la_ll_reversal": ["I", "II", "III", "aVL", "aVF"],
    "v1_v2_swap":     ["V1", "V2"],
    "v2_v3_swap":     ["V2", "V3"],
    "single_lead_polarity_inversion": list(LEAD_ORDER),
}

SWAP_REVERSAL_CORRUPTIONS = [
    "ra_la_reversal", "ra_ll_reversal", "la_ll_reversal",
    "v1_v2_swap", "v2_v3_swap",
]


class ECGCorruptions:
    """
    Every method takes a full 12-lead tensor of shape (12, T) and returns a
    corrupted copy of the same shape. Callers must NOT pre-slice to a subset.
    """

    @staticmethod
    def _check(ecg):
        if ecg.shape[0] != 12:
            raise ValueError(
                f"Corruptions require the full 12-lead signal, got shape {tuple(ecg.shape)}. "
                "Apply corruption BEFORE extracting the lead subset."
            )

    @classmethod
    def ra_la_reversal(cls, ecg, **kwargs):
        """RA/LA: I -> -I; II <-> III; aVR <-> aVL; aVF unchanged."""
        cls._check(ecg)
        out = ecg.clone()
        out[I] = -ecg[I]
        out[II], out[III] = ecg[III], ecg[II]
        out[AVR], out[AVL] = ecg[AVL], ecg[AVR]
        return out

    @classmethod
    def ra_ll_reversal(cls, ecg, **kwargs):
        """RA/LL: II -> -II; I -> -III; III -> -I; aVR <-> aVF; aVL unchanged."""
        cls._check(ecg)
        out = ecg.clone()
        out[II] = -ecg[II]
        out[I], out[III] = -ecg[III], -ecg[I]
        out[AVR], out[AVF] = ecg[AVF], ecg[AVR]
        return out

    @classmethod
    def la_ll_reversal(cls, ecg, **kwargs):
        """LA/LL: III -> -III; I <-> II; aVL <-> aVF; aVR unchanged."""
        cls._check(ecg)
        out = ecg.clone()
        out[III] = -ecg[III]
        out[I], out[II] = ecg[II], ecg[I]
        out[AVL], out[AVF] = ecg[AVF], ecg[AVL]
        return out

    @classmethod
    def v1_v2_swap(cls, ecg, **kwargs):
        cls._check(ecg)
        out = ecg.clone()
        out[V1], out[V2] = ecg[V2], ecg[V1]
        return out

    @classmethod
    def v2_v3_swap(cls, ecg, **kwargs):
        cls._check(ecg)
        out = ecg.clone()
        out[V2], out[V3] = ecg[V3], ecg[V2]
        return out

    @classmethod
    def single_lead_polarity_inversion(cls, ecg, corruption_lead=None, **kwargs):
        """
        Invert one lead, identified by NAME (global), not by position within a
        subset. `corruption_lead` may be a lead name or a global index.
        """
        cls._check(ecg)
        if corruption_lead is None:
            raise ValueError("single_lead_polarity_inversion requires corruption_lead")
        idx = LEAD_IDX[corruption_lead] if isinstance(corruption_lead, str) else int(corruption_lead)
        out = ecg.clone()
        out[idx] = -ecg[idx]
        return out


CORRUPTION_FNS = {
    "ra_la_reversal": ECGCorruptions.ra_la_reversal,
    "ra_ll_reversal": ECGCorruptions.ra_ll_reversal,
    "la_ll_reversal": ECGCorruptions.la_ll_reversal,
    "v1_v2_swap": ECGCorruptions.v1_v2_swap,
    "v2_v3_swap": ECGCorruptions.v2_v3_swap,
    "single_lead_polarity_inversion": ECGCorruptions.single_lead_polarity_inversion,
}


def apply_corruption(ecg, corruption, corruption_lead=None):
    """Apply a named corruption to a full 12-lead tensor."""
    if corruption is None:
        return ecg
    if corruption not in CORRUPTION_FNS:
        raise KeyError(f"Unknown corruption: {corruption}")
    return CORRUPTION_FNS[corruption](ecg, corruption_lead=corruption_lead)


def corruption_affects_subset(corruption, subset_leads, corruption_lead=None):
    """
    True if the corruption alters at least one lead retained in `subset_leads`.

    Note this is about whether the RETAINED signal changes, which is the
    meaningful notion now that corruption precedes extraction. A subset holding
    only V1 IS affected by v1_v2_swap (V1 now carries V2's signal), which the
    previous both-leads-present check wrongly treated as inapplicable.
    """
    if corruption == "single_lead_polarity_inversion":
        if corruption_lead is None:
            return False
        name = corruption_lead if isinstance(corruption_lead, str) else LEAD_ORDER[int(corruption_lead)]
        return name in subset_leads
    affected = set(CORRUPTION_AFFECTED_LEADS[corruption])
    return len(affected.intersection(set(subset_leads))) > 0
