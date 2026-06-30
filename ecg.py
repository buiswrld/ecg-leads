import torch

class ECGCorruptions:
    """
    Dynamically maps physical ECG corruption swaps based on the active lead subset.
    """
    LEAD_MAP = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    @classmethod
    def _get_indices(cls, leads_in_tensor, lead_a, lead_b):
        """Helper to find local positions of leads in the current tensor slice."""
        try:
            idx_a = leads_in_tensor.index(lead_a)
            idx_b = leads_in_tensor.index(lead_b)
            return idx_a, idx_b
        except ValueError:
            return None, None

    @classmethod
    def ra_la_reversal(cls, ecg: torch.Tensor, leads: list = None) -> torch.Tensor:
        """Swaps RA and LA electrodes safely."""
        current_leads = leads if leads is not None else cls.LEAD_MAP
        corrupted = ecg.clone()
        
        # Lead I inversion
        idx_1, _ = cls._get_indices(current_leads, "I", "I")
        if idx_1 is not None and idx_1 < corrupted.shape[0]: 
            corrupted[idx_1] = -ecg[idx_1]
        
        # Lead II and III swap
        idx_2, idx_3 = cls._get_indices(current_leads, "II", "III")
        if idx_2 is not None and idx_3 is not None and idx_2 < corrupted.shape[0] and idx_3 < corrupted.shape[0]:
            corrupted[idx_2], corrupted[idx_3] = ecg[idx_3], ecg[idx_2]
            
        # aVR and aVL swap
        idx_avr, idx_avl = cls._get_indices(current_leads, "aVR", "aVL")
        if idx_avr is not None and idx_avl is not None and idx_avr < corrupted.shape[0] and idx_avl < corrupted.shape[0]:
            corrupted[idx_avr], corrupted[idx_avl] = ecg[idx_avl], ecg[idx_avr]
        return corrupted

    @classmethod
    def ra_ll_reversal(cls, ecg: torch.Tensor, leads: list = None) -> torch.Tensor:
        """Swaps RA and LL electrodes safely."""
        current_leads = leads if leads is not None else cls.LEAD_MAP
        corrupted = ecg.clone()
        
        idx_2, _ = cls._get_indices(current_leads, "II", "II")
        if idx_2 is not None and idx_2 < corrupted.shape[0]: 
            corrupted[idx_2] = -ecg[idx_2]
        
        idx_1, idx_3 = cls._get_indices(current_leads, "I", "III")
        if idx_1 is not None and idx_3 is not None and idx_1 < corrupted.shape[0] and idx_3 < corrupted.shape[0]:
            corrupted[idx_1], corrupted[idx_3] = -ecg[idx_3], -ecg[idx_1]
            
        idx_avr, idx_avf = cls._get_indices(current_leads, "aVR", "aVF")
        if idx_avr is not None and idx_avf is not None and idx_avr < corrupted.shape[0] and idx_avf < corrupted.shape[0]:
            corrupted[idx_avr], corrupted[idx_avf] = ecg[idx_avf], ecg[idx_avr]
        return corrupted

    @classmethod
    def la_ll_reversal(cls, ecg: torch.Tensor, leads: list = None) -> torch.Tensor:
        """Swaps LA and LL electrodes safely."""
        current_leads = leads if leads is not None else cls.LEAD_MAP
        corrupted = ecg.clone()
        
        idx_3, _ = cls._get_indices(current_leads, "III", "III")
        if idx_3 is not None and idx_3 < corrupted.shape[0]: 
            corrupted[idx_3] = -ecg[idx_3]
        
        idx_1, idx_2 = cls._get_indices(current_leads, "I", "II")
        if idx_1 is not None and idx_2 is not None and idx_1 < corrupted.shape[0] and idx_2 < corrupted.shape[0]:
            corrupted[idx_1], corrupted[idx_2] = ecg[idx_2], ecg[idx_1]
            
        idx_avl, idx_avf = cls._get_indices(current_leads, "aVL", "aVF")
        if idx_avl is not None and idx_avf is not None and idx_avl < corrupted.shape[0] and idx_avf < corrupted.shape[0]:
            corrupted[idx_avl], corrupted[idx_avf] = ecg[idx_avf], ecg[idx_avl]
        return corrupted

    @classmethod
    def v1_v2_swap(cls, ecg: torch.Tensor, leads: list = None) -> torch.Tensor:
        """Swaps V1 and V2 safely."""
        current_leads = leads if leads is not None else cls.LEAD_MAP
        corrupted = ecg.clone()
        idx_v1, idx_v2 = cls._get_indices(current_leads, "V1", "V2")
        if idx_v1 is not None and idx_v2 is not None and idx_v1 < corrupted.shape[0] and idx_v2 < corrupted.shape[0]:
            corrupted[idx_v1], corrupted[idx_v2] = ecg[idx_v2], ecg[idx_v1]
        return corrupted

    @classmethod
    def v2_v3_swap(cls, ecg: torch.Tensor, leads: list = None) -> torch.Tensor:
        """Swaps V2 and V3 safely."""
        current_leads = leads if leads is not None else cls.LEAD_MAP
        corrupted = ecg.clone()
        idx_v2, idx_v3 = cls._get_indices(current_leads, "V2", "V3")
        if idx_v2 is not None and idx_v3 is not None and idx_v2 < corrupted.shape[0] and idx_v3 < corrupted.shape[0]:
            corrupted[idx_v2], corrupted[idx_v3] = ecg[idx_v3], ecg[idx_v2]
        return corrupted

    @classmethod
    def single_lead_polarity_inversion(cls, ecg: torch.Tensor, lead_idx: int = 0, leads: list = None) -> torch.Tensor:
        """Inverts the polarity of a single specified lead safely."""
        corrupted = ecg.clone()
        if lead_idx < corrupted.shape[0]:
            corrupted[lead_idx] = -ecg[lead_idx]
        return corrupted
