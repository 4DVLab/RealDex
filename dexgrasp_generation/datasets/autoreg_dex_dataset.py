from datasets.dex_dataset import DFCDataset

class AutoRegDexDataset(DFCDataset):
    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)
        
    def __getitem__(self, item):
        return super().__getitem__(item)