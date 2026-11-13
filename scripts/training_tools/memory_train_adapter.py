"""Train only selected modules using the official SAM2Train implementation."""
from training.model.sam2 import SAM2Train


class MemoryOnlySAM2Train(SAM2Train):
    def __init__(self,**kwargs):
        kwargs['freeze_image_encoder']=True
        super().__init__(**kwargs)
        prefixes=('memory_attention.','memory_encoder.','sam_mask_decoder.')
        for name,param in self.named_parameters():
            param.requires_grad_(name.startswith(prefixes))
        # Object-score head is contained in sam_mask_decoder on SAM2.1.
        if not any(p.requires_grad for p in self.parameters()):
            raise RuntimeError('No matching trainable modules; check official SAM version')
