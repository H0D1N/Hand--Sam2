import copy

from .sam2_modified import SAM2Modified


class SAM2DualHandMemory(SAM2Modified):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.left_memory_attention = copy.deepcopy(self.memory_attention)
        self.right_memory_attention = copy.deepcopy(self.memory_attention)

        self.left_memory_encoder = copy.deepcopy(self.memory_encoder)
        self.right_memory_encoder = copy.deepcopy(self.memory_encoder)

        del self.memory_attention
        del self.memory_encoder