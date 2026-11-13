"""Replay the fixed-shape SAM image encoder without repeated Python launches.

Image encoding and a fixed-shape attention calculation are captured. SAMURAI
memory selection, Kalman, mask selection and identity/loss decisions are unchanged.
"""
import torch


class CudaBackbone:
    @torch.inference_mode()
    def __init__(self, model):
        size = model.image_size
        self.image = torch.zeros((1, 3, size, size), device=model.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            for _ in range(3):
                model.forward_image(self.image)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            self.output = model.forward_image(self.image)
        torch.cuda.synchronize()

    def __call__(self, image):
        self.image.copy_(image)
        self.graph.replay()
        return self.output


class CudaMemoryAttention(torch.nn.Module):
    """Capture only the common full-memory shape; all other shapes use eager.

    Memory contents and frame selection are supplied anew on every call. This
    does not pad, truncate, reorder or otherwise change SAMURAI memory.
    """
    @torch.inference_mode()
    def __init__(self, module, model):
        super().__init__()
        self.inner = module
        self.eval()
        self.hits = 0
        tokens = (model.image_size // model.backbone_stride) ** 2
        self.ptr_tokens = model.max_obj_ptrs_in_encoder * (model.hidden_dim // model.mem_dim)
        memory_tokens = model.num_maskmem * tokens + self.ptr_tokens
        self.inputs = dict(
            curr=torch.zeros((tokens, 1, model.hidden_dim), device=model.device),
            curr_pos=torch.zeros((tokens, 1, model.hidden_dim), device=model.device),
            memory=torch.zeros((memory_tokens, 1, model.mem_dim), device=model.device),
            memory_pos=torch.zeros((memory_tokens, 1, model.mem_dim), device=model.device))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            for _ in range(3):
                module(**self.inputs, num_obj_ptr_tokens=self.ptr_tokens)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            self.output = module(**self.inputs, num_obj_ptr_tokens=self.ptr_tokens)
        torch.cuda.synchronize()

    def forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
        values = dict(curr=curr[0] if isinstance(curr, list) else curr,
                      curr_pos=curr_pos[0] if isinstance(curr_pos, list) else curr_pos,
                      memory=memory, memory_pos=memory_pos)
        if self.hits == 0:
            self.last_layout = {key: (tuple(value.shape), str(value.dtype)) if value is not None else None
                                for key, value in values.items()}
            self.last_layout['ptr_tokens'] = num_obj_ptr_tokens
        compatible = num_obj_ptr_tokens == self.ptr_tokens and all(
            value is not None and value.shape == self.inputs[key].shape and value.dtype == self.inputs[key].dtype
            for key, value in values.items())
        if not compatible:
            return self.inner(curr=curr, memory=memory, curr_pos=curr_pos,
                              memory_pos=memory_pos, num_obj_ptr_tokens=num_obj_ptr_tokens)
        for key, value in values.items():
            self.inputs[key].copy_(value)
        self.graph.replay()
        self.hits += 1
        return self.output


class CudaMaskDecoder:
    """Fixed tracking prompt shape only; user box prompts stay on eager path."""
    @torch.inference_mode()
    def __init__(self, model):
        self.original = model.sam_mask_decoder.forward
        self.hits = 0
        side = model.image_size // model.backbone_stride
        device = model.device
        self.inputs = dict(
            image_embeddings=torch.zeros((1, 256, side, side), device=device),
            image_pe=torch.zeros((1, 256, side, side), device=device, dtype=torch.float16),
            sparse_prompt_embeddings=torch.zeros((1, 2, 256), device=device),
            dense_prompt_embeddings=torch.zeros((1, 256, side, side), device=device))
        self.high_res = [torch.zeros((1, 32, side * 4, side * 4), device=device, dtype=torch.float16),
                         torch.zeros((1, 64, side * 2, side * 2), device=device, dtype=torch.float16)]
        # Hiera/FPN and the memory-attention reshape feed channels-last views
        # into this decoder. Preserve that layout so cuDNN picks the same math.
        for key in ('image_embeddings', 'image_pe'):
            self.inputs[key] = self.inputs[key].contiguous(memory_format=torch.channels_last)
        self.high_res = [value.contiguous(memory_format=torch.channels_last) for value in self.high_res]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            for _ in range(3):
                self.original(**self.inputs, multimask_output=True, repeat_image=False, high_res_features=self.high_res)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            self.output = self.original(**self.inputs, multimask_output=True, repeat_image=False, high_res_features=self.high_res)
        torch.cuda.synchronize()

    def __call__(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
                 multimask_output, repeat_image, high_res_features=None):
        values = dict(image_embeddings=image_embeddings, image_pe=image_pe,
                      sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings)
        if self.hits == 0:
            self.last_layout = {key: (tuple(value.shape), str(value.dtype)) for key, value in values.items()}
            self.last_layout['high_res'] = [(tuple(v.shape), str(v.dtype)) for v in (high_res_features or [])]
            self.last_layout['flags'] = (multimask_output, repeat_image)
        matches = (multimask_output and not repeat_image and high_res_features is not None
                   and len(high_res_features) == 2 and all(
                       value.shape == self.inputs[key].shape and value.dtype == self.inputs[key].dtype
                       for key, value in values.items()) and all(
                       value.shape == target.shape and value.dtype == target.dtype
                       for value, target in zip(high_res_features, self.high_res)))
        if not matches:
            return self.original(**values, multimask_output=multimask_output,
                                 repeat_image=repeat_image, high_res_features=high_res_features)
        for key, value in values.items():
            self.inputs[key].copy_(value)
        for target, value in zip(self.high_res, high_res_features):
            target.copy_(value)
        self.graph.replay()
        self.hits += 1
        # Scores are retained in SAMURAI history. They must not alias graph
        # buffers which the next replay overwrites.
        return tuple(value.clone() for value in self.output)
