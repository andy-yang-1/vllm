"""Multi-head attention."""
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
from xformers import ops as xops

from vllm import attention_ops
from vllm import modified_attention_ops
from vllm import cache_ops
from vllm import pos_encoding_ops
from vllm.core.block_manager import BlockSpaceManager
from vllm.model_executor.input_metadata import InputMetadata

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 128]


# debug_dict: Dict[Tuple[int,int,int],Tuple[torch.Tensor,torch.Tensor]] = {}
debug_dict: Dict[int,Tuple[torch.Tensor,torch.Tensor]] = {}

def modified_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query = query * scale
    attn = torch.einsum('qhd,khd->hqk', query, key)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum('hqk,khd->qhd', attn, value)

    # Sort the values in attn and return their indices
    attn_scores_summed = torch.sum(attn, dim=0)
    sorted_values, sorted_indices = torch.sort(attn_scores_summed, dim=-1)
    sorted_indices = sorted_indices.squeeze(0)
    return out, sorted_indices


def modified_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    layer_id: int,
) -> Tuple[torch.Tensor, List[int]]:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    sorted_indexs = []
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        # Get the block_table for the current layer
        block_table = block_tables[i][layer_id]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):

            block_number = block_table[j]

            #TODO (andy) change offset after enabling block_size > 1
            block_offset = 0

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)

        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_size ** 0.5)
        out, sorted_indices = modified_masked_attention(q, keys, values, scale)
        out = out.view(num_heads, head_size)
        output[i].copy_(out, non_blocking=True)

        sorted_indexs.append(sorted_indices)

    return output, sorted_indexs



class PagedAttention(nn.Module):
    """GPT-style multi-head PagedAttention.

    This class takes flattened 1D query, key, and value tensors as input. The
    input 1D tensors can be split into three parts: the prompt tokens, the
    generation tokens, and the paddings.

    |<------------------------------------- num_valid_tokens ------------------------------------->|
    |<--------------- num_prompt_tokens -------------->|<------- num_generation_tokens (M) ------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--generation_0-->|...|<--generation_M-1-->|<--padding-->|

    The prompts might have different lengths, while the generation tokens always
    have length 1. The paddings are appended to make the input length a multiple
    of 8, which is desirable for Tensor Cores.

    The class does the following:
    1. Perform multi_query_kv_attention for the prompts. This operation does
        not use the KV cache.
    2. Wait for the cache operations (e.g., swap, copy) to finish. The cache
        operations are issued by the cache engine before executing the forward
        pass of the model, and they are executed asynchronously.
    3. Reshape and store the input key and value tensors in the KV cache.
    4. Perform single_query_cached_kv_attention for the generation tokens.
        This operation reads the previous key and value tensors from the KV
        cache.
    5. Output a flattened 1D tensor.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, layer_id: int = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.attn_op = xops.fmha.cutlass.FwOp()
        self.layer_id = layer_id

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,                   # [num_prompt_tokens, num_heads, head_size]
        query: torch.Tensor,                    # [num_prompt_tokens, num_heads, head_size]
        key: torch.Tensor,                      # [num_prompt_tokens, num_heads, head_size]
        value: torch.Tensor,                    # [num_prompt_tokens, num_heads, head_size]
        attn_bias: xops.AttentionBias,
    ) -> torch.Tensor:
        # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attn_bias,
            p=0.0,
            scale=self.scale,
            op=self.attn_op,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output.copy_(out.squeeze(0))
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,           # [num_generation_tokens, num_heads, head_size]
        query: torch.Tensor,            # [num_generation_tokens, num_heads, head_size]
        key_cache: torch.Tensor,        # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,      # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
    ) -> None:
        # block_size = value_cache.shape[3]
        # attention_ops.single_query_cached_kv_attention(
        #     output,
        #     query,
        #     key_cache,
        #     value_cache,
        #     self.scale,
        #     input_metadata.block_tables,
        #     input_metadata.context_lens,
        #     block_size,
        #     input_metadata.max_context_len,
        # )

        # _, sorted_indices = modified_single_query_cached_kv_attention(
        #     output,
        #     query,
        #     key_cache,
        #     value_cache,
        #     input_metadata.block_tables,
        #     input_metadata.context_lens,
        #     self.layer_id
        # )

        num_tokens = query.shape[0]
        num_heads = query.shape[1]
        block_size = value_cache.shape[3]

        max_context_len = input_metadata.max_context_len
        attn = torch.zeros(num_tokens, num_heads, max_context_len, dtype=torch.float, device='cuda')

        modified_attention_ops.modified_single_query_cached_kv_attention(
            output,
            attn,
            query,
            key_cache,
            value_cache,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            max_context_len,
            self.layer_id
        )

        # def discard():
        #     attn_score = torch.sum(attn, dim=1)
        #     for i in range(input_metadata.num_generation_tokens):
        #         seq_id = input_metadata.seq_groups[i+input_metadata.num_prompts][0][0]
        #         min_index = torch.argmin(attn_score[i][:input_metadata.context_lens[i]])
        #         BlockSpaceManager.discard_queue.append((seq_id, self.layer_id, min_index))

        def discard():
            heavy_const = 10
            attn_score = torch.sum(attn, dim=1)
            sorted_indices = torch.argsort(attn_score, dim=1)
            sorted_indices = sorted_indices[:, -heavy_const:].tolist() # denote cache size > heavy_const (keep size)
            seq_ids = [input_metadata.seq_groups[i+input_metadata.num_prompts][0][0] for i in range(input_metadata.num_generation_tokens)]
            BlockSpaceManager.discard_queue.append((seq_ids, self.layer_id, sorted_indices))

            # for i in range(input_metadata.num_generation_tokens):
            #     seq_id = input_metadata.seq_groups[i+input_metadata.num_prompts][0][0]
            #     indices = sorted_indices[i]
            #     # indices = torch.sort(indices,descending=True)[0]
            #     for index in indices:
            #         BlockSpaceManager.discard_queue.append((seq_id, self.layer_id, index))
        
        discard()

        # sorted_indices = None
        # torch.cuda.empty_cache()
        # heavy_ratio = 0.2
        # recent_ratio = 0.1
        # #TODO this part heavily affects performance
        # for i in range(input_metadata.num_generation_tokens):
        #     seq_id = input_metadata.seq_groups[i+input_metadata.num_prompts][0][0]
        #     indices = sorted_indices[i].tolist()
        #     # if len(indices) != input_metadata.context_lens[seq_id]:
        #     #     print("debug")
        #     prompt_len = len(input_metadata.seq_data[seq_id].prompt_token_ids)
        #     heavy_const = int(heavy_ratio * prompt_len)
        #     recent_const = prompt_len - int(recent_ratio * prompt_len)
        #     indices = [ind for ind in indices if ind < recent_const and ind < input_metadata.context_lens[i]]
        #     if len(indices) > heavy_const:
        #         indices = indices[:len(indices)-heavy_const]
        #         indices.sort(reverse=True)  # Sort indices from high to low
        #         for index in indices:
        #             BlockSpaceManager.discard_queue.append((seq_id, self.layer_id, index))
                    

    def forward(
        self,
        query: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key: torch.Tensor,                      # [num_tokens, num_heads * head_size]
        value: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key_cache: Optional[torch.Tensor],      # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: Optional[torch.Tensor],    # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:                          # [num_tokens, num_heads * head_size]
        # NOTE: The query, key, and value tensors must be sliced from a qkv
        # tensor of shape [num_tokens, 3 * num_heads * head_size].

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)

        # Pre-allocate the output tensor.
        output = torch.empty_like(query)

        # Compute the attention op for prompts.
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            self.multi_query_kv_attention(
                output[:num_prompt_tokens],
                query[:num_prompt_tokens],
                key[:num_prompt_tokens],
                value[:num_prompt_tokens],
                input_metadata.attn_bias,
            )

        # Wait until the cache op is done.
        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        # When key_cache and value_cache are not provided, the new key
        # and value vectors will not be cached.
        num_valid_tokens = input_metadata.num_valid_tokens
        if (num_valid_tokens > 0 and key_cache is not None
            and value_cache is not None):
            # The stride is 3 because the key and value are sliced from qkv.
            # print(f"input_metadata.slot_mapping: {input_metadata.slot_mapping[self.layer_id]} -> layer_id: {self.layer_id}")
            
            # TODO (andy): something wrong with reshape: num_valid_tokens doesn't match
            # input_ids padded to max but slot_mapping is not (cancel padding?)
            
            x = key_cache.shape[-1]
            block_size = key_cache.shape[-2]

            # for i in range(num_valid_tokens):
            #     reshaped_key = key[i].reshape(self.num_heads, self.head_size // x, x)
            #     block_idx = torch.div(input_metadata.slot_mapping[self.layer_id][i], block_size, rounding_mode='floor')
            #     block_offset = input_metadata.slot_mapping[self.layer_id][i] % block_size
            #     # block_idx = input_metadata.slot_mapping[self.layer_id][i]
            #     # block_offset = 0
            #     assert block_idx < key_cache.shape[0], "block_idx out of range"
            #     key_cache[block_idx, :, :, block_offset, :] = reshaped_key
            #     value_cache[block_idx, :, :, block_offset] = value[i]

            cache_ops.reshape_and_cache(
                key[:num_valid_tokens],
                value[:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata.slot_mapping[self.layer_id],
            )
            # for i in range(num_valid_tokens):
            #     debug_dict[int(input_metadata.slot_mapping[self.layer_id][i])] = (key[i], value[i])

        if input_metadata.num_generation_tokens > 0:
            assert key_cache is not None and value_cache is not None, (
                "key_cache and value_cache must be provided when "
                "generating tokens."
            )
            # Compute the attention op for generation tokens.
            self.single_query_cached_kv_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata)

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, self.num_heads * self.head_size)


class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
    ) -> None:
        super().__init__(num_heads, head_size, scale)

        # Create the cos and sin cache.
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j -> ij', t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # FIXME(woosuk): This assumes that we configure the default dtype when
        # initializing the model. Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        # Embedding size: [max_position, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,                # [num_tokens]
        query: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key: torch.Tensor,                      # [num_tokens, num_heads * head_size]
        value: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key_cache: torch.Tensor,                # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,              # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:                          # [num_tokens, num_heads * head_size]
        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
        )
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )
