#include <torch/extension.h>

void modified_single_query_cached_kv_attention(
  torch::Tensor& out,
  torch::Tensor& attn,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  int layer_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "modified_single_query_cached_kv_attention",
    &modified_single_query_cached_kv_attention,
    "Compute the attention between an input query and the cached key/value tensors, and return the attention weights");
}
