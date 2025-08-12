import sys
import json

import torch
from flash_attn import flash_attn_with_kvcache

from tokenizer import Tokenizer  # 只需要 Tokenizer

# -----------------------------
# 設定與模型載入
# -----------------------------
device = 'cuda'
model_name = 'Meta-Llama-3-8B-Instruct'
tokenizer_path = f'{model_name}/original/tokenizer.model'
tokenizer = Tokenizer(model_path=tokenizer_path)

# 載入模型權重
model = torch.load(
    f'{model_name}/original/consolidated.00.pth',
    map_location=device,
    mmap=False,
    weights_only=False
)

with open(f'{model_name}/original/params.json', 'r') as f:
    config = json.load(f)

dim = config['dim']
n_layers = config['n_layers']
n_heads = config['n_heads']
n_kv_heads = config['n_kv_heads']
vocab_size = config['vocab_size']  # Llama-3: 128256
multiple_of = config['multiple_of']
ffn_dim_multiplier = config['ffn_dim_multiplier']
norm_eps = config['norm_eps']
rope_theta = torch.tensor(config['rope_theta'], device=device)
head_dim = dim // n_heads  # 128
max_seq_len = 8192

# Llama-3 Special Token IDs
LLAMA3_BEGIN_OF_TEXT = 128000
LLAMA3_START_HEADER = 128006
LLAMA3_END_HEADER = 128007
LLAMA3_EOT_ID = 128009
LLAMA3_ASSISTANT_HEADER = 128008

stop_tokens_set = {LLAMA3_EOT_ID}
stop_tokens_tensor = torch.tensor(list(stop_tokens_set), device=device)

# -----------------------------
# Embedding Layer
# -----------------------------
embedding_layer = torch.nn.Embedding(vocab_size, dim, device=device, _weight=model['tok_embeddings.weight'])

# -----------------------------
# RoPE 頻率預計算
# -----------------------------
freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
freqs_for_each_token = torch.outer(torch.arange(max_seq_len, device=device), freqs)
freqs_cis_max = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

# -----------------------------
# RoPE 工具函數
# -----------------------------
def reshape_for_broadcast(freqs_cis, x):
    shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

# -----------------------------
# 前向推理函數
# -----------------------------
def forward(tokens, start_pos):
    bsz, T = tokens.shape
    assert bsz == 1, "Batch size must be 1"

    # 🔍 安全檢查：token 是否在合法範圍內
    if tokens.max() >= vocab_size or tokens.min() < 0:
        invalid = tokens[(tokens < 0) | (tokens >= vocab_size)]
        raise ValueError(f"Invalid token(s) found: {invalid.tolist()} (vocab_size={vocab_size})")

    final_embedding = embedding_layer(tokens)
    freqs_cis = freqs_cis_max[start_pos:start_pos+T]

    for layer in range(n_layers):
        q_layer = model[f'layers.{layer}.attention.wq.weight']
        k_layer = model[f'layers.{layer}.attention.wk.weight']
        v_layer = model[f'layers.{layer}.attention.wv.weight']
        w_layer = model[f'layers.{layer}.attention.wo.weight']

        layer_embedding_norm = rms_norm(final_embedding, model[f'layers.{layer}.attention_norm.weight'])

        q = layer_embedding_norm @ q_layer.T
        k = layer_embedding_norm @ k_layer.T
        v = layer_embedding_norm @ v_layer.T

        q = q.view(bsz, T, n_heads, head_dim)
        k = k.view(bsz, T, n_kv_heads, head_dim)
        v = v.view(bsz, T, n_kv_heads, head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        k_cache, v_cache = kv_cache[layer]
        y = flash_attn_with_kvcache(
            q, k_cache, v_cache, k, v,
            cache_seqlens=start_pos,
            causal=True
        )
        stacked_qkv_attention = y.view(bsz, T, dim)

        embedding_delta = stacked_qkv_attention @ w_layer.T
        embedding_after_edit = final_embedding + embedding_delta

        ffn_norm = rms_norm(embedding_after_edit, model[f'layers.{layer}.ffn_norm.weight'])
        w1 = model[f'layers.{layer}.feed_forward.w1.weight']
        w2 = model[f'layers.{layer}.feed_forward.w2.weight']
        w3 = model[f'layers.{layer}.feed_forward.w3.weight']
        up = torch.functional.F.silu(ffn_norm @ w1.T)
        gate = ffn_norm @ w3.T
        output_after_feedforward = (up * gate) @ w2.T

        final_embedding = embedding_after_edit + output_after_feedforward

    final_embedding = rms_norm(final_embedding, model['norm.weight'])
    logits = final_embedding[:, -1, :] @ model['output.weight'].T
    next_token = torch.argmax(logits, dim=-1)
    return next_token

# -----------------------------
# 手動構建 Llama-3 對話 prompt（修正版：加上 bos/eos=False）
# -----------------------------
def build_prompt(tokenizer, dialog):
    """
    dialog: [{'role': 'user', 'content': '...'}]
    """
    tokens = [LLAMA3_BEGIN_OF_TEXT]  # <|begin_of_text|>

    for message in dialog:
        role = message["role"]
        content = message["content"]

        # 設定 header
        if role == "system":
            tokens.extend([LLAMA3_START_HEADER, LLAMA3_END_HEADER])
        elif role == "user":
            tokens.extend([LLAMA3_START_HEADER, LLAMA3_END_HEADER])
        elif role == "assistant":
            tokens.extend([LLAMA3_ASSISTANT_HEADER, LLAMA3_END_HEADER])
        else:
            raise ValueError(f"Unknown role: {role}")

        # ✅ 關鍵修正：加上 bos=False, eos=False
        content_tokens = tokenizer.encode(content, bos=False, eos=False)
        tokens.extend(content_tokens)
        tokens.append(LLAMA3_EOT_ID)  # <|eot_id|>

    # 準備生成 assistant 回應
    tokens.extend([LLAMA3_ASSISTANT_HEADER, LLAMA3_END_HEADER])
    return tokens

# -----------------------------
# 主程式：支援 Prompt Cache
# -----------------------------
with open('my-sharegpt-filtered.json', 'r', encoding='utf-8') as f:
    sharegpt = json.load(f)

num_requests = int(sys.argv[1])
requests = []
for i in range(num_requests):
    convs = sharegpt[i]['conversations']
    if len(convs) > 0:
        requests.append({'role': 'user', 'content': convs[0]['value']})

# 🔁 全局 KV Cache（重複使用，不重建）
kv_cache = [
    (
        torch.zeros((1, max_seq_len, n_kv_heads, head_dim), dtype=torch.bfloat16, device=device),
        torch.zeros((1, max_seq_len, n_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    )
    for _ in range(n_layers)
]

# 🟨 Prompt Cache 池
kv_cache_pool = {}

total_fragmented_memory = 0

print(f"開始處理 {num_requests} 個請求（支援 Prompt Cache）...\n")

for req_idx, dialog in enumerate(requests):
    print(f"[{req_idx+1}/{num_requests}] 處理中...")

    # ✅ 使用修正版 build_prompt
    prompt_tokens = build_prompt(tokenizer, [dialog])
    prompt_len = len(prompt_tokens)

    # 🔍 安全檢查
    if not all(0 <= t < vocab_size for t in prompt_tokens):
        invalid = [t for t in prompt_tokens if not (0 <= t < vocab_size)]
        raise ValueError(f"Invalid token in prompt: {invalid}")

    # 初始化 tokens (1, max_seq_len)
    pad_id = 128012  # 常見的 pad_id，可根據 tokenizer 調整
    tokens = torch.full((1, max_seq_len), pad_id, dtype=torch.long, device=device)
    tokens[0, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    input_text_mask = tokens != pad_id
    eos_reached = torch.tensor([False], device=device)
    generated_tokens = []

    prompt_key = tuple(prompt_tokens)

    # 🔍 檢查是否命中 cache
    if prompt_key in kv_cache_pool:
        print(f"✅ 命中緩存！跳過 Prefill，直接生成")
        cached_kv = kv_cache_pool[prompt_key]
        for layer in range(n_layers):
            k_c, v_c = cached_kv[layer]
            kv_cache[layer][0][:, :prompt_len] = k_c[:, :prompt_len]
            kv_cache[layer][1][:, :prompt_len] = v_c[:, :prompt_len]
        prev_pos = prompt_len
    else:
        print(f"🆕 新 prompt，執行 Prefill (長度: {prompt_len})...")
        prev_pos = 0
        with torch.no_grad():
            forward(tokens[:, prev_pos:prompt_len], prev_pos)
        prev_pos = prompt_len

        # 🔖 緩存 KV
        cached_kv = [(k.clone(), v.clone()) for k, v in kv_cache]
        kv_cache_pool[prompt_key] = cached_kv

    # ✅ 開始生成
    while prev_pos < max_seq_len - 1:
        cur_pos = prev_pos + 1
        next_token = forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[0, cur_pos] = next_token.item()
        generated_tokens.append(next_token.item())

        if next_token.item() in stop_tokens_set:
            eos_reached[0] = True

        prev_pos = cur_pos

        if eos_reached.item():
            break

    # 解碼生成結果
    start_idx = prompt_len
    end_idx = start_idx + len(generated_tokens)
    toks = tokens[0, start_idx:end_idx].tolist()

    cleaned_tokens = []
    for t in toks:
        if t in stop_tokens_set:
            break
        cleaned_tokens.append(t)

    try:
        decoded = tokenizer.decode(cleaned_tokens)
    except Exception as e:
        decoded = f"[Decode Error: {e}] Tokens: {cleaned_tokens}"

    print("📝 提示：", dialog['content'])
    print("💬 回覆：", decoded)
    print("-" * 80)

    # 計算碎片記憶體
    seq_len = prompt_len + len(cleaned_tokens)
    fragmented_slots = max_seq_len - seq_len
    fragmented_memory = fragmented_slots * n_kv_heads * head_dim * 2 * 2 * n_layers  # bfloat16 = 2 bytes
    total_fragmented_memory += fragmented_memory

    print(f"📊 [請求 {req_idx+1}] 生成長度: {len(cleaned_tokens)}, "
          f"總序列長度: {seq_len}, "
          f"碎片: {fragmented_memory / 1e6:.2f} MB")

# -----------------------------
# 最終統計
# -----------------------------
total_gb = total_fragmented_memory / 1e9
total_ratio = total_fragmented_memory / torch.cuda.get_device_properties(0).total_memory

print(f"\n✅ 總碎片記憶體: {total_gb:.2f} GB ({total_ratio * 100:.2f}%)")