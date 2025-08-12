import sys
import json

import torch
from flash_attn import flash_attn_with_kvcache

from tokenizer import ChatFormat, Tokenizer

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
    weights_only=False  # 若有 warning，可改 True；但若權重含 tensor 則需 False
)

with open(f'{model_name}/original/params.json', 'r') as f:
    config = json.load(f)

dim = config['dim']
n_layers = config['n_layers']
n_heads = config['n_heads']
n_kv_heads = config['n_kv_heads']
vocab_size = config['vocab_size']
multiple_of = config['multiple_of']
ffn_dim_multiplier = config['ffn_dim_multiplier']
norm_eps = config['norm_eps']
rope_theta = torch.tensor(config['rope_theta'], device=device)
head_dim = dim // n_heads  # 128
max_seq_len = 8192

# stop_tokens 用來判斷生成是否結束
stop_tokens = torch.tensor(list(tokenizer.stop_tokens), device=device)

# -----------------------------
# Embedding Layer
# -----------------------------
embedding_layer = torch.nn.Embedding(vocab_size, dim, device=device, _weight=model['tok_embeddings.weight'])

# -----------------------------
# RoPE 頻率預計算
# -----------------------------
freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
freqs_for_each_token = torch.outer(torch.arange(max_seq_len, device=device), freqs)
freqs_cis_max = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)  # (max_seq_len, head_dim//2)

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

    final_embedding = embedding_layer(tokens)
    freqs_cis = freqs_cis_max[start_pos:start_pos+T]  # (T, head_dim//2)

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

        # 使用全局共享的 kv_cache（會被重用）
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
# 主程式：重用 KV Cache
# -----------------------------
# 讀取 ShareGPT 數據
with open('my-sharegpt-filtered.json', 'r', encoding='utf-8') as f:
    sharegpt = json.load(f)

num_requests = int(sys.argv[1])
requests = []
for i in range(num_requests):
    convs = sharegpt[i]['conversations']
    if len(convs) > 0:
        requests.append({'role': 'user', 'content': convs[0]['value']})

# 🔁 只創建一次 KV Cache（重複使用）
kv_cache = [
    (
        torch.zeros((1, max_seq_len, n_kv_heads, head_dim), dtype=torch.bfloat16, device=device),
        torch.zeros((1, max_seq_len, n_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    )
    for _ in range(n_layers)
]

total_fragmented_memory = 0

print(f"開始處理 {num_requests} 個請求（重用 KV Cache）...\n")

for req_idx, dialog in enumerate(requests):
    print(f"[{req_idx+1}/{num_requests}] 處理中...")

    # Tokenize 對話
    prompt_tokens = ChatFormat(tokenizer).encode_dialog_prompt([dialog])
    prompt_len = len(prompt_tokens)

    # 初始化 tokens (1, max_seq_len)
    tokens = torch.full((1, max_seq_len), tokenizer.pad_id, dtype=torch.long, device=device)
    tokens[0, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    input_text_mask = tokens != tokenizer.pad_id
    eos_reached = torch.tensor([False], device=device)

    generated_tokens = []
    prev_pos = 0  # 🔄 每次都從 0 開始，KV Cache 自動覆蓋舊資料

    # 自回歸生成
    for cur_pos in range(prompt_len, max_seq_len):
        # 🔁 使用同一組 kv_cache，但從 prev_pos 開始寫入
        next_token = forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[0, cur_pos] = next_token.item()
        generated_tokens.append(next_token.item())

        # 檢查是否遇到 stop token
        if next_token.item() in tokenizer.stop_tokens:
            eos_reached[0] = True

        prev_pos = cur_pos

        if eos_reached.item():
            break

    # 解碼生成結果（只取生成部分）
    start_idx = prompt_len
    end_idx = start_idx + len(generated_tokens)
    toks = tokens[0, start_idx:end_idx].tolist()

    # 移除 stop token 及之後內容
    cleaned_tokens = []
    for t in toks:
        if t in tokenizer.stop_tokens:
            break
        cleaned_tokens.append(t)

    decoded = tokenizer.decode(cleaned_tokens)
    print("📝 提示：", dialog['content'])
    print("💬 回覆：", decoded)
    print("-" * 80)

    # 計算此請求的記憶體碎片
    seq_len = prompt_len + len(cleaned_tokens)
    fragmented_slots = max_seq_len - seq_len
    fragmented_memory_bytes = fragmented_slots * n_kv_heads * head_dim * 2 * 2 * n_layers  # bfloat16 = 2 bytes
    total_fragmented_memory += fragmented_memory_bytes

    print(f"📊 [請求 {req_idx+1}] 生成長度: {len(cleaned_tokens)}, "
          f"總序列長度: {seq_len}, "
          f"碎片記憶體: {fragmented_memory_bytes / 1e6:.2f} MB")

# -----------------------------
# 最終統計
# -----------------------------
total_gb = total_fragmented_memory / 1e9
total_ratio = total_fragmented_memory / torch.cuda.get_device_properties(0).total_memory

print(f"\n✅ 完成 {num_requests} 個請求")
print(f"💥 總碎片記憶體: {total_gb:.2f} GB ({total_ratio * 100:.2f}%)")