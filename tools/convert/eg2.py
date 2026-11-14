"""EmbeddingGemma 2 text encoder, re-implemented for Core ML / Apple Neural Engine (ANE) conversion.

Everything here mirrors transformers' EmbeddingGemma2TextModel (+ mean pooling + L2 normalize) and is checked
against it (see verify_hf.py). Differences are numerically neutral rewrites that make the fp16 ANE graph safe:
  * RMSNorm divides each row by its max-abs first (fp16 overflow guard; exact in real arithmetic).
  * Attention uses the "fold" layout: the query heads sharing one K/V head are folded into the row dimension
    ([1,kvh,g*S,hd] @ [1,kvh,hd,S]). The obvious repeat_interleave layout makes the ANE write zeros for query
    heads 1..3 of the global layers (5/11/17/23, kvh=1) at large S. Do not change this without re-running the
    self-test on the ANE.
  * RoPE tables are graph constants at absolute positions 0..S-1. A runtime positions / rope input miscompiles.
  * Every mask is a runtime input (bias tensor). Graph-constant masks miscompile on the ANE for S > 512.
  * The per-layer-embedding (PLE) projection is computed per layer from a [512,512] slice of the [12288,512]
    weight, so the bucket functions and the long-path layer functions carry identical weight constants and
    the multifunction merge stores them once.

Modules (all fp32 here; converted with compute_precision=FLOAT16):
  Bucket(S, K)        embed_S          one call embeds up to K packed sequences of total length <= S
  SwaLayer(i)         long_swa_l{i}    sliding layer i on one 1024-row chunk with 512-row halos
  GlobalLayer(i, S)   long{S}_glb_l{i} global layer i over the whole (padded) document, S in {4096, 8192}
  GlobalHead(S)       long{S}_head     global layer 23 + final norm + mean pool + projection + L2 normalize
  Reference(S)        fp32 ground truth (attention_mask input, constant band), HF-equivalent
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

NEG = -1e4            # additive mask value (fp16-safe)
WINDOW = 512          # config.sliding_window; HF bidirectional sliding mask keeps |i - j| <= 512
CHUNK = 1024          # long path: query rows per sliding-layer call
HALO = 512            # long path: key rows on each side of the chunk (== WINDOW)
LONG_BUCKETS = (4096, 8192)
BUCKETS = (64, 128, 256, 512, 1024, 2048)
SCALE = math.sqrt(512)


def k_for(S):
    """pooled slots (packed sequences) per bucket call"""
    return max(8, S // 32)


def rms(x, w=None, eps=1e-6):
    s = torch.maximum(x.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=x.dtype))
    y = x / s
    out = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + eps / (s * s))
    return out * w if w is not None else out


def rot_half(x, h):
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(n, hd, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    fr = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos(), emb.sin()


def attend_fold(q, k, v, bias, kvh):
    """q [1,4,Sq,hd], k/v [1,kvh,Sk,hd], bias broadcastable to [1,4,Sq,Sk] -> [1,4,Sq,hd]"""
    nh, sq, hd, sk = (int(t) for t in (q.shape[1], q.shape[2], q.shape[3], k.shape[2]))  # static when traced
    g = nh // kvh
    qf = q.reshape(1, kvh, g * sq, hd)
    s = torch.matmul(qf, k.transpose(-1, -2)).reshape(1, nh, sq, sk) + bias
    w = torch.softmax(s, dim=-1).reshape(1, kvh, g * sq, sk)
    return torch.matmul(w, v).reshape(1, nh, sq, hd)


class Weights(nn.Module):
    """fp32 copies of the text-model weights (language_model.* in model.safetensors)."""

    def __init__(self, hf_dir):
        super().__init__()
        cfg = json.load(open(f"{hf_dir}/config.json"))["text_config"]
        self.cfg = cfg
        f = safe_open(f"{hf_dir}/model.safetensors", "pt")

        def g(k):
            return nn.Parameter(f.get_tensor("language_model." + k).float(), requires_grad=False)

        self.L = cfg["num_hidden_layers"]
        assert cfg["sliding_window"] == WINDOW and cfg["hidden_size"] == 512
        ple_w = g("ple.per_layer_model_projection.weight")          # [24*512, 512]
        self.ple_norm = g("ple.per_layer_projection_norm.weight")
        self.norm = g("norm.weight")
        self.proj = g("embedding_projection.weight")                  # [768, 512]
        self.layers = nn.ModuleList()
        for i, t in enumerate(cfg["layer_types"]):
            m = nn.Module()
            for k in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                      "post_feedforward_layernorm", "ple_block.post_per_layer_input_norm"):
                setattr(m, k.replace(".", "_"), g(f"layers.{i}.{k}.weight"))
            for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                setattr(m, k, g(f"layers.{i}.self_attn.{k}.weight"))
            for k in ("gate_proj", "up_proj", "down_proj"):
                setattr(m, k, g(f"layers.{i}.mlp.{k}.weight"))
            m.ple_gate = g(f"layers.{i}.ple_block.per_layer_input_gate.weight")
            m.ple_proj = g(f"layers.{i}.ple_block.per_layer_projection.weight")
            m.ple_w = nn.Parameter(ple_w[i * 512:(i + 1) * 512].clone(), requires_grad=False)
            m.scalar = g(f"layers.{i}.layer_scalar")
            m.hd = int(m.q_norm.shape[0])
            m.kvh = int(m.k_proj.shape[0]) // m.hd
            m.swa = t == "sliding_attention"
            m.theta = cfg["rope_parameters"][t]["rope_theta"]
            self.layers.append(m)
        self.global_layers = [i for i, m in enumerate(self.layers) if not m.swa]


def layer(W, i, x, x0, cq, sq, ck, sk, bias, q0=0, nq=None):
    """Decoder layer i. x [1,Nk,512] hidden rows used as keys; queries (and the residual stream) are rows
    q0..q0+nq of x (all rows when nq is None). x0 = inputs_embeds of the query rows (per-layer input)."""
    m = W.layers[i]
    hd, kvh = m.hd, m.kvh
    h = rms(x, m.input_layernorm)
    if nq is not None:
        x, hq = x[:, q0:q0 + nq], h[:, q0:q0 + nq]
    else:
        hq = h
    q = rms(F.linear(hq, m.q_proj).reshape(1, -1, 4, hd).transpose(1, 2), m.q_norm)
    k = rms(F.linear(h, m.k_proj).reshape(1, -1, kvh, hd).transpose(1, 2), m.k_norm)
    v = rms(F.linear(h, m.v_proj).reshape(1, -1, kvh, hd).transpose(1, 2))
    q = q * cq + rot_half(q, hd // 2) * sq
    k = k * ck + rot_half(k, hd // 2) * sk
    o = attend_fold(q, k, v, bias, kvh).transpose(1, 2).reshape(1, -1, 4 * hd)
    x = x + rms(F.linear(o, m.o_proj), m.post_attention_layernorm)
    h = rms(x, m.pre_feedforward_layernorm)
    h = F.linear(F.gelu(F.linear(h, m.gate_proj), approximate="tanh") * F.linear(h, m.up_proj), m.down_proj)
    x = x + rms(h, m.post_feedforward_layernorm)
    ple = rms(F.linear(x0, m.ple_w) * (512 ** -0.5), W.ple_norm)
    gte = F.gelu(F.linear(x, m.ple_gate), approximate="tanh") * ple
    x = x + rms(F.linear(gte, m.ple_proj), m.ple_block_post_per_layer_input_norm)
    return x * m.scalar


def head(W, x, pool):
    """x [1,S,512], pool [K,S] -> [K,768] L2-normalized (all-zero pool rows give zero rows)"""
    h = rms(x, W.norm)[0]
    e = F.linear(torch.matmul(pool, h), W.proj)
    s = torch.maximum(e.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=e.dtype))
    e = e / s
    return e * torch.rsqrt((e * e).sum(-1, keepdim=True) + 1e-6)


class _Tables(nn.Module):
    def __init__(self, W, n):
        super().__init__()
        for name, hd in (("swa", 256), ("glb", 512)):
            th = next(m.theta for m in W.layers if (m.hd == hd))
            c, s = rope_tables(n, hd, th)
            self.register_buffer(f"cos_{name}", c, persistent=False)
            self.register_buffer(f"sin_{name}", s, persistent=False)

    def get(self, m):
        n = "swa" if m.swa else "glb"
        return getattr(self, f"cos_{n}"), getattr(self, f"sin_{n}")


class Bucket(nn.Module):
    """embed_S: inputs_embeds [1,S,512], global_bias [1,1,S,S], local_bias [1,1,S,S], pool [K,S] -> [K,768]"""

    def __init__(self, W, S):
        super().__init__()
        self.W, self.S = W, S
        self.t = _Tables(W, S)

    def forward(self, inputs_embeds, global_bias, local_bias, pool):
        x = inputs_embeds
        for i, m in enumerate(self.W.layers):
            c, s = self.t.get(m)
            x = layer(self.W, i, x, inputs_embeds, c, s, c, s, local_bias if m.swa else global_bias)
        return head(self.W, x, pool)


class SwaLayer(nn.Module):
    """long_swa_l{i}: hidden_window [1,C+2H,512], inputs_embeds [1,C,512], bias [1,1,C,C+2H] -> hidden_out [1,C,512]
    RoPE positions are window-local (RoPE only depends on i - j)."""

    def __init__(self, W, i):
        super().__init__()
        self.W, self.i = W, i
        assert W.layers[i].swa
        self.t = _Tables(W, CHUNK + 2 * HALO)

    def forward(self, hidden_window, inputs_embeds, bias):
        c, s = self.t.get(self.W.layers[self.i])
        return layer(self.W, self.i, hidden_window, inputs_embeds, c[HALO:HALO + CHUNK], s[HALO:HALO + CHUNK], c, s,
                     bias, q0=HALO, nq=CHUNK)


class GlobalLayer(nn.Module):
    """long{S}_glb_l{i}: hidden [1,S,512], inputs_embeds [1,S,512], key_bias [1,1,1,S] -> hidden_out [1,S,512]"""

    def __init__(self, W, i, S):
        super().__init__()
        self.W, self.i = W, i
        assert not W.layers[i].swa
        self.t = _Tables(W, S)

    def forward(self, hidden, inputs_embeds, key_bias):
        c, s = self.t.get(self.W.layers[self.i])
        return layer(self.W, self.i, hidden, inputs_embeds, c, s, c, s, key_bias)


class GlobalHead(GlobalLayer):
    """long{S}_head: the last layer (23, global) fused with the output head.
    hidden, inputs_embeds, key_bias as GlobalLayer, pool [1,S] (host.long_pool: 2^-10 on the n valid rows; not 1/n, which underflows fp16) -> embedding [1,768]"""

    def __init__(self, W, S):
        super().__init__(W, W.L - 1, S)

    def forward(self, hidden, inputs_embeds, key_bias, pool):
        return head(self.W, super().forward(hidden, inputs_embeds, key_bias), pool)


class Reference(nn.Module):
    """fp32 ground truth for ONE sequence: inputs_embeds [1,S,512], attention_mask [1,S] -> [1,768].
    Standard (non-fold) attention with repeat_interleave, constant sliding band, mask-based mean pooling;
    equals sentence-transformers EmbeddingGemma 2 at cos 1.000000 (verify_hf.py)."""

    def __init__(self, W, S):
        super().__init__()
        self.W, self.S = W, S
        self.t = _Tables(W, S)
        i = torch.arange(S)
        self.register_buffer("band", torch.where((i[:, None] - i[None, :]).abs() <= WINDOW, 0.0, NEG)
                             .reshape(1, 1, S, S), persistent=False)

    def forward(self, inputs_embeds, attention_mask):
        W, S = self.W, self.S
        x = inputs_embeds
        pad = ((1.0 - attention_mask) * NEG).reshape(1, 1, 1, S)
        for i, m in enumerate(W.layers):
            c, s = self.t.get(m)
            hd, kvh = m.hd, m.kvh
            h = rms(x, m.input_layernorm)
            q = rms(F.linear(h, m.q_proj).reshape(1, S, 4, hd).transpose(1, 2), m.q_norm)
            k = rms(F.linear(h, m.k_proj).reshape(1, S, kvh, hd).transpose(1, 2), m.k_norm)
            v = rms(F.linear(h, m.v_proj).reshape(1, S, kvh, hd).transpose(1, 2))
            q = q * c + rot_half(q, hd // 2) * s
            k = (k * c + rot_half(k, hd // 2) * s).repeat_interleave(4 // kvh, dim=1)
            v = v.repeat_interleave(4 // kvh, dim=1)
            bias = pad + self.band if m.swa else pad
            w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, dim=-1)
            o = torch.matmul(w, v).transpose(1, 2).reshape(1, S, 4 * hd)
            x = x + rms(F.linear(o, m.o_proj), m.post_attention_layernorm)
            h = rms(x, m.pre_feedforward_layernorm)
            h = F.linear(F.gelu(F.linear(h, m.gate_proj), approximate="tanh") * F.linear(h, m.up_proj), m.down_proj)
            x = x + rms(h, m.post_feedforward_layernorm)
            ple = rms(F.linear(inputs_embeds, m.ple_w) * (512 ** -0.5), W.ple_norm)
            gte = F.gelu(F.linear(x, m.ple_gate), approximate="tanh") * ple
            x = x + rms(F.linear(gte, m.ple_proj), m.ple_block_post_per_layer_input_norm)
            x = x * m.scalar
        h = rms(x, W.norm)
        wpool = (attention_mask / attention_mask.sum(-1, keepdim=True)).reshape(1, 1, S)
        e = F.linear(torch.matmul(wpool, h), W.proj)[:, 0, :]
        return e / e.norm(dim=-1, keepdim=True)


_REFS = {}


def ref_embed(W, rows):
    """fp32 reference embedding of one sequence given its input rows [n,512] (fp32)"""
    n = len(rows)
    S = 1 << max(6, (n - 1).bit_length())
    if S not in _REFS:
        _REFS[S] = Reference(W, S).eval()
    x = torch.zeros(1, S, 512)
    x[0, :n] = torch.as_tensor(np.asarray(rows, np.float32))
    msk = torch.zeros(1, S)
    msk[0, :n] = 1
    with torch.no_grad():
        return _REFS[S](x, msk)[0].numpy()
