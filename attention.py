from torch.nn.attention.flex_attention import flex_attention
import torch
flex_attention = torch.compile(flex_attention)
def attention(q,
            k,
            v,
            causal=False,
            dropout_p=0.,
            dtype=torch.float32,
             ):
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p)

    out = out.transpose(1, 2).contiguous()
    return out

def flex_attention_wrapper(q,
            k,
            v,
            causal=False,
            dropout_p=0.,
            dtype=torch.float32,
             ):
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = flex_attention(q, k, v)
    out = out.transpose(1, 2).contiguous()
    return out