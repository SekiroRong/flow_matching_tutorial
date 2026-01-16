import torch
from torch import nn
import math

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - (c // 2), c // 2], dim=1)

    # loop over samples
    output = []
    for i, (h, w) in enumerate(grid_sizes.tolist()):
        seq_len = h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:h].view(1, h, 1, -1).expand(1, h, w, -1),
            freqs[1][:w].view(1, 1, w, -1).expand(1, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

class LayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

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

class SelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class CrossAttention(SelfAttention):

    def forward(self, x, condition):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(condition)).view(b, -1, n, d)
        v = self.v(condition).view(b, -1, n, d)

        # compute attention
        x = attention(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm2 = LayerNorm(dim, eps)
        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                          eps)
        self.norm3 = nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        condition,
        # context,
        # context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], grid_sizes,
            freqs)
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, e):
            if condition is not None:
                x = x + self.cross_attn(self.norm3(x), condition)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, e)
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x
        
class Dummy_DiT(nn.Module):
    def __init__(self, 
                 patch_size=(1, 1),
                 num_layers=4,
                 in_dim=16,
                 dim=32,
                 freq_dim=64,
                 ffn_dim=64,
                 out_dim=1,
                 qk_norm=True,
                 num_heads=4,
                 window_size=(5, 5),
                 eps=1e-6
                ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        # embeddings
        self.conv_in = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.patch_embedding = nn.Conv2d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.label_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, ffn_dim, num_heads,
                              window_size, qk_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # calculate 2D rope params
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - (d // 2)),
            rope_params(1024, (d // 2))
        ],dim=1)

    def forward(
        self,
        x,
        t,
        y=None,
    ):
        # turn input_image into tokens [bs, C, H, W] -> [bs, dim, H, W]
        x = self.conv_in(x)
        x = self.patch_embedding(x)
        grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in x])
        x = x.flatten(2).transpose(1, 2)

        if y is not None:
            y = self.label_embedding(
            sinusoidal_embedding_1d(self.freq_dim, y).float()).unsqueeze(1)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        kwargs = dict(
            e=e0,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            condition=y,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('hwpqc->cphqw', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return torch.stack(out)
        


if __name__ == "__main__":
    dummy_model = Dummy_DiT()
    dummy_input = torch.zeros((2, 1, 28, 28))
    dummy_model(dummy_input, torch.tensor([0.5, 0.2]))
    dummy_model(dummy_input, torch.tensor([0.5, 0.2]), torch.tensor([0, 1]))
        