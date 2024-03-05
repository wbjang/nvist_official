import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Type
from timm.models.vision_transformer import Mlp, DropPath, LayerScale

# Some of the code here is adapted from the timm library and DiT:
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    # https://github.com/facebookresearch/DiT
# and then modified by the author.

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """
    Modulates the input tensor by scaling and shifting its elements.
    We use this function when we use Adaptive Layer Normalisation instead of LayerNorm.

    Parameters:
    x (Tensor): The input tensor to be modulated.
    shift (Tensor): The tensor containing shift values for modulation.
    scale (Tensor): The tensor containing scale factors for modulation.

    Returns:
    Tensor: The modulated tensor after applying the scale and shift.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# now we change qkv_norm = True
class Attention(nn.Module):
    """
    Implements a multi-head attention mechanism with optional normalization on Q, K, and V vectors.

    This class supports both standard multi-head attention using dot-product of query (Q), key (K),
    and value (V) vectors, and an option to normalize Q, K, and V vectors before applying attention.
    It provides flexibility in the dimensionality of Q, K, and V through `dim_qk` and `dim_v` parameters
    and allows the addition of bias in the linear transformation of Q, K, and V.

    Parameters:
    - dim (int): Dimensionality of the input feature vector.
    - num_heads (int, optional): Number of attention heads. Default: 8.
    - qkv_bias (bool, optional): If True, adds a learnable bias to Q, K, V projections. Default: False.
    - qkv_norm (bool, optional): If True, applies normalization to Q, K, V vectors. Default: True.
    - attn_drop (float, optional): Dropout rate for attention weights. Default: 0.0.
    - proj_drop (float, optional): Dropout rate for the output of the attention block. Default: 0.0.
    - norm_layer (Type[nn.Module], optional): Normalization layer to use for Q, K, V vectors if `qkv_norm` is True.
      Default: nn.LayerNorm.
    - dim_qk (Optional[int], optional): Dimensionality of Q and K vectors. If None, uses `dim`. Default: None.
    - dim_v (Optional[int], optional): Dimensionality of V vectors. If None, uses `dim`. Default: None.

    The forward pass accepts Q, K (optional), and V (optional) tensors, and returns the attention output tensor.
    If K or V are not provided, they default to Q.

    Forward Parameters:
    - q (Tensor): Query tensor of shape [Bq, Nq, Cq].
    - k (Optional[Tensor], optional): Key tensor of shape [Bk, Nk, Ck]. If None, `k` defaults to `q`. Default: None.
    - v (Optional[Tensor], optional): Value tensor of shape [Bv, Nv, Cv]. If None, `v` defaults to `k`. Default: None.

    Returns:
    - Tensor: Output tensor of the attention mechanism.

    The attention mechanism scales the dot product of Q and K by the inverse square root of the dimension of K,
    applies softmax to obtain attention weights, then computes a weighted sum of V vectors. Optionally, if 
    `qkv_norm` is enabled, Q, K, and V vectors are normalized before attention is applied. The output then 
    goes through a linear projection and an optional dropout.
    """
    
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qkv_norm: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            dim_qk: Optional[int] = None,
            dim_v: Optional[int] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        if dim_qk is None: 
            dim_qk = dim
        if dim_v is None: 
            dim_v = dim

        #self.fused_attn = use_fused_attn()
        # refere to https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        self.fused_attn = torch.backends.cuda.sdp_kernel(enable_math=False)

        self.q = nn.Linear(dim, dim_qk, bias=qkv_bias)
        self.k = nn.Linear(dim, dim_qk, bias=qkv_bias)
        self.v = nn.Linear(dim, dim_v, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qkv_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qkv_norm else nn.Identity()
        self.v_norm = norm_layer(self.head_dim) if qkv_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        # do we need dim_out? dim_out = dim
        self.proj = nn.Linear(dim_v, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: Tensor, k: Optional[Tensor] = None, v: Optional[Tensor] = None) -> Tensor:
        if k is None: 
            k = q
        if v is None: 
            v = k

        Bq, Nq, Cq = q.shape
        Bk, Nk, Ck = k.shape
        Bv, Nv, Cv = v.shape
        # after self.qkv(x) : B, N, 3*C -> B, N, 3, self.num_heads, self.head_dim
        # 3, B, self.num_heads, N, self.head_dim (after shuffling)

        q = self.q(q).reshape(Bq, Nq, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k(k).reshape(Bk, Nk, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v(v).reshape(Bv, Nv, self.num_heads, self.head_dim).permute(0,2,1,3)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v : B, self.num_heads, N, self.head_dim
        # q, k, v = qkv.unbind(0)
        # we are computing the attention matrix
        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # attn: B, self.num_heads, N, N
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            # x : B, self.num_heads, N, self.head_dim
            x = attn @ v

        # x: B, N, self.num_heads, self.head_dim -> B, N, C
        x = x.transpose(1, 2).reshape(Bq, Nq, Cv)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Block(nn.Module):
    """
    Implements a transformer block with optional adaptive layer normalization (AdaLN), layer scaling,
    and path dropping for the Vision Transformer architecture.

    This block comprises a sequence of operations starting with normalization, followed by a multi-head
    self-attention mechanism, optional separate normalization for key and value vectors, layer scaling, 
    path dropping, a second normalization, a multilayer perceptron (MLP), and another set of layer scaling
    and path dropping operations.

    Parameters:
    - dim (int): Dimensionality of the input feature vector.
    - num_heads (int): Number of attention heads.
    - mlp_ratio (float, optional): Ratio of the MLP hidden dimension to the input dimension. Default: 4.0.
    - qkv_bias (bool, optional): If True, adds a learnable bias to Q, K, V projections. Default: True.
    - kv_separate_norm (bool, optional): If True, applies separate normalization to K and V vectors. Default: False.
    - proj_drop (float, optional): Dropout rate for the output of attention and MLP blocks. Default: 0.0.
    - attn_drop (float, optional): Dropout rate for attention weights. Default: 0.0.
    - init_values (Optional[float], optional): Initial values for layer scaling. If None, layer scaling is not used. Default: None.
    - drop_path (float, optional): Drop path rate. If 0., path dropping is not used. Default: 0.0.
    - act_layer (Type[nn.Module], optional): Activation layer to use within the MLP. Default: nn.GELU.
    - norm_layer (Type[nn.Module], optional): Normalization layer to use. Default: nn.LayerNorm.
    - use_adaln (bool, optional): If True, enables adaptive layer normalization (AdaLN). Default: False.
    - mlp_layer (Type[nn.Module], optional): MLP layer class. Default: Mlp.

    The forward pass expects a tensor `x` as input and returns the transformed tensor.

    Forward Parameters:
    - x (Tensor): Input tensor of shape [batch_size, num_patches + 1, dim].

    Returns:
    - Tensor: Output tensor of the same shape as the input.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            kv_separate_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            use_adaln: bool = False,
            mlp_layer: Type[nn.Module] = Mlp,
    ):
        super().__init__()
        if use_adaln: 
            norm_layer_affine = not use_adaln
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True)
            )
        else:
            norm_layer_affine = True

        self.use_adaln = use_adaln
        self.norm1 = norm_layer(dim, eps=1e-6, elementwise_affine=norm_layer_affine)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_norm=False, # because we already normalize the values here
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        if kv_separate_norm:
            self.norm_k = norm_layer(dim, eps=1e-6)
            self.norm_v = norm_layer(dim, eps=1e-6)
        

        self.kv_separate_norm = kv_separate_norm
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, eps=1e-6, elementwise_affine=norm_layer_affine)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: Tensor, k: Optional[Tensor] = None, v: Optional[Tensor] = None, c: Optional[Tensor] = None) -> Tensor:

        if self.use_adaln:
            assert c is not None
            q = self.norm1(x) # no elementwife_affine
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            q = modulate(q, shift_msa, scale_msa)
            if k is None: 
                k = q
            else: 
                if self.kv_separate_norm: 
                    k = self.norm_k(k)
                else: 
                    k = self.norm1(k)
            if v is None: 
                v = q
            else: 
                if self.kv_separate_norm: 
                    v = self.norm_v(v)
                else: 
                    v= self.norm1(v)
            x = x + gate_msa.unsqueeze(1) * self.drop_path1(self.ls1(self.attn(q=q, k=k, v=v)))
            x = x + gate_mlp.unsqueeze(1) * self.drop_path2(self.ls2(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))))
            
        else:
            q = self.norm1(x)
            if k is None: 
                k = x
            if v is None: 
                v = x
            if self.kv_separate_norm:
                k, v = self.norm_k(k), self.norm_v(v)
            else:
                k, v = self.norm1(k), self.norm1(v)

            x = x + self.drop_path1(self.ls1(self.attn(q=q, k=k, v=v)))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
    