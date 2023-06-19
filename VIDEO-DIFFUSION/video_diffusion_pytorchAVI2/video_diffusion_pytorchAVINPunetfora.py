import math
import copy
import torch
import torchvision
import cv2
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from video_diffusion_pytorch.text import tokenize, bert_embed, BERT_MODEL_DIM
from prettytable import PrettyTable
from torchsummary import summary


# helpers functions

def exists(x):
    return x is not None

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups #A: arr es un vector de dimensio 1xgroups amb coeficients divisor
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)#A: crea un vector de dimensio "shape" on tot son "Trues"
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)#A: crea un vector de dimensio "shape" on tot som "False"
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    #A: Take a look at https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        #A: heads = 8, max_distance = 32
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        #A: ret=
        # tensor([[ 0, 16, 16, 16, 16, 16, 16, 16],
        # [ 0,  0, 16, 16, 16, 16, 16, 16],
        # [ 0,  0,  0, 16, 16, 16, 16, 16],
        # [ 0,  0,  0,  0, 16, 16, 16, 16],
        # [ 0,  0,  0,  0,  0, 16, 16, 16],
        # [ 0,  0,  0,  0,  0,  0, 16, 16],
        # [ 0,  0,  0,  0,  0,  0,  0, 16],
        # [ 0,  0,  0,  0,  0,  0,  0,  0]])
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        #A: ret=
        # tensor([[ 0, 17, 18, 19, 20, 21, 22, 23],
        # [ 1,  0, 17, 18, 19, 20, 21, 22],
        # [ 2,  1,  0, 17, 18, 19, 20, 21],
        # [ 3,  2,  1,  0, 17, 18, 19, 20],
        # [ 4,  3,  2,  1,  0, 17, 18, 19],
        # [ 5,  4,  3,  2,  1,  0, 17, 18],
        # [ 6,  5,  4,  3,  2,  1,  0, 17],
        # [ 7,  6,  5,  4,  3,  2,  1,  0]])
        return ret

    def forward(self, n, device):
        #A: n=frames=8
        q_pos = torch.arange(n, dtype = torch.long, device = device)#A: [0,1,2,3,4,5,6,7]
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        #A: si n=8, rel_pos=
        # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        # [-1,  0,  1,  2,  3,  4,  5,  6],
        # [-2, -1,  0,  1,  2,  3,  4,  5],
        # [-3, -2, -1,  0,  1,  2,  3,  4],
        # [-4, -3, -2, -1,  0,  1,  2,  3],
        # [-5, -4, -3, -2, -1,  0,  1,  2],
        # [-6, -5, -4, -3, -2, -1,  0,  1],
        # [-7, -6, -5, -4, -3, -2, -1,  0]])
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        #A: rp_bucket=
        # tensor([
        # [ 0, 17, 18, 19, 20, 21, 22, 23],
        # [ 1,  0, 17, 18, 19, 20, 21, 22],
        # [ 2,  1,  0, 17, 18, 19, 20, 21],
        # [ 3,  2,  1,  0, 17, 18, 19, 20],
        # [ 4,  3,  2,  1,  0, 17, 18, 19],
        # [ 5,  4,  3,  2,  1,  0, 17, 18],
        # [ 6,  5,  4,  3,  2,  1,  0, 17],
        # [ 7,  6,  5,  4,  3,  2,  1,  0]])
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


# small helper modules

#A: Exponential moving average (EMA) handler can be used to compute a smoothed version of model
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        #A: el sentit de la formula que retorno https://pytorch.org/ignite/generated/ignite.handlers.ema_handler.EMAHandler.html
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        #A: x es la t=[t1,t2,t3,t4]
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) #A: tensor de dimensio 1xhalf_dim on en cada posició hi ha exp(-i*emb) on la i va de 0 a half_dim-1
        emb = x[:, None] * emb[None, :] #A: multipliquem el vector columna x pel vector fila emb donant lloc a una matriu de dimension b x half_dim
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        #A: retorna el mateix tensor x pero s'ha normalitzat la dimensio de canals
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        #A: time_emb_dim es dim*4 sempre
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        #A: x es de dimensio b x dim x f x h x w
        #A: time_emb es de dimensio 4 x time_emb_dim(=dim·4)
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        #A: self.scale = dim_head ** -0.5   al final he fet l'escalat amb sqrt(d_k) 
        self.heads = heads
        hidden_dim = dim_head * heads
        #A: Les tres lines de dalt son iguals que les de la classe Attention (Temporal Attention)
        
        self.to_qkv = nn.Conv2d(dim, dim_head * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim_head, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        
        #A: ES REALMENT NECESSARIA LA LINEA DE SOTA? si perq la conv necessita que els 3 ultims canals siguin els que son
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b e x y -> b e (x y)')
        qkv = qkv.chunk(3, dim = 1)
        #A: self.to_qkv multiplica cada pixel de x per un weight (parametre que s'anira aprenent). Aixo es fa hidden_dim * 3 (=768) vegades, el nou nombre de canals a la sortida de la convolucio
        #A: la dimensio dels canals es divideix en 3 de tal forma que ens quedem amb 3 tuples de dimensio (b·f) x dim_head x h x w on les dimensions h,w estan multiplicades per un parametre diferent en cada pixel que s'haura d'aprendre 
        q, k, v = rearrange_many(qkv, 'b e (h a) -> b h e a', h = self.heads)
        #A: q, k, v passen a ser de dimensio (b·f) x num_heads x dim_head x num_pixels_per_frame on cadascun dels coeficients de la ultima dimensio esta multiplicat per un parametre que s'apendra
        #A: fins aqui te sentit. Les qi, ki, vi son els pixels multiplicats per un parametre a aprendre. Com que estem fent atencio espaial, "els elements de la sequencia" son els pixels

        dot_product = torch.einsum('b h d m, b h d n -> b h m n', q, k)
        scale_factor = math.sqrt(k.size(-1))
        scaled_dot_product = dot_product / scale_factor
        similarity_scores = F.softmax(scaled_dot_product, dim=-1)
        #A: similarity_scores es de dimensio (b·f) x num_heads x num_pixels_per_frame x num_pixels_per_frame. A traves de la ultima dimensio recorro les k i a traves de la penultima les q

        weighted_values = torch.einsum('b h e d, b h a d -> b h e a', similarity_scores, v) #A: (num_pixels_per_frame x num_pixels_per_frame) x (num_pixels_per_frame x dim_head) = num_pixels_per_frame x dim_head)
        #A: weighted_values es de dimensio (b·f) x num_heads x num_pixels_per_frame x dim_head 
        #A: IMPORTANT: (la matriu dim_head x dim_head conte la informacio de com es relacionen els pixels dins de la imatge. Si mes no, aquesta informacio es independent a la posicio que ocupen. Per aixo s'hauria de valorar utilitzar algun tipus de positional embedding per tambe mantenir informacio sobre la posicio dels pixels dins de la imatge)
        
        concat_values = rearrange(weighted_values, 'b h p e -> b e (h p)')
        #A:concat_values es de dimensio b*f, dim_head, h*w

        img_size = rearrange(concat_values, 'b e (h w) -> b e h w', h = h, w = w)
        #A:img_size es de dimensio b*f, dim_head, h, w
        
        out = self.to_out(img_size)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b, f = f)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        #A: fn es un objecte de la classe Attention
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        #A: x es el vector de dimensio bxdim'c'xfxhxw que ha estat normalitzat en la dimensio c. El dim es el nombre de canals a la sortida de la convolucio feta previament
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        #A: Si shape = (2, 3, 4, 5, 6) llavors reconstitute_kwargs es
        # {
        #     'b': 2,
        #     'c': 3,
        #     'f': 4,
        #     'h': 5,
        #     'w': 6
        # }
        #A: x bxdim'c'xfxhxw
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        #A: x bxh·wxfxdim'c'
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    #A: LA CLASSE ATTENTION ES LA QUE FEM SERVIR PER TEMPORAL ATTENTION. Bueno, just a sobre de la classe EinopsToAndFrom ho defineix com attention along space and time
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)
        #A: first argument is input size and second argument is output size

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        #A: x bxh·wxfxdim'c' normalitzat en la dimensio c
        n, device = x.shape[-2], x.device
        #A: n es el nombre de frames. Fent memoria enrere aquest nombre de frames ve del nombre de frames de les imatges utilitzades d'input

        #A: La dimensio que entra a nn.Linear es la dels canals, es a dir, l'input de la primera neurona sera (-,-,-,0), el de la segona (-,-,-,1)...
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #A: qkv es l'output de self.to_qkv(x) dividit en la dimensio dels canals en un tuple de tres tensors es a dir bxh·wxfx(dim'c')/3

        # split out heads
        
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
        #A: q, k, v pasen a ser de dimensio b x h·w x heads x f x dim_heads
        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention
        #A: we enter to this if to take into account the relative position
        #A: BUT IN THE INIT_TEMPORAL_ATTN WE ARE NOT REALLY TAKING INTO ACCOUNT THE TEMPORAL (FRAME) INFORMATION, RIGHT? EACH OF THE CHANNELS THAT WE ARE TRYING TO CREATE ITS EMBEDDING REPRESENTATION ONLY CONTAINS FEATURES RELATED WITH THE SPATIAL (HxW) DIMENSION RIGHT? BECAUSE OF THE Conv3D WE DID WITH (1,7,7)
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        # similarity
        #A: fem el producte escalar entre ks i qs dels diferents heads per obtenir la semblansa
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        #A: es fa el producte escalar de les matrius de dimensio f x dim_heads on a dim_heads hi ha les ki i qi de cada head
        #A: el producte escalar (f x dim_heads) · (f x dim_heads)^T = (f x f)

        #A: CREC QUE LA SUMA NO ESTA BEN FETA PERQUE SIM ES EN LA DIMENSIO DELS CANALS I POS_BIAS EN LA DIMENSIO DELS FRAMES
        # relative positional bias
        if exists(pos_bias):
            #A: sim torch.Size([4, 16384, 8, 8, 8]) b x h·w x heads x f x f
            #A: pos_bias torch.Size([8, 8, 8]) embedding_dim (=num_heads) x f x f
            sim = sim + pos_bias
            #A: sim torch.Size([4, 16384, 8, 8, 8])

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model

class Unet3DNP(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        block_type = 'resnet',
        resnet_groups = 8
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding
        #A: RotaryEmbedding es un metode de Relative Positional Encoding millorat
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))
        
        #A: self.time_rel_pos_bias es una matriu que conte els embeddings. Els coeficients de la matriu es van actualitzant en cada step per minimitzar la funcio de cost    
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet
        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))
        #A: El 2n argument de PreNorm es un objecte de la classe EinopsToAndFrom

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            #A: ind [0,1,2,3] (dim_in, dim_out) [(128, 128), (128, 256), (256, 512), (512, 1024)]
            is_last = ind >= (num_resolutions - 1)#A: true en la ultima iteracio
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                #Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity() #A: el Downsample es de 2 i es fa en la dimensio dels pixels (dimensions h i w)
            ]))

        mid_dim = dims[-1] #A: agafa l'ultim dim_out del for anterior
        
        #A: PER QUE CANVIA L'ORDRE DE LES OPERACIONS EN AQUESTA ZONA INTERMITJA? PER QUE NO SEGUEIX EL MATEIX ORDRE QUE AL FOR PREVI?
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        #A: PER QUE AQUI CALCULA EL SPATIAL ATTENTION D'AQUESTA FORMA DIFERENT I NO COM AL FOR PREVI??
        #spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))
        #self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, SpatialLinearAttention(mid_dim, heads = attn_heads)))
        
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                #Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        batch, device = x.shape[0], x.device
        
        #A: com que nosaltres no li passem el focus_present_mask, s'executa prob_mask_like
        #A: com que pasem prob_focus_present=0, focus_present_mask sera un vector de dimensio "shape" on tot som "False"
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        #A: x.shape[2] es el nombre de frames de cada gif
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        #A: time_rel_pos_bias es un vector de dimensio [embedding_dim, batch_size, num_frames_de_cada_video]
        
        #A: x es el tensor de dimensions bxcxfxhxw al qual se li ha afegit el sorroll
        x = self.init_conv(x)
        #A: x es tensor de dimensions b x dim x f x h x w (hem especificat que el nombre de canals a la sortida sigui dim)
        #A: per fer-ho, per cada batch, hem utilitzat init_dim filtres diferents de dimensio (3'c', 1'f', 7'h', 7'w') amb un padding de (0,3,3)

        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)
        #A: self.init_temporal_attn es un residual bloc aixi que retorna x=x+x' on x' es la x a la que se li ha aplicat attention
        #A: x continua sent de dimensio b x dim x f x h x w
        r = x.clone()

        #A: time es el tensor de dimensio 1 x batch_size que conte fins a quin timestep s'ha afegit soroll en cadascun dels gifs que composen el batch
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # classifier free guidance

        h = []

        for block1, block2, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            #x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            #x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape #A: el "*_" serveix per trencar l'estructura torch.Size([size]) amb la que et retorna pytorch la mida i quedarte nomes amb el valor size
    out = a.gather(-1, t)
    #A: out es de la mateixa mida que t, es a dir, (1, batch_size) 
    #A: quan cridem extract passant-li d'argument "a" un tensor d'alphas, a va d'1 a un valor proper a 0. 
    #A: out sera un tensor amb els valors d'alpha desordenats
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))#A: aqui creem un tuple d'1s de longitud len(x_shape)-1. Es a dir el que enviem es de dimensio bx1x1x1x1

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64) #A: torch.linspace et crea un tnesor de dimensio steps amb valors equiespaiats entre 0 i timestep ordenats de mes petit a mes gran
    #A: per tant estem agafant beta com una constant que va creixent linealment (en realitat no es linealment perq a les proximes linies se li aplica cos) en cada timestep
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2 #ADUDA: perq aquesta formula? formula 18 WELL EXPLAINED
    #A: ara alphas_cumprod es un tenosor que va d'un valor molt proxim a 1 a 0
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    #A: ara alphas_cumprod es un tenosor que va d'1 a 0
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])#A: el "1:" vol dir que no agafa el primer coeficient del tensor i el ":-1" vol dir que no agafa l'últim coeficient del tensor
    #A: alphas_cumprod era de dimensio timesteps+1 pero al fer [1:] i [:-1] a la linia de dalt, betas es de dimensio timesteps
    #A: (alphas_cumprod[1:] / alphas_cumprod[:-1]) va d'un valor proper a 1 fins a 0. per tant, betas va d'un valor proper a 0 fins a 1
    return torch.clip(betas, 0, 0.9999) #A: fa un clipping de betas entre 0 i 0.9999. Abans l'ultim valor del tensor betas era 1 ara passara a ser 0.9999

class GaussianDiffusionNP(nn.Module):
    def __init__(
        self,
        *,
        height,
        width,
        num_frames,
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000, #A: timesteps es el nombre de vegades que fas froward i backward propagation en l'entrenament d'una xarxa neuronal
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9,
        device = 'cuda:0'
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.device = device

        #A: betas es un tensor de dimensio timesteps+1
        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        #A:alphas va d'un valor proper a 1 fins a 0.0001
        alphas_cumprod = torch.cumprod(alphas, axis=0) #A: cumprod retorna un tensor resultat de multiplicar acumulativament els valors del tensor d'entrada
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)#A: fem padding d'una columna a l'esquerra amb valor 1 i ens olvidem de l'ultima columna
        #A:alphas_cumprod_prev va d'1 fins a un valor proper a 0

        timesteps, = betas.shape #A: per assegurar-se
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32).to(self.device))
        #A: register_buffer es un metode de la classe Module (superclasse de GaussianDiffusion)
        
        #ADUDA: no acabo d'entendre la utilitat del register_buffer
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        #A: formula 10 WELL EXPLAINED
        
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        #A: posterior_mean_coef1 I posterior_mean_coef2 formen part de la formula 11 de WELL EXPLAINED

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, unet, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        #A: el noise que passem com argument al metode predict_start_from_noise es el noise predit per la UNET
        x_recon = self.predict_start_from_noise(x, t=t, noise = unet.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, unet, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(unet = unet, x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, unet, shape, cond = None, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)#A: tensor de dimensions bxcxfxhxw que es tot soroll

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(unet, img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, unet, cond = None, cond_scale = 1., batch_size = 16):
        device = next(unet.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        return self.p_sample_loop(unet, (batch_size, self.channels, self.num_frames, self.height, self.width), cond = cond, cond_scale = cond_scale)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x):
        b = x.shape[0]
        x = normalize_img(x)
        real_noise = torch.randn_like(x)
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        x_noisy = extract(self.sqrt_alphas_cumprod, t, x.shape) * x + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * real_noise
        return x_noisy, t, real_noise


# trainer class

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif
#A: dim tensor = 3x8x4*128x4*128
def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    array = tensor.detach().cpu().numpy()

    # reshape the numpy array to video dimensions
    array = np.transpose(array, (1, 2, 3, 0))
    array = np.uint8(array * 255)

    # create video using OpenCV
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 4, (tensor.shape[3], tensor.shape[2]))
    for i in range(tensor.shape[1]):
        out.write(array[i])
    out.release()
    return 

# gif -> (channels, frame, height, width) tensor



def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    #A: retorna una concatenacio de tots els tensors
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        height,
        width,
        channels = 3,
        num_frames = 16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['avi']
    ):
        super().__init__()
        self.folder = folder
        self.height = height
        self.width = width
        self.channels = channels
        #A: self.paths conte el path cap a cada un dels gifs
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            #T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #T.CenterCrop(image_size),
            #T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    #A: If x is an instance of Dataset, then x[i] is roughly equivalent to type(x).__getitem__(x, i)
    #A: El metode __getitem__ s'executa cada cop que es fa "Dataset[i]" o "for gg in Dataset"
    def __getitem__(self, index):
        path = self.paths[index]
        #A: tensor es un tensor de dimensions Channels x Frames x Height x Weight corresponent al gif del path path
        #tensor = gif_to_tensor(path, self.channels, transform = self.transform)
        tensor=torchvision.io.read_video(str(path), output_format='TCHW')[0].float()/255
        #tensor = torch.stack([self.transform(x) for x in tensor], dim=0)
        tensor = self.transform(tensor)
        tensor=torch.transpose(tensor, 0,1)
        #A: Es retorna el tensor tensor. El sel.cast_... es un metode de control de dimensions del tensors
        return self.cast_num_frames_fn(tensor)

# trainer class

class TrainerNP(object):
    def __init__(
        self,
        unet,
        diffusion,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None,
        loss_type
    ):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion
        #A: EMA=Exponential Moving Average
        self.ema = EMA(ema_decay)
        #A: copia l'estructura, no els parametres
        self.ema_model = copy.deepcopy(self.unet)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.height = diffusion.height
        self.width = diffusion.width
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.loss_type = loss_type

        height = diffusion.height
        width = diffusion.width
        channels = diffusion.channels
        num_frames = diffusion.num_frames

        #A: self.ds es un objecte instancia de la classe Dataset amb metode __getitem__ entre d'altres
        self.ds = Dataset(folder, height, width, channels = channels, num_frames = num_frames)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        #A: si batch_size=None, self.dl es un objecte (instancia de DataLoader) iterable (degut a cycle()) dels tensors corresponents a tots els gifs que l'usuari subministra a l'entrenament. "El format d'aquests tensors s'ha definit a __getitem__ de la classe Dataset" (Entre cometes perq a cycle iterem un objecte de la classe Dataloader, no Dataset. Tot i així hi ha part de cert perq Dataset es argument de Dataloader i, per tant, al iterar sobre DataLoader, en part, "iteres sobre Dataset")
        #A: si batch_size=num, self.dl es un objecte (instancia de DataLoader) iterable (degut a cycle()) de tensors on cada tensor representa un mini batch. Per tant, les dimensions dels tensors son BatchSize x Channels x Frames x Height x Weight
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(self.unet.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        #A: If the forward pass for a particular op has float16 inputs, the backward pass for that op will produce float16 gradients. Gradient values with small magnitudes may not be representable in float16. These values will flush to zero (“underflow”), so the update for the corresponding parameters will be lost. To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and invokes a backward pass on the scaled loss(es). Gradients flowing backward through the network are then scaled by the same factor. In other words, gradient values have a larger magnitude, so they don’t flush to zero.
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        #A: Recordar que self.ema_model es una "copia" (que es copia i que no?=>copia l'estructura pero no els parametres) d'un objecte instancia GaussianModel. Aqui copiem els parametres
        self.ema_model.load_state_dict(self.unet.state_dict()) 
        #A: que es state_dict? https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
        #A: Per tant, en aquest metode ens asegurem que ema_model i model s'inicialitzin amb els mateixos parametres

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()#A: es copien els parametres apresos al model copia "ema_model"
            return
        self.ema.update_model_average(self.ema_model, self.unet)

    def train(self, dev,):
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                x = next(self.dl).to(dev)
                
                x_noisy, t, real_noise = self.diffusion.q_sample(x)
                
                with autocast(enabled = self.amp):
                    predicted_noise = self.unet(x_noisy, t)
                    
                    if self.loss_type == 'l1':
                        loss = F.l1_loss(real_noise, predicted_noise)
                    elif self.loss_type == 'l2':
                        loss = F.mse_loss(real_noise, predicted_noise)
                    else:
                        raise NotImplementedError()

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')
                #A: faig print de l'step i de l'error. com que fem update de parametres tenint en compte dos minibatchs, es printeja 2 cops el mateix step

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)#A: update parameters
            self.scaler.update()#A: s'actualitzen els parametres de tot el tema de l'escalat. Abans s'havia escalat tenint en compte els 2 primers minibatches, ara es tindran en compte els dos seguents
            self.opt.zero_grad()#A: es posen a 0 de nou els gradients

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                torch.save(self.ema_model.state_dict(), str(self.results_folder / f'model-{milestone}.pt'))

            self.step += 1
            
            count_parameters(self.unet)
            #count_parameters(self.ema_model)
            #print('GPU memory used:', torch.cuda.max_memory_allocated(dev))

        print('training completed')
