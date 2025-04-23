import torch
import torch.nn as nn
import numpy as np
import lightning as L
from mm_lego.models import SNN, MILAttentionNet
from einops.layers.torch import Reduce
from typing import *
from mm_lego.models.model_utils import *
import math
import torch.nn.functional as F
import ot



class LegoBlock(nn.Module):
    def __init__(self,
                 in_shape: Tuple,
                 num_classes: int=4,
                 name: str = None,
                 encoder: nn.Module = None,
                 freeze: str = None,
                 l_c: int = 64,  # latent channels (num latents)
                 l_d: int= 64,  # latent dimensions
                 depth: int = 2,
                 attn_dropout: float = 0.,
                 ff_dropout: float = 0.,
                 fourier_encode_data: bool = True,
                 fourier_dim: int = 1,
                 final_classifier_head: bool = True,
                 weight_sharing: bool = True,
                 track_imaginary: bool = True,  # track imaginary component of latent
                 normalise: bool = True,
                 frequency_domain: bool = True,
                 **kwargs
                 ):
        super().__init__()
        self.in_shape = in_shape
        self.encoder = encoder
        self.name = name
        self.i_c, self.i_d = in_shape # input channels, input dimensions
        self.track_imaginary = track_imaginary
        self.normalise = normalise
        self.frequency_domain = frequency_domain
        self.l_c = l_c # 
        self.l_d = l_d
        self.depth = depth
        self.num_classes = num_classes
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.fourier_encode_data = fourier_encode_data
        self.fourier_dim = fourier_dim
        self.final_classifier_head = final_classifier_head
        self.weight_sharing = weight_sharing

        self.latent = nn.Parameter(torch.randn(self.l_c, self.l_d))


        # get encoder dims
        if self.encoder is not None:
            # get input features into last layer/head (that's what we'll pass into fusion layer)
            self.i_d = [mod for mod in self.encoder.modules()][-1].in_features

        # define cacheable standard blocks
        # TODO - double check context_dim and mod_attn
        get_cross_attn = lambda: PreNorm(l_d, Attention(query_dim=self.l_d, context_dim=self.i_d, heads=8, dim_head=64, dropout=attn_dropout), context_dim=self.i_d)
        get_cross_ff = lambda: PreNorm(self.l_d, FeedForward(self.l_d, dropout=ff_dropout, snn=True))
        get_mod_attn = lambda: PreNorm(self.i_d, Attention(query_dim=self.i_d, heads=8, dim_head=64, dropout=attn_dropout))
        get_mod_ff = lambda: PreNorm(self.i_d, FeedForward(self.i_d, dropout=ff_dropout, snn=True))


        get_cross_attn, get_cross_ff, get_mod_attn, get_mod_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_mod_attn, get_mod_ff))

        self.layers = nn.ModuleList([])

        for i in range(self.depth):
            should_cache = i > 0 and not self.weight_sharing
            cache_args = {'_cache': should_cache}


            mod_layers = []
            if self.encoder is None:
                mod_layers.append(get_mod_attn(**cache_args))
                mod_layers.append(get_mod_ff(**cache_args))
            else:
                mod_layers.append(self.encoder)
            self.n_mod_layers = len(mod_layers)


            cross_attn_layers = []
            cross_attn_layers.append(get_cross_attn(**cache_args))
            cross_attn_layers.append(get_cross_ff(**cache_args))

            self.layers.append(nn.ModuleList(
                [*mod_layers, *cross_attn_layers]
            ))


            self.to_logits = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(self.l_d),
                nn.Linear(self.l_d, self.num_classes)
            ) if final_classifier_head else nn.Identity()

    def forward(self, x: torch.Tensor, l: nn.Parameter = None, return_embeddings: bool = False, complex: bool = False):

        if type(x) == list:
            x = x[0] # unimodal block only
        b, *_ = x.shape

        if l is not None:
            self.latent = nn.Parameter(l)
            # note: passed in latent form another model expected to already have batch dim
        else:
            l = repeat(self.latent, "n d -> b n d", b = b)


        # own layer pass
        for i in range(self.depth):
            mod_layers = self.layers[i][:self.n_mod_layers]
            fusion_attn, fusion_ff = self.layers[i][self.n_mod_layers:]

            # pass through all modality-specific layers (these could be external encoders if passed in)
            for mod_layer in mod_layers:
                if self.encoder is None:
                    x_enc = mod_layer(x) + x # encode original at each pass
                else:
                    try:
                        x_enc = mod_layer([x])
                    except:
                        x_enc = mod_layer(x)

                    if len(x_enc.shape) == 2: # in case encoders squeeze
                        x_enc = x_enc.unsqueeze(1)

            # fourier transform l and x
            # if self.fourier_dim != "None":
            if self.frequency_domain:
                # normalise
                if self.normalise:
                    l = l / l.norm(dim=self.fourier_dim, keepdim=True)
                l_real = torch.fft.fft(l, dim=self.fourier_dim).real
                l_imag = torch.fft.fft(l, dim=self.fourier_dim).imag
                x_real = torch.fft.fft(x_enc, dim=self.fourier_dim).real
            else:
                l_real = l
                x_real = x_enc
            # fusion state passing (real component only)
            l = fusion_attn(l_real, context=x_real) + l_real
            l = fusion_ff(l) + l

            # skip inverse transform in last iteration (want head to be in fourier domain)
            if i == self.depth - 1:
                break
            # inverse transform otherwise
            # if self.fourier_dim != "None":
            if self.frequency_domain:
                # reconstruct complex tensor and apply inverse fft (of which we pass the real component)
                if self.track_imaginary:
                    l = torch.complex(l, l_imag)
                l = torch.fft.ifft(l, dim=self.fourier_dim)


        if return_embeddings:
            if complex and self.frequency_domain:
                return torch.complex(l, l_imag)
            else:
                return l
        else:
            # check if l is a complex tensor
            if l.dtype == torch.complex64:
                l = l.real
            return self.to_logits(l)

    def _check_args(self, freeze: str, alias: str):
        valid_freeze = [None, "encoder", "all"]
        assert freeze in valid_freeze, f"`freeze` arg must be one of {valid_freeze}"

        assert isinstance(alias, str), "`alias` arg must be a string"


    def freeze(self):
        pass

    def unfreeze(self):
        pass



class PlugHeads(nn.Module):
    def __init__(self, head1: nn.Sequential, head2: nn.Sequential, alpha: float = 0.5, method: str = "slerp"):
        super().__init__()
        self.head1 = head1
        self.head2 = head2
        self.alpha = alpha
        self.method = method

        # check assumpgions
        assert len(head1) == len(head2), "Heads must have the same length"
        for layer1, layer2 in zip(head1, head2):
            assert type(layer1) == type(layer2), "Corresponding layers must be of the same type"

        self.combined_layers = nn.ModuleList()
        for layer1, layer2 in zip(head1, head2):
            if self.method == "linear":
                combined_layer = self._linear_interpol(layer1, layer2, alpha)
            elif self.method == "ot":
                combined_layer = self._optimal_transport_layers(layer1, layer2, alpha)
            elif self.method == "slerp":
                combined_layer = self._slerp_interpol(layer1, layer2, alpha)
            else:
                raise NotImplementedError(f"Head merging method {self.method} not implemented")
            self.combined_layers.append(combined_layer)

    def forward(self, x):
        for layer in self.combined_layers:
            x = layer(x)
        return x


    def _slerp(self, a: torch.Tensor, b: torch.Tensor, alpha: float):
        dot = torch.sum(a * b) / (torch.norm(a) * torch.norm(b))
        dot = torch.clamp(dot, -1.0, 1.0)  # Ensure dot is within the domain of arccos
        theta = torch.acos(dot) * alpha
        relative_vector = b - a * dot
        relative_vector = relative_vector / torch.norm(relative_vector)
        return (a * math.cos(theta)) + (relative_vector * math.sin(theta))


    def _slerp_interpol(self, layer1, layer2, alpha):
        combined_layer = None
        if isinstance(layer1, nn.Linear):
            combined_layer = nn.Linear(layer1.in_features, layer1.out_features)
            combined_layer.weight.data = self._slerp(layer1.weight.data, layer2.weight.data, alpha)
            combined_layer.bias.data = self._slerp(layer1.bias.data, layer2.bias.data, alpha)
        elif isinstance(layer1, nn.LayerNorm):
            combined_layer = nn.LayerNorm(layer1.normalized_shape)
            combined_layer.weight.data = self._slerp(layer1.weight.data, layer2.weight.data, alpha)
            combined_layer.bias.data = self._slerp(layer1.bias.data, layer2.bias.data, alpha)
        elif isinstance(layer1, Reduce):
            combined_layer = layer1  # Reduce ops are identical
        else:
            raise NotImplementedError(f"Layer type {type(layer1)} not supported for combination")

        return combined_layer


    def _linear_interpol(self, layer1, layer2, alpha):
        combined_layer = None
        if isinstance(layer1, nn.Linear):
            combined_layer = nn.Linear(layer1.in_features, layer1.out_features)
            combined_layer.weight.data = alpha * layer1.weight.data + (1 - alpha) * layer2.weight.data
            combined_layer.bias.data = alpha * layer1.bias.data + (1 - alpha) * layer2.bias.data
        elif isinstance(layer1, nn.LayerNorm):
            combined_layer = nn.LayerNorm(layer1.normalized_shape)
            combined_layer.weight.data = alpha * layer1.weight.data + (1 - alpha) * layer2.weight.data
            combined_layer.bias.data = alpha * layer1.bias.data + (1 - alpha) * layer2.bias.data
        elif isinstance(layer1, Reduce):
            combined_layer = layer1  # Reduce ops are identical
        else:
            raise NotImplementedError(f"Layer type {type(layer1)} not supported for combination")
        return combined_layer


    def _optimal_transport_layers(self, layer1, layer2, alpha):
        optimal_layer = None
        if isinstance(layer1, nn.Linear):
            optimal_layer = nn.Linear(layer1.in_features, layer1.out_features)
            optimal_layer.weight.data = self._optimal_transport(layer1.weight.data, layer2.weight.data, alpha)
            optimal_layer.bias.data = self._optimal_transport(layer1.bias.data.unsqueeze(1), layer2.bias.data.unsqueeze(1), alpha).squeeze(1)
        elif isinstance(layer1, nn.LayerNorm):
            optimal_layer = nn.LayerNorm(layer1.normalized_shape)
            optimal_layer.weight.data = self._optimal_transport(layer1.weight.data.unsqueeze(1), layer2.weight.data.unsqueeze(1), alpha).squeeze(1)
            optimal_layer.bias.data = self._optimal_transport(layer1.bias.data.unsqueeze(1), layer2.bias.data.unsqueeze(1), alpha).squeeze(1)
        elif isinstance(layer1, Reduce):
            optimal_layer = layer1  # Assuming Reduce layers are identical
        else:
            raise NotImplementedError(f"Layer type {type(layer1)} not supported for combination")
        return optimal_layer


    def _optimal_transport(self, weights1, weights2, alpha):
        weights1_np = weights1.cpu().detach().numpy()
        weights2_np = weights2.cpu().detach().numpy()

        # cost matrix
        M = ot.dist(weights1_np, weights2_np, metric='euclidean')

        transport_plan = ot.emd(np.ones(weights1_np.shape[0]), np.ones(weights2_np.shape[0]), M)

        combined_weights = (1 - alpha) * weights1_np + alpha * np.dot(transport_plan, weights2_np)

        combined_weights = torch.tensor(combined_weights, dtype=weights1.dtype, device=weights1.device)
        return combined_weights


# def norm(tensor: torch.Tensor, dim: int = -1):
#     return tensor / tensor.norm(dim=dim, keepdim=True
class FFTLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.fft.fft(x, dim=self.dim).real

class IFFTLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.fft.ifft(x, dim=self.dim).real

class LegoBase(nn.Module):
    def __init__(self,
                 blocks: List[LegoBlock],
                 head_method: str = "slerp",
                 alpha: float = 0.5,
                 ):
        super().__init__()
        self.blocks = blocks
        self.modalities = len(blocks)
        self.head_method = head_method
        self.alpha = alpha

        self.layers = nn.ModuleList([b for b in blocks])

        self.to_logits = PlugHeads(head1=blocks[0].to_logits, head2=blocks[1].to_logits, alpha=alpha, method=self.head_method)


        self.l_c, self.l_d = blocks[0].l_c, blocks[0].l_d
        self.L = nn.Parameter(torch.randn(self.l_c, self.l_d))

    def _check_inputs(self):
        # check that blocks have equal latent dims
        ref_d, ref_c = self.blocks[0].l_d, self.blocks[0].l_c
        for block in self.blocks:
            assert block.l_d == ref_d, "All blocks must have equal latent dimensions"
            assert block.l_c == ref_c, "All blocks must have equal latent channels"

    def forward(self, x: List[torch.Tensor], return_embeddings: bool = False):
        pass


class LegoFuse(LegoBase):
    def __init__(self,
                 blocks: List[LegoBlock],
                 depth: int = 2,
                 fuse_method: str = "stack",
                 head_method: str = "slerp",
                 **kwargs
                 ):

        super().__init__(blocks, head_method)
        self.depth = depth
        self.method = fuse_method

    def forward(self, x: List[torch.Tensor], return_embeddings: bool = False):
        b, *_ = x[0].shape

        if len(self.L.shape) == 2:
            l = repeat(self.L, "n d -> b n d", b=b)

        if self.method == "stack":
            for i, block in enumerate(self.layers):
                if i == 0:
                    l = block(x[i], return_embeddings=True)
                else:
                    l = block(x[i], l=l, return_embeddings=True)


        elif self.method == "weave":
            for j in range(self.depth):
                for i, block in enumerate(self.layers):
                    # print(f"Applying single pass from block {block.name}")
                    mod_layers = block.layers[0][:block.n_mod_layers]
                    fusion_attn, fusion_ff = block.layers[0][block.n_mod_layers:]
                    for mod_layer in mod_layers:
                        x[i] = mod_layer(x[i]) + x[i]
                    # fusion state passing
                    l = fusion_attn(l, context=x[i]) + l
                    l = fusion_ff(l) + l
        else:
            raise NotImplementedError(f"Merge method {self.method} not implemented")

        if return_embeddings:
            return l
        else:
            return self.to_logits(l)



class LegoMerge(LegoBase):

    def __init__(self,
                 blocks: List[LegoBlock],
                 merge_method: str = "sum",
                 head_method: str = "slerp",
                 alpha: float = 0.8,
                 **kwargs
                 ):
        super().__init__(blocks, head_method, alpha=alpha)
        self.method = merge_method

    def forward(self, x: List[torch.Tensor], return_embeddings: bool = False, complex: bool = False):


        # take the sum of latents in fourier domain before passing to head
        latents = []
        for i, block in enumerate(self.layers):
            # l = block(x[i], return_embeddings=True, complex=True)
            l = block(x[i], return_embeddings=True, complex=True)
            latents.append(l)
        if self.method == "sum":
            l = sum(latents)
        elif self.method == "product":
            l = torch.prod(torch.stack(latents), dim=0)
        elif self.method == "mean":
            l = torch.mean(torch.stack(latents), dim=0)
        elif self.method == "harmonic":
            l1 = latents[0]
            l2 = latents[1]
            # magnitudes and phases
            mag_l1 = torch.abs(l1)
            mag_l2 = torch.abs(l2)
            phase_l1 = torch.angle(l1)
            phase_l2 = torch.angle(l2)

            # weighted harmonic mean
            mag = 2 * self.alpha * (1 - self.alpha) * mag_l1 * mag_l2 / (self.alpha * mag_l2 + (1 - self.alpha) * mag_l1)
            phase = (phase_l1 + phase_l2) / 2

            # combine into complex array
            l = mag * torch.exp(1j * phase)
        else:
            raise NotImplementedError(f"Merge method {self.method} not implemented")


        if return_embeddings:
            if complex:
                return l
            else:
                return l.real

        else:
            return self.to_logits(l.real)


if __name__ == "__main__":
    b = 10
    # tabular
    # Note - dimensions always denoted as
    t_c = 1  # number of channels (1 for tabular) ; note that channels correspond to modality input/features
    t_d = 2189  # dimensions of each channel
    i_c = 100  # number of patches
    i_d = 1024  # dimensions per patch
    l_c = 256  # number of latent channels (num_latents)
    l_d = 32  # latent dims
    # latent_dim
    query = torch.randn(b, t_c, t_d)
    latent = torch.randn(b, l_c, l_d)

    tabular_data = torch.randn(b, t_c, t_d)  # expects (b dims channels)
    img_data = torch.randn(b, i_c, i_d)

    tab_enc = SNN(t_d, final_head=False)
    img_enc = MILAttentionNet(torch.Size((i_c, i_d)), final_head=False, size_arg="tcga")

    tab_block = LegoBlock(in_shape=(t_c, t_d), encoder=tab_enc)
    img_block = LegoBlock(in_shape=(i_c, i_d), encoder=img_enc)


    out= img_enc([img_data])
    print(out.shape)
    # tab_block = LegoBlock(in_shape=(t_c, t_d), name="tab")
    # img_block = LegoBlock(in_shape=(i_c, i_d), name = "img")
    #
    # tab_l = tab_block(tabular_data, return_embeddings=True)
    img_block([img_data])
    # tab_block([tabular_data])
    #
    # merge = LegoMerge(blocks=[tab_block, img_block])
    # merge_logits = merge([tabular_data, img_data])
    # print(merge_logits)

    # fuse = LegoFuse(blocks=[tab_block, img_block], depth=2)
    # fuse_logits = fuse([tabular_data, img_data])


    # print(fuse_logits.shape)
    # print(tab_block)
    # print(fuse)
    # fuse_latent = fuse([tabular_data, img_data], return_embeddings=True)
    # print(fuse_latent)
    """
    Documentation
    
    # train from scratch
    tab_block = LegoBlock(in_shape=(t_d, t_c), encoder=None)
    img_block = LegoBlock(in_shape=(i_c, i_d), encoder=None)
    
    
    # use own encoder, use as adapter
    snn = SNN(input_dim=t_d, final_head=False)
    abmil = MILAttentionNet(input_dim=i_d, final_head=False, size_arg="tcga")
    tab_block = LegoBlock(in_shape=(t_d, t_c), encoder=snn) # may rename to LegoAdapter
    
    
    # if none, use own encoder
    blocks = [tab_block, img_block]
    
    model = LegoFuse(blocks)
    model = LegoMerge(blocks)
    
    
    """


