from mm_lego.models.model_utils import *
from math import pi, log
from functools import wraps
from typing import *
from einops.layers.torch import Reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

class HealNet(nn.Module):
    def __init__(
        self,
        *,
        modalities: int,
        num_freq_bands: int = 2,
        depth: int = 3,
        max_freq: float=2,
        input_channels: List,
        input_axes: List,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        num_classes: int = 4,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        fourier_encode_data: bool = True,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        snn: bool = True,
    ):
        super().__init__()
        assert len(input_channels) == len(input_axes), 'input channels and input axis must be of the same length'
        assert len(input_axes) == modalities, 'input axis must be of the same length as the number of modalities'

        self.input_axes = input_axes
        self.input_channels=input_channels
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = modalities
        self.self_per_cross_attn = self_per_cross_attn

        self.fourier_encode_data = fourier_encode_data

        # get fourier channels and input dims for each modality
        fourier_channels = []
        input_dims = []
        for axis in input_axes:
            fourier_channels.append((axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0)
        for f_channels, i_channels in zip(fourier_channels, input_channels):
            input_dims.append(f_channels + i_channels)


        # initialise shared latent bottleneck
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # modality-specific attention layers
        funcs = []
        for m in range(modalities):
            funcs.append(lambda m=m: PreNorm(latent_dim, Attention(latent_dim, input_dims[m], heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dims[m]))
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout, snn = snn))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout, snn = snn))

        get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])


        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(get_latent_attn(**cache_args, key = block_ind))
                self_attns.append(get_latent_ff(**cache_args, key = block_ind))


            cross_attn_layers = []
            for j in range(modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))


            self.layers.append(nn.ModuleList(
                [*cross_attn_layers, self_attns])
            )

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()


    # def _handle_missing(self, tensors: List[torch.Tensor]):



    def forward(self,
                tensors: List[torch.Tensor],
                mask: Optional[torch.Tensor] = None,
                missing: Optional[torch.Tensor] = None,
                return_embeddings: bool = False
                ):

        for i in range(len(tensors)):
            data = tensors[i]
            # sanity checks
            b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
            assert len(axis) == self.input_axes[i], (f'input data for modality {i+1} must hav'
                                                          f' the same number of axis as the input axis parameter')

            # fourier encode for each modality
            if self.fourier_encode_data:
                pos = torch.linspace(0, 1, axis[0], device = device, dtype = dtype)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, 'n d -> () n d')
                enc_pos = repeat(enc_pos, '() n d -> b n d', b = b)
                data = torch.cat((data, enc_pos), dim = -1)

            # concat and flatten axis for each modality
            data = rearrange(data, 'b ... d -> b (...) d')
            tensors[i] = data


        x = repeat(self.latents, 'n d -> b n d', b = b) # note: batch dim should be identical across modalities

        for layer in self.layers:
            for i in range(self.modalities):
                cross_attn= layer[i*2]
                cross_ff = layer[(i*2)+1]
                try:
                    x = cross_attn(x, context = tensors[i], mask = mask) + x
                    x =  cross_ff(x) + x
                except:
                    pass

            if self.self_per_cross_attn > 0:
                self_attn, self_ff = layer[-1]

                x = self_attn(x) + x
                x = self_ff(x) + x

        if return_embeddings:
            return x

        return self.to_logits(x)

    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Helper function which returns all attention weights for all attention layers in the model
        Returns:
            all_attn_weights: list of attention weights for each attention layer
        """
        all_attn_weights = []
        for module in self.modules():
            if isinstance(module, Attention):
                all_attn_weights.append(module.attn_weights)
        return all_attn_weights


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

    tabular_data = torch.randn(b, t_d, t_c)  # expects (b dims channels)
    image_data = torch.randn(b, i_c, i_d)

    # unimodal case
    m1 = HealNet(modalities=1,
                          input_channels=[t_d],
                          input_axes=[1], # second axis
                          )
    # bi-modal case
    m2 = HealNet(modalities=2,
                 input_channels=[t_c, i_c], # level of attention
                 input_axes = [1,1],
                 )


    logits1 = m1([tabular_data])
    logits2 = m2([tabular_data, image_data])



    print(logits1)
    print(logits2)

