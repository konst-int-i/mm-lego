"""
Some smoke tests for baseline shapes
"""

import pytest
from mm_lego.models import HealNet, Attention, MCAT
from mm_lego.models.baselines.mcat import MILAttentionNet, SNN
import torch

@pytest.fixture(scope="module")
def vars():
    b = 10
    # tabular
    # Note - dimensions always denoted as
    t_c = 1  # number of channels (1 for tabular) ; note that channels correspond to modality input/features
    t_d = 2189  # dimensions of each channel
    i_c = 100  # number of patches
    i_d = 2048  # dimensions per patch
    l_c = 256  # number of latent channels (num_latents)
    l_d = 32  # latent dims
    # latent_dim
    query = torch.randn(b, t_c, t_d)
    latent = torch.randn(b, l_c, l_d)

    tabular_data = torch.randn(b, t_d, t_c)  # expects (b dims channels)
    image_data = torch.randn(b, i_c, i_d)
    return b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, tabular_data, image_data


def test_healnet(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, tabular_data, image_data = vars

    # unimodal case smoke test
    m1 = HealNet(modalities=1,
                 input_channels=[t_c],
                 input_axes=[1],  # second axis
                 num_classes=5
                 )
    logits1 = m1([tabular_data])
    assert logits1.shape == (b, 5)

    # bi-modal case
    m2 = HealNet(modalities=2,
                 input_channels=[t_c, i_c],  # level of attention
                 input_axes=[1, 1],
                 )
    logits2 = m2([tabular_data, image_data])
    assert logits2.shape == (b, 4) # default num_classes

    # check misaligned args
    with pytest.raises(AssertionError):
        m3 = HealNet(modalities=1,
                     input_channels=[t_c, i_c],  # level of attention
                     input_axes=[1, 1],
                     )

def test_cross_attention(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, _, _ = vars
    # attention - works on final dim in tensor (l_d and t_d)
    # This gives attention matrix for the latent channels l_c
    attention = Attention(query_dim=l_d, context_dim=t_d)
    # NOTE - traditional attention expects the latent as the query and returns the latent
    # Problem is that this also means that the attention-matrix is at the latent level
    updated_latent = attention(x=latent, context=query)

    assert updated_latent.shape == (b, l_c, l_d)


def test_self_attention(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, _, _ = vars
    attention = Attention(query_dim=l_d)

    update = attention(x=latent)

    # Updated latent and original should have the same shape
    assert latent.shape == update.shape


def test_mcat(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, tab_tensor, img_tensor = vars

    m1 = MCAT(n_classes=4,
              omic_shape=tab_tensor.squeeze().shape[1:], # note: expects [t_d], not expanded [t_c, t_d], so need to squeeze
              wsi_shape = img_tensor.shape[1:]
              )
    logits2 = m1([tab_tensor, img_tensor])
    assert logits2.shape == (b, 4)



def test_mil_attention(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, tab_tensor, img_tensor = vars

    img_tensor = torch.randn(size=(b, i_c, i_d))

    model = MILAttentionNet(
        input_dim=torch.Size((i_c, i_d)),
        n_classes=4
    )
    logits = model(data=[img_tensor])
    assert logits.shape == (b, 4)


def test_snn(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, _, _ = vars

    tab_tensor = torch.randn(size=(b, t_c, t_d))

    model = SNN(
        input_dim=t_d,
        n_classes=4
    )
    logits = model(data=[tab_tensor])
    # print(logits.shape)
    assert logits.shape == (b, 4)

