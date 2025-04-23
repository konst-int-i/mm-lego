import pytest
from mm_lego.utils import Config
from mm_lego.pipeline import Fusion

@pytest.fixture
def args():

    test_config = {
        "folds": 1,
        "model": "healnet",
        "wandb": False,
        }
    config = Config("config/config_dev.yml").read()
    config.update(test_config)
    config.epochs = 1 # need to update this way due to nesting
    config.data.tcga.modalities = ["tab", "img"]
    return config

def test_fusion(args):

    config = args
    
    # check that the right n_modalities and config is passed with the models
    config.model = "mcat"
    f1 = Fusion(config, source="tcga")
    assert f1.n_modalities == 2
    assert f1.config.data.tcga.modalities == ["tab", "img"]
    
    config.model = "snn"
    f2 = Fusion(config, source="tcga")
    assert f2.n_modalities == 1
    assert f2.config.data.tcga.modalities == ["tab"]

    config.model = "amil"
    f3 = Fusion(config, source="tcga")
    assert f3.n_modalities == 1
    assert f3.config.data.tcga.modalities == ["img"]

