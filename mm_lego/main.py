"""
Run all experiments
"""
import wandb
from datetime import datetime
import pandas as pd
import os
from typing import *
from pathlib import Path
from box import Box
from mm_lego.pipeline import Fusion
from mm_lego.utils import Config
from sklearn.model_selection import ParameterGrid
import argparse
import multiprocessing

from mm_lego.utils.wandb import _teardown_wandb_sweeps, _cleanup_model_log


def run(config):


    pipeline = Fusion(config=config)
    source = pipeline.config.source
    n_mods = len(pipeline.config.data[source].modalities)
    print(f"Running {pipeline.config.model} on {source} with {n_mods} modalities")
    if pipeline.config.model in ["legofuse", "legomerge", "legomerge-tune"] and n_mods > 1:
        print("Running lego pipeline")
        pipeline.run_lego()
    else:
        print("Running standard pipeline")
        pipeline.run()



def merge_ablation(config):

    config.model = "legomerge"
    config.model_params.lego.use_cache = False
    config.model_params.lego.use_final = False

    exp = [
        {"track_imaginary": False, "frequency_domain": False},
        {"track_imaginary": True, "frequency_domain": True},
        {"track_imaginary": False, "frequency_domain": True},
    ]

    result_df = pd.DataFrame()
    for p in exp:
        config.model_params.lego.track_imaginary = p["track_imaginary"]
        config.model_params.lego.frequency_domain = p["frequency_domain"]
        pipeline = Fusion(config)
        _, c_index = pipeline.run_lego()
        result_df = pd.concat([result_df, pd.DataFrame([{"track_imaginary": p["track_imaginary"], "frequency_domain": p["frequency_domain"], "c_index": c_index}])], ignore_index=True)
    print(result_df)
    result_df.to_csv("result_log/final/merge_ablation.csv")


def mini_experiments(config):
    """
    Wrapper function for small experiments during development
    """

    # create parameter grid
    # grid = ParameterGrid({
    #     "dataset": ["mimic-mortality"],
    #     "model": ["snn", "snn-block", "amil", "amil-block"],
    #     # launch a sweep for each?
    # })
    # datasets = [ "tcga-kirp", "tcga-ucec", "tcga-blca", "tcga-brca", "isic-isic", "mimic-mortality", "mimic-icd9"]
    # datasets = ["mimic-mortality", "mimic-icd9", "isic-isic"]
    # datasets = ["isic-isic"]
    # datasets = ["tcga-kirp", "tcga-ucec", "tcga-blca", "tcga-brca"]
    datasets = ["mimic-mortality", "mimic-icd9"]
    # datasets = ["tcga-blca"]
    # datasets = ["mimic-mortality", "mimic-icd9", "tcga-blca", "tcga-brca", "tcga-kirp", "tcga-ucec", "isic-isic"]
    # models = ["snn", "snn-block", "amil", "amil-block"]
    models = ["snn-block", "snn",  "amil", "amil-block"]

    result_df = pd.DataFrame()

    for d in datasets:

        blocks = []
        for m in models:
            source = d.split("-")[0]
            dataset = d.split("-")[1]
            if source == "mimic":
                config.loader.batch_size = 512
                config.model_params.lego.depth = 1 # low depth for high N
            elif source == "isic":
                config.loader.batch_size = 128
            else:
                config.loader.batch_size = 64
                # high depth for low N
                config.model_params.lego.depth = 2

            config.wandb=False
            config.model = m
            config.source = source
            config.data[source].dataset = dataset
            pipeline = Fusion(config=config)

            print(f"Running {m} on {dataset}")
            model, test_c_index = pipeline.run()

            result_df = pd.concat([result_df, pd.DataFrame([{"model": m, "dataset": dataset, "c_index": test_c_index}])], ignore_index=True)

            if m in ["snn-block", "amil-block"]:
                blocks.append(model)
        print(f"Blocks: {blocks}")

        # run merge
        config.model = "legomerge"
        if source in ["tcga", "isic"]:
            config.data[source].modalities = ["tab", "img"]
        else:
            config.data[source].modalities = ["tab", "ts"]
        pipeline = Fusion(config=config)
        model, test_c_index = pipeline.run_lego(blocks=blocks)
        result_df = pd.concat([result_df, pd.DataFrame([{"model": config.model, "dataset": dataset, "c_index": test_c_index}])], ignore_index=True)

        # run merge with tuning
        config.model = "legomerge-tune"
        pipeline = Fusion(config=config)
        model, test_c_index = pipeline.run_lego(blocks=blocks)
        result_df = pd.concat([result_df, pd.DataFrame([{"model": config.model, "dataset": dataset, "c_index": test_c_index}])], ignore_index=True)

        # result_df.to_csv("result_log/merge_uplift.csv")

    print(result_df)




def run_plan(config):
    """
    The great plan
    """
    # run directory for model log
    sweep_stamp = f"{datetime.now().strftime('%y%m%d_%H%M%S')}"

    mm_grid = ParameterGrid({
        "model": ["legomerge"],
        "dataset": ["mimic-mortality", "tcga-brca", "tcga-blca", "tcga-ucec", "tcga-kirp"],
        "mods": ["mm"]
    })

    # img_grid = ParameterGrid({
    #     "model": ["legoblock"],
    #     # "dataset": ["tcga-blca"],
    #     "dataset": ["tcga-brca", "tcga-blca", "mimic-icd9", "tcga-ucec", "tcga-kirp", "mimic-mortality", "isic-isic"],
    #     "mods": ["img"]
    # })
    # # #
    # tab_grid = ParameterGrid({
    #     "model": ["legoblock"],
    #     "dataset": ["tcga-brca", "tcga-blca", "mimic-icd9", "tcga-ucec", "tcga-kirp", "mimic-mortality", "isic-isic"],
    #     "mods": ["tab"],
    # })


    # check that there are no active runs
    _teardown_wandb_sweeps()

    # num_workers = int(multiprocessing.cpu_count()//2)
    num_workers = 1
    # num_workers = 3
    print(f"Running sweeps with {num_workers} workers")
    sweep_results = {}

    with multiprocessing.Pool(processes=num_workers) as pool:

        # tab-only sweeps
        # check if variable is defined
        if "tab_grid" in locals():
            for result in pool.imap_unordered(_run_sweep, [(sweep_stamp,params, config) for params in tab_grid]):
                pass

        # image-only sweeps
        if "img_grid" in locals():
            for result in pool.imap_unordered(_run_sweep, [(sweep_stamp,params, config) for params in img_grid]):
                pass
        # multimodal sweeps

        if "mm_grid" in locals():
            for result in pool.imap_unordered(_run_sweep, [(sweep_stamp, params, config) for params in mm_grid]):
                pass

    # collect sweep IDs
    print(f"FINISHED SWEEPS with sweep stamp: {sweep_stamp}")
    pool.close()
    pool.join()



def _run_sweep(args):
    sweep_stamp, params, config = args

    source, dataset = params["dataset"].split("-")

    sweep_name = f"{params['model']}_{dataset}_{params['mods']}"

    # read sweep config as dict
    if params['model'] == "legomerge":
        sweep_config = Config("config/final_merge_sweep.yml").read()
    else:
        sweep_config = Config(f"config/{source}_sweep.yml").read()
    sweep_config["name"] = sweep_stamp + "_" + sweep_name
    if params["model"] in ["legoblock"]:
        if params["mods"] == "tab":
            sweep_config["run_cap"] = 20
        else:
            sweep_config["run_cap"] = 10
    elif params["model"] in ["legomerge", "legomerge-tune"]:
        sweep_config["run_cap"] = 20

    # read in model-specific sweep params (if exists)
    model_sweep_config = Path(f"config/{params['model']}_sweep_params.yml")
    if os.path.exists(model_sweep_config):
            model_config = Config(model_sweep_config).read()
            if params['model'] in ["legofuse", "legoblock", "legomerge-tune"]: # all using 'lego' in config
                sweep_config = _assign_nested(sweep_config, ["parameters", "model_params", "parameters", "lego"], model_config)
            elif params['model'] in ["legomerge"]:
                pass
            else:
                sweep_config = _assign_nested(sweep_config, ["parameters", "model_params", "parameters", params['model']], model_config)

    # set fixed values in sweep config
    # note that these will later update the regular config in the pipeline
    sweep_config = _assign_nested(sweep_config, ["parameters", "model", "value"], params["model"])
    sweep_config = _assign_nested(sweep_config, ["parameters", "source", "value"], source)
    sweep_config = _assign_nested(sweep_config, ["parameters", "data", "parameters", "tcga", "parameters", "dataset", "value"], dataset)

    # need to assign modalities to ensure correct pipeline is called
    if params['mods'] == "img":
        if source in ["tcga", "isic"]:
            sweep_config = _assign_nested(sweep_config, ["parameters", "data", "parameters", "tcga", "parameters", "modalities", "value"], ["img"])
        elif source == "mimic":
            sweep_config = _assign_nested(sweep_config, ["parameters", "data", "parameters", "mimic", "parameters", "modalities", "value"], ["ts"])
    elif params['mods'] == "tab":
            sweep_config = _assign_nested(sweep_config, ["parameters", "data", "parameters", f"{source}", "parameters", "modalities", "value"], ["tab"])
    else:
        if source in ["tcga", "isic"]:
            sweep_config = _assign_nested(sweep_config, ["parameters", "data", "parameters", "tcga", "parameters", "modalities", "value"], ["tab", "img"])
        elif source == "mimic":
            sweep_config = _assign_nested(sweep_config, ["parameters", "data", "parameters", "mimic", "parameters", "modalities", "value"], ["tab", "ts"])


    sweep_id = wandb.sweep(sweep=sweep_config.to_dict(), project="mm-lego")
    wandb.agent(sweep_id, project="mm-lego")

    # run
    # os.system(f"/home/kh701/mambaforge/envs/castle/bin/wandb sweep --stop mm-lego/{sweep_id}")
    # once done, cleanup suboptimal runs to avoid clutter
    _cleanup_model_log(sweep_id)

    return True


def _assign_nested(config: Config, keys: List[str], value: Any):
    """
    Recurvsively assigns nested dict to box config
    """
    if len(keys) == 1:
        config[keys[0]] = value
    else:
        key = keys.pop(0)
        if key not in config:
            config[key] = {}
        _assign_nested(config[key], keys, value)
    return config


if __name__ == "__main__":
    wandb.login()

    # parse some args
    parser = argparse.ArgumentParser(description="Which types of experiments to run")
    parser.add_argument("--run", action="store_true", help="Run single experiment")
    parser.add_argument("--mini", action="store_true", help="Run mini experiment grid")
    parser.add_argument("--plan", action="store_true", help="Run the grand plan")
    parser.add_argument("--ablation", action="store_true", help="Run merge ablation experiment")
    parser.add_argument("--config", type=str, default="config_dev.yml", help="Name of config file ")
    args = parser.parse_args()

    if args.plan:
        config = Config(f"config/config_prod.yml").read()
    else:
        config = Config(f"config/{args.config}").read()
    if args.run:
        run(config)
    elif args.mini:
        mini_experiments(config)
    elif args.ablation:
        merge_ablation(config)
    elif args.plan:
        run_plan(config)

