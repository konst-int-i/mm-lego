# Some wandb cleanup utils
import subprocess
from pathlib import Path
import shutil
import wandb
from mm_lego.utils import setup_logging

logger = setup_logging()

def _teardown_wandb_sweeps(org: str = "camb-explain", project: str = "mm-lego"):
    print("Terminating unfinished sweeps...")
    api = wandb.Api()
    project = api.project(project)
    sweeps = project.sweeps()
    for sweep in sweeps:
        id = sweep.id
        # get sweep
        sweep = api.sweep(f"mm-lego/{id}") # note: somehow errors when using {project} argument
        # get status
        status = sweep.state
        if status not in ["CANCELED", "FINISHED"]:
            subprocess.run(f"/home/kh701/mambaforge/envs/castle/bin/wandb sweep --cancel camb-explain/mm-lego/{id}", shell=True)

def _get_sweeps_from_stamp(target_stamp: str):
    api = wandb.Api()
    project = api.project("mm-lego")
    sweeps = project.sweeps()
    target_sweeps = [sweep for sweep in sweeps if target_stamp in sweep.name]
    return target_sweeps


def _get_best_model_from_sweep(sweep_id: str) -> tuple:
    """
    For a given sweep id, get the best model based on the valuation metric (e.g., test c-index).
    Args:
        sweep_id (str):

    Returns:
        Tuple: best_run (wandb run object), other_runs (list of wandb run objects)
    """
    api = wandb.Api()
    sweep = api.sweep(f"mm-lego/{sweep_id}")
    best_metric = 0.0
    best_run = None
    other_runs = []
    run_ids = [run.id for run in sweep.runs]
    for id in run_ids:
        run = api.run(f"mm-lego/{id}")
        metric_value = run.summary.get('mean_test_c_index')
        # print(metric_value)
        if metric_value is not None and metric_value > best_metric:  # > as higher metrics are preferred (may need changing)
            best_metric = metric_value
            best_run = run
        else:
            other_runs.append(run)
    return best_run, other_runs


def _cleanup_model_log(sweep_id: str) -> None:
    """
    Removes all models logs except the best from a given sweep.
    Args:
        sweep_id (str): wandb sweep id

    Returns:
        None, but clearnups `model_log` directory
    """
    # sweeps = _get_sweeps_from_stamp(baseline_stamp)
    # for sweep in sweeps:
    logger.info(f"Cleaning up model_log for sweep {sweep_id}...")
    best_run, other_runs = _get_best_model_from_sweep(sweep_id)
    # delete other runs from scratch store
    logger.info(f"Deleting runs {[r.name for r in other_runs]}")
    for run in other_runs:
        del_path = Path(f"model_log/{run.name}")
        if del_path.exists():
            logger.info(f"Deleting {del_path}...")
            shutil.rmtree(del_path)
    return None
