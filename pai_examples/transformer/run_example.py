"""Run PerforatedAI's transformer example (train.py, unchanged) with PAI settings applied first.

The example has no option for these settings, so they are set here before train.py runs:

    PAI_MODE=pb|gd      pb (default): Perforated Backpropagation / cascade correlation, which PAI
                        enables automatically when perforatedbp is installed. gd: dendrites are
                        trained by gradient descent together with the network (open-source mode).
    PAI_CONTROL=1       control: perforate_model(doing_pai=False), so PAI only restarts the
                        learning rate at each would-be switch and never adds dendrites.
    PAI_MAX_DENDRITES=N cap on dendrite sets (or, for a control, on LR restarts). PAI may stop sooner.
    RUN_SUFFIX=_x       appended to the W&B run name so runs can be told apart.

In PB mode (not a control) every neuron's PB score is also saved after each dendrite-phase
validation, to pai_neurons/set<n>_val<k>.npz (n: dendrite set, k: validation in its dendrite phase).

PAI's configuration is confirmed non-interactively (PAI 4.0 otherwise asks in a terminal before
perforate_model), so this also works under nohup or Slurm. Outputs are written to the current
directory; run each experiment from its own directory. Arguments are passed on to train.py.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # train.py imports its sibling modules

from perforatedai import globals_perforatedai as GPA  # noqa: E402
from perforatedai import utils_perforatedai as UPA  # noqa: E402
import wandb  # noqa: E402

GPA.pc.set_configuration_confirmed(True)
mode = os.environ.get("PAI_MODE", "pb")
if mode == "gd":
    GPA.pc.set_perforated_backpropagation(False)
    GPA.pc.set_no_extra_n_modes(True)
elif mode != "pb":
    sys.exit("PAI_MODE must be pb or gd")
if os.environ.get("PAI_MAX_DENDRITES"):
    GPA.pc.set_max_dendrites(int(os.environ["PAI_MAX_DENDRITES"]))
control = os.environ.get("PAI_CONTROL") == "1"
if control:
    _perforate = UPA.perforate_model
    UPA.perforate_model = lambda model, *a, **k: _perforate(model, *a, **{**k, "doing_pai": False})
if os.environ.get("RUN_SUFFIX"):
    _init = wandb.init
    wandb.init = lambda *a, **k: _init(*a, **{**k, "name": k.get("name", "run") + os.environ["RUN_SUFFIX"]})



def _save_neuron_pb_scores():
    """In PB mode, save every neuron's running-best |Pearson| (PB score) after PAI updates the scores
    at each dendrite-phase validation, as pai_neurons/set<n>_val<k>.npz (n: dendrite set being
    trained, k: validation within its dendrite phase; one array per perforated module, shaped like
    PB's score buffer). PAI's per-module PB score is the max of these values. Same hook as pai_setup.py in the
    ViT code: add_validation_score calls TPB.update_pb_scores before deciding on a switch, and a
    switch discards the per-neuron values. Never raises."""
    try:
        import numpy as np
        from perforatedbp import tracker_pbp as TPB
    except ImportError:
        return
    original, counts = TPB.update_pb_scores, {}

    def update_pb_scores(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            member_vars = GPA.pai_tracker.member_vars
            if member_vars.get("mode") == "p":
                dendrite_set = member_vars.get("num_dendrites_added", 0) + 1
                counts[dendrite_set] = counts.get(dendrite_set, 0) + 1
                # Unflattened: here PB keeps one score per (sequence position, neuron); PAI's per-neuron
                # score is the max over the non-neuron axes (the last axis is the neuron axis here).
                arrays = {m.name.strip("."): m.dendrite_module.dendrite_values[0].best_pearson_correlation
                          .detach().float().abs().cpu().numpy()
                          for m in GPA.pai_tracker.neuron_module_vector}
                os.makedirs("pai_neurons", exist_ok=True)
                np.savez(os.path.join("pai_neurons", f"set{dendrite_set}_val{counts[dendrite_set]}.npz"), **arrays)
        except Exception as e:  # noqa: BLE001
            print(f"[run_example] could not save per-neuron PB scores: {e!r}", flush=True)
        return result

    TPB.update_pb_scores = update_pb_scores


if mode == "pb" and not control:
    _save_neuron_pb_scores()

print(f"[run_example] mode={mode} control={control} "
      f"perforated_backpropagation={GPA.pc.get_perforated_backpropagation()} "
      f"max_dendrites={GPA.pc.get_max_dendrites()}", flush=True)
sys.argv = [os.path.join(HERE, "train.py")] + sys.argv[1:]
runpy.run_path(os.path.join(HERE, "train.py"), run_name="__main__")
