"""PerforatedAI (Perforated Backpropagation) support for main.py.

Fine-tunes a trained SimpleVisionTransformer with dendrites on selected encoder MLPs.
PAI is imported only when --pai-layers is set, so baseline runs need neither the package
nor PAI credentials. PAI reads PAIEMAIL and PAITOKEN from the environment.

PAI writes its graphs, CSVs and model saves to ./pai, so main.py changes into the run's
log directory before calling perforate().
"""
import pdb

import torch
import torch.nn as nn


GPA = UPA = None


def _no_pdb(*args, **kwargs):
    # PAI calls pdb.set_trace() on some errors. Batch jobs have no stdin, so fail loudly instead.
    raise RuntimeError("PerforatedAI called pdb.set_trace(); see traceback")


def _import_pai():
    # Imported on first use so baseline runs never load PAI.
    global GPA, UPA
    pdb.set_trace = _no_pdb
    from perforatedai import globals_perforatedai
    from perforatedai import utils_perforatedai
    GPA, UPA = globals_perforatedai, utils_perforatedai

# Submodule of torchvision's MLPBlock to perforate: Linear(D, 4D), GELU, Dropout, Linear(4D, D), Dropout.
# 'ln_mlp' wraps the pre-norm ln_2 together with the whole MLP.
TARGETS = {'mlp': None, 'ln_mlp': None, 'fc1': 0, 'fc2': 3}


def add_arguments(parser):
    group = parser.add_argument_group('PerforatedAI')
    group.add_argument('--pai-layers', nargs='*', type=int, default=None, metavar='I',
                       help='0-indexed encoder layers whose MLP gets dendrites, e.g. 3 7 11. '
                            'Enables PAI fine-tuning; requires --pai-init.')
    group.add_argument('--pai-target', default='mlp', choices=sorted(TARGETS),
                       help='perforate the whole MLP block, the MLP with its pre-norm ln_2, '
                            'or only its first/second Linear')
    group.add_argument('--pai-control', action='store_true',
                       help='same wrapping, tracker and schedule, but never add dendrites')
    group.add_argument('--pai-init', default='', type=str, metavar='PATH',
                       help='trained checkpoint (main.py format) to fine-tune from')
    group.add_argument('--max-dendrites', default=1, type=int,
                       help='maximum dendrite rounds PAI may add')
    group.add_argument('--pai-n-epochs-to-switch', default=10, type=int,
                       help='minimum validations in a neuron phase before PAI may switch')
    group.add_argument('--pai-p-epochs-to-switch', default=2, type=int,
                       help='minimum validations in a dendrite phase before PAI may switch')
    group.add_argument('--pai-initial-correlation-batches', default=None, type=int,
                       help="training batches before PAI scores dendrites (PAI default 100); "
                            "must be below --log-steps")
    group.add_argument('--pai-patience', default=2, type=int,
                       help='ReduceLROnPlateau patience, in validations')


def check_args(parser, args):
    args.pai = args.pai_layers is not None
    if not args.pai:
        return
    if not args.pai_layers:
        parser.error('--pai-layers needs at least one layer index')
    if not args.pai_init:
        parser.error('--pai-layers requires --pai-init')
    if not all(0 <= i < args.num_layers for i in args.pai_layers):
        parser.error(f'--pai-layers must be in [0, {args.num_layers})')
    if args.distributed or args.ngpus_per_node > 1:
        parser.error('PAI runs support a single GPU only; pass --gpu 0')
    if args.resume:
        parser.error('--resume is not supported with PAI runs')
    if args.accum_freq != 1:
        # PAI stores one set of dendrite outputs and neuron errors per optimizer step.
        parser.error('PAI runs need --accum-freq 1')


def load_init(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> fine-tuning from '{path}' (step {checkpoint['step']}, best acc@1 {checkpoint['best_acc1']:.4f})")


def perforate(model, args):
    _import_pai()
    for i in args.pai_layers:
        block = model.encoder.layers[i]
        mlp = block.mlp
        index = TARGETS[args.pai_target]
        if args.pai_target == 'ln_mlp':
            # forward() computes mlp(ln_2(x)), so moving ln_2 inside the wrapper keeps the math.
            block.mlp = GPA.PAISequential([block.ln_2, mlp])
            block.ln_2 = nn.Identity()
        elif index is None:
            block.mlp = GPA.PAISequential([mlp])
        else:
            mlp[index] = GPA.PAISequential([mlp[index]])

    pc = GPA.pc
    pc.set_testing_dendrite_capacity(False)
    pc.set_max_dendrites(args.max_dendrites)
    pc.set_n_epochs_to_switch(args.pai_n_epochs_to_switch)
    pc.set_p_epochs_to_switch(args.pai_p_epochs_to_switch)
    if args.pai_initial_correlation_batches is not None:
        pc.set_initial_correlation_batches(args.pai_initial_correlation_batches)
    # Dendrites only on the wrapped targets. Everything else is tracked as a whole module, so
    # PAI never wraps the Linear that nn.MultiheadAttention uses internally.
    pc.set_modules_to_perforate([GPA.PAISequential])
    pc.set_module_names_to_perforate(['PAISequential'])
    pc.set_modules_to_track([nn.MultiheadAttention, nn.LayerNorm, nn.Conv2d, nn.Linear])
    # Encoder activations are [batch, tokens, channels]; neurons are the last dimension.
    pc.set_output_dimensions([-1, -1, 0])
    pc.set_weight_decay_accepted(True)
    pc.set_unwrapped_modules_confirmed(True)
    pc.set_configuration_confirmed(True)  # targets are set above; skip PAI's interactive picker

    return UPA.perforate_model(model, doing_pai=not args.pai_control, save_name='pai', maximizing_score=True)


def setup_optimizer(model, args):
    # PAI builds the optimizer itself so it can swap to dendrite-only parameters in dendrite
    # phases. PAI steps the scheduler from validation scores; main.py must not step it.
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    opt_args = {'params': model.parameters(), 'lr': args.lr, 'betas': (args.beta1, args.beta2),
                'weight_decay': args.weight_decay}
    sched_args = {'mode': 'max', 'patience': args.pai_patience}
    return GPA.pai_tracker.setup_optimizer(model, opt_args, sched_args)


def add_score(model, acc1, args, device, force_switch=False):
    """Report a validation score. Returns (model, restructured, training_complete)."""
    # Percent, so PAI's improvement thresholds behave as they did in the HW1 experiments.
    # Actual return order is (net, restructured, training_complete); the docstring lists it differently.
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        100 * acc1, model, force_switch=force_switch)
    return model.to(device), bool(restructured), training_complete


def in_dendrite_phase():
    return GPA.pai_tracker.member_vars.get('mode') == 'p'


def stats():
    member_vars = GPA.pai_tracker.member_vars
    return {
        'pai/mode': 1 if in_dendrite_phase() else 0,
        'pai/dendrites_added': member_vars.get('num_dendrites_added', 0),
        # PAI's count excludes candidate-dendrite scaffolding, so it is the real model size.
        'pai/params': (member_vars.get('param_counts') or [0])[-1],
    }
