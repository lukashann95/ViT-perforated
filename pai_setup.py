"""PerforatedAI (Perforated Backpropagation) support for main.py.

Adds dendrites to selected encoder MLPs of a SimpleVisionTransformer, either fine-tuning a
trained checkpoint (--pai-init), training from scratch (usually with --pai-warmup), or resuming a
PAI run's neuron phase with its full tracker, optimizer and scheduler state (--pai-resume).
PAI is imported only when --pai-layers is set, so baseline runs need neither the package
nor PAI credentials. PAI reads PAIEMAIL and PAITOKEN from the environment.

PAI writes its graphs, CSVs and model saves to ./pai, so main.py changes into the run's
log directory before calling perforate().
"""
import json
import os
import pdb
import re
import time

import torch
import torch.nn as nn


GPA = UPA = None

# Context for diagnose(): main.py records the step and GradScaler events, and forward pre-hooks
# keep a reference (no copy) to the last input of every module that can grow dendrites.
DIAG = {'inputs': {}, 'hooks': []}


def _finite_summary(tensor):
    t = tensor.detach()
    bad = (~torch.isfinite(t)).sum().item() if t.is_floating_point() else 0
    absmax = t.float().abs().max().item() if t.numel() else 0.0
    return f"shape {tuple(t.shape)} {str(t.dtype).replace('torch.', '')} non-finite {bad} absmax {absmax:.4g}"


def diagnose(dump_path='pai_diagnostics.pt'):
    """Print the state of the dendrite modules and save it to dump_path (in the log directory)."""
    try:
        print("=== PAI diagnostics ===")
        for key in ('step', 'scale', 'last_scale_drop_step'):
            print(f"  {key}: {DIAG.get(key)}")
        tracker = GPA.pai_tracker.member_vars
        print(f"  tracker: mode {tracker.get('mode')}, epoch {tracker.get('num_epochs_run')}, "
              f"dendrites added {tracker.get('num_dendrites_added')}")
        dump = {'context': {k: v for k, v in DIAG.items() if k not in ('inputs', 'hooks')}, 'modules': {}}
        for module in GPA.pai_tracker.neuron_module_vector:
            name = getattr(module, 'name', type(module).__name__)
            x = DIAG['inputs'].get(id(module))
            print(f"  module {name}")
            print(f"    last input: {_finite_summary(x) if x is not None else 'none recorded'}")
            for pname, p in module.named_parameters():
                if not torch.isfinite(p).all() or 'candidate' in pname:
                    print(f"    param {pname}: {_finite_summary(p)}")
            for bname, b in module.named_buffers():
                if b.is_floating_point() and not torch.isfinite(b).all():
                    print(f"    buffer {bname}: {_finite_summary(b)}")
            dump['modules'][name] = {
                'input': x.detach().cpu() if x is not None else None,
                'state': {k: v.detach().cpu() for k, v in module.state_dict().items()},
            }
        torch.save(dump, dump_path)
        print(f"  saved {os.path.abspath(dump_path)}")
    except Exception as exc:  # diagnostics must not hide the original error
        print(f"  diagnostics failed: {exc!r}")


def register_diagnostics(model):
    """Keep a reference to the last input of each dendrite module. Call again after PAI
    restructures the model, since it may rebuild modules."""
    for handle in DIAG['hooks']:
        handle.remove()
    DIAG['hooks'], DIAG['inputs'] = [], {}
    for module in GPA.pai_tracker.neuron_module_vector:
        DIAG['hooks'].append(module.register_forward_pre_hook(
            lambda m, inputs: DIAG['inputs'].__setitem__(id(m), inputs[0])))


def _no_pdb(*args, **kwargs):
    # PAI calls pdb.set_trace() on some errors. Batch jobs have no stdin, so report and fail loudly.
    diagnose()
    raise RuntimeError("PerforatedAI called pdb.set_trace(); see traceback")


def _import_pai():
    # Imported on first use so baseline runs never load PAI.
    global GPA, UPA
    pdb.set_trace = _no_pdb
    from perforatedai import globals_perforatedai
    from perforatedai import utils_perforatedai
    GPA, UPA = globals_perforatedai, utils_perforatedai


def _dendrites_in_fp32():
    # Under fp16 autocast, perforatedai 4.0.0 produces NaN/inf candidate outputs on the first
    # dendrite-phase forward pass. Run the dendrite and candidate forward in fp32; the rest of
    # the network stays fp16. Full fp32 at batch 1024 does not fit on a 32 GB V100.
    from perforatedai import modules_perforatedai
    cls = modules_perforatedai.PAIDendriteModule
    if getattr(cls.forward, '_fp32', False):
        return
    forward = cls.forward

    def fp32_forward(self, *args, **kwargs):
        args = [a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args]
        with torch.autocast(device_type='cuda', enabled=False):
            return forward(self, *args, **kwargs)

    fp32_forward._fp32 = True
    cls.forward = fp32_forward

# Submodule of torchvision's MLPBlock to perforate: Linear(D, 4D), GELU, Dropout, Linear(4D, D), Dropout.
# 'ln_mlp' wraps the pre-norm ln_2 together with the whole MLP.
TARGETS = {'mlp': None, 'ln_mlp': None, 'fc1': 0, 'fc2': 3}


def add_arguments(parser):
    group = parser.add_argument_group('PerforatedAI')
    group.add_argument('--pai-layers', nargs='*', type=int, default=None, metavar='I',
                       help='0-indexed encoder layers whose MLP gets dendrites, e.g. 3 7 11. '
                            'Enables PAI; fine-tunes --pai-init if given, else trains from scratch.')
    group.add_argument('--pai-target', default='mlp', choices=sorted(TARGETS),
                       help='perforate the whole MLP block, the MLP with its pre-norm ln_2, '
                            'or only its first/second Linear')
    group.add_argument('--pai-control', action='store_true',
                       help='same wrapping, tracker and schedule, but never add dendrites')
    group.add_argument('--pai-init', default='', type=str, metavar='PATH',
                       help='trained checkpoint (main.py format) to fine-tune from; '
                            'omit to train from random initialization')
    group.add_argument('--pai-resume', default='', type=str, metavar='PATH',
                       help='main.py checkpoint of a PAI run saved before its first switch. '
                            'Continues that run (step, tracker history, optimizer, LR scheduler) '
                            'with this run\'s --pai-layers/--pai-target/--pai-control, so PAI '
                            'decides the first switch itself.')
    group.add_argument('--pai-warmup', default=0, type=int, metavar='STEPS',
                       help='linear LR warmup from 0 to --lr over the first STEPS optimizer steps. '
                            'PAI owns the scheduler, so --warmup does not apply to PAI runs.')
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
    group.add_argument('--pai-candidate-grad-clipping', default=None, type=float,
                       help='clip candidate dendrite gradients to this norm (PAI default: off)')


def check_args(parser, args):
    args.pai = args.pai_layers is not None
    if not args.pai:
        return
    if not args.pai_layers:
        parser.error('--pai-layers needs at least one layer index')
    if args.pai_warmup < 0:
        parser.error('--pai-warmup must be non-negative')
    if args.pai_resume and args.pai_init:
        parser.error('--pai-resume and --pai-init are mutually exclusive')
    if not all(0 <= i < args.num_layers for i in args.pai_layers):
        parser.error(f'--pai-layers must be in [0, {args.num_layers})')
    if args.distributed or args.ngpus_per_node > 1:
        parser.error('PAI runs support a single GPU only; pass --gpu 0')
    if args.resume:
        parser.error('--resume is not supported with PAI runs')
    if args.accum_freq != 1:
        # PAI stores one set of dendrite outputs and neuron errors per optimizer step.
        parser.error('PAI runs need --accum-freq 1')


def plain_key(key, target):
    """Map one key of a perforated model's state_dict to the plain model's key.

    Returns None for PAI bookkeeping (tracker_string, dendrite_module.*, shape buffers). Neuron
    weights only: dendrite weights, if any, are dropped too.
    """
    if key == 'tracker_string' or '.dendrite_module.' in key or key.endswith(
            ('.this_output_dimensions', '.this_node_index')):
        return None
    key = key.replace('.main_module.', '.')
    # The perforated target is wrapped as PAISequential([...]) -> '<module>.model.<i>.'.
    if target == 'mlp':
        key = re.sub(r'\.mlp\.model\.0\.', '.mlp.', key)
    elif target == 'ln_mlp':
        # PAISequential([ln_2, mlp]) replaced mlp, and ln_2 became Identity.
        key = re.sub(r'\.mlp\.model\.0\.', '.ln_2.', key)
        key = re.sub(r'\.mlp\.model\.1\.', '.mlp.', key)
    else:  # fc1 / fc2: mlp[i] = PAISequential([Linear])
        key = re.sub(r'\.mlp\.(\d+)\.model\.0\.', r'.mlp.\1.', key)
    return key


def detect_target(state_dict):
    """The --pai-target a perforated state_dict was saved with."""
    keys = [k.replace('.main_module.', '.') for k in state_dict]
    if any('.mlp.model.1.' in k for k in keys):
        return 'ln_mlp'
    if any('.mlp.model.0.' in k for k in keys):
        return 'mlp'
    return 'fc1'  # fc1 and fc2 map identically


def plain_state_dict(state_dict, target=None):
    target = target or detect_target(state_dict)
    plain = {}
    for key, value in state_dict.items():
        new = plain_key(key, target)
        if new is None:
            continue
        if new in plain:
            raise ValueError(f'two perforated keys map to {new}')
        plain[new] = value
    return plain


def tracker_string(checkpoint):
    tensor = checkpoint['state_dict']['tracker_string']
    return ''.join(map(chr, tensor.tolist()))


def tracker_value(string, name):
    for line in string.splitlines():
        if line.startswith(name + ','):
            return line.split(',', 1)[1]
    raise KeyError(name)


def load_resume(model, path):
    """Load the neuron weights of a pre-switch PAI checkpoint into the unperforated model.

    The weights may come from a run that perforated other layers: before the first switch the
    wrapping is bookkeeping only. Returns the checkpoint for restore_tracker/restore_training.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    string = tracker_string(checkpoint)
    mode, dendrites = tracker_value(string, 'mode'), int(tracker_value(string, 'num_dendrites_added'))
    if mode != 'n' or dendrites:
        raise ValueError(f"--pai-resume needs a checkpoint from before the first switch; '{path}' "
                         f"is in mode {mode} with {dendrites} dendrites")
    model.load_state_dict(plain_state_dict(checkpoint['state_dict']), strict=True)
    print(f"=> resuming PAI neuron phase from '{path}' (step {checkpoint['step']}, "
          f"epoch {tracker_value(string, 'num_epochs_run')}, "
          f"last improved epoch {tracker_value(string, 'epoch_last_improved')})")
    return checkpoint


def restore_tracker(model, checkpoint, args):
    """Restore PAI's tracker from the checkpoint, after perforate().

    Follows UPA.load_system, minus its reset of current_best_validation_score and
    epoch_last_improved: those restart PAI's countdown to the next switch, while this resume
    continues it, so PAI switches when the original run would have.
    """
    tracker = GPA.pai_tracker
    tracker.reset_module_vector(model, False)
    tracker.from_string(tracker_string(checkpoint))
    # Run identity comes from this run's flags, not the source run (e.g. a control arm).
    tracker.member_vars['doing_pai'] = not args.pai_control
    tracker.saved_time = time.time()
    tracker.loaded = True
    # PAI saves before the next epoch starts; load_system calls this to fix the off-by-one.
    tracker.start_epoch(internal_call=True)


def resume_best_model_path(resume_path):
    """PAI's best_model.pt saved with a --pai-resume checkpoint: <snapshot>/pai/ for archived
    snapshots, <run>/pai/ for <run>/checkpoints/."""
    folder = os.path.dirname(os.path.realpath(resume_path))
    for candidate in (os.path.join(folder, 'pai', 'best_model.pt'),
                      os.path.join(os.path.dirname(folder), 'pai', 'best_model.pt')):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"no pai/best_model.pt next to '{resume_path}'; PAI reloads the best "
                            "model when it switches, so a resume needs it")


def _load_pai_file(path):
    """A PAI save (safetensors state_dict), with tied weights restored as UPA.load_net does."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    state = load_file(path)
    with safe_open(path, framework='pt') as f:
        metadata = f.metadata() or {}
    for secondary, primary in json.loads(metadata.get('weight_tying', '{}')).items():
        if primary in state:
            state[secondary] = state[primary]
    return state


def write_best_model(model, args):
    """Write pai/best_model.pt for a resumed run, in this run's wrapping.

    When PAI switches to dendrite training it reloads pai/best_model.pt (weights and tracker
    of the best epoch so far) and adds dendrites to that. A resumed run has not saved one yet,
    so rebuild the source run's best model: this model's state_dict structure, with every
    neuron weight and the tracker taken from the source's best_model.pt. Call after the first
    training backward pass (PAI creates its dendrite buffers there), with the run's log
    directory as cwd.
    """
    source = _load_pai_file(resume_best_model_path(args.pai_resume))
    best = plain_state_dict(source)
    state, copied = {}, 0
    for key, value in model.state_dict().items():
        # dendrite_module.parent_module.* is a second reference to the wrapped module's weights.
        name = plain_key(key.replace('.dendrite_module.parent_module.', '.main_module.'), args.pai_target)
        if name is None:
            state[key] = value.detach().cpu().clone()
            continue
        if name not in best or best[name].shape != value.shape:
            raise ValueError(f"source best model has no weight for '{key}' ('{name}')")
        state[key] = best[name].clone()
        copied += 1
    state['tracker_string'] = source['tracker_string'].clone()
    os.makedirs('pai', exist_ok=True)
    if GPA.pc.get_using_safe_tensors():
        UPA._save_file_deduped(state, 'pai/best_model.pt')
    else:
        raise NotImplementedError('PAI is set to pickle whole models; write_best_model needs safetensors')
    string = ''.join(map(chr, source['tracker_string'].tolist()))
    print(f"=> wrote pai/best_model.pt from '{resume_best_model_path(args.pai_resume)}': {copied} "
          f"weights, best epoch {tracker_value(string, 'num_epochs_run')} "
          f"(score {float(tracker_value(string, 'current_best_validation_score')):.3f})")


def restore_training(optimizer, scheduler, scaler, checkpoint, args):
    """Restore optimizer, LR scheduler and GradScaler state; set the resume step."""
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp and checkpoint.get('scaler'):
        scaler.load_state_dict(checkpoint['scaler'])
    args.start_step = checkpoint['step']
    print(f"=> resumed at step {args.start_step}, lr {optimizer.param_groups[0]['lr']:.3g}")


def load_init(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> fine-tuning from '{path}' (step {checkpoint['step']}, best acc@1 {checkpoint['best_acc1']:.4f})")


def perforate(model, args):
    _import_pai()
    if args.amp:
        _dendrites_in_fp32()
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
    if args.pai_candidate_grad_clipping is not None:
        pc.set_candidate_grad_clipping(args.pai_candidate_grad_clipping)
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

    model = UPA.perforate_model(model, doing_pai=not args.pai_control, save_name='pai', maximizing_score=True)
    register_diagnostics(model)
    return model


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
    if restructured:
        register_diagnostics(model)
    return model.to(device), bool(restructured), training_complete


def warmup_lr(optimizer, args, step):
    # ReduceLROnPlateau only lowers the LR at validations, and PAI rebuilds the optimizer only
    # after a phase switch, so overwriting the LR for the first steps is safe as long as the
    # warmup ends before the first plateau or switch (warmup < pai_n_epochs_to_switch epochs).
    for group in optimizer.param_groups:
        group['lr'] = args.lr * step / args.pai_warmup


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
