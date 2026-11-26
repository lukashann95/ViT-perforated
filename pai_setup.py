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
import math
import os
import pdb
import re
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Per backward pass in dendrite phases: is the gradient at a dendrite module's output non-finite?
_OVERFLOW_FLAGS = []


def _check_output_grad(grad):
    _OVERFLOW_FLAGS.append(~torch.isfinite(grad).all())


def _watch_output_grad(module, inputs, output):
    # The gradient at the module output is the error signal PAI trains dendrites on.
    if torch.is_grad_enabled() and in_dendrite_phase():
        out = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(out) and out.requires_grad:
            out.register_hook(_check_output_grad)


def dendrite_grads_overflowed():
    """True if a backward pass since the last call left a non-finite gradient at the output
    of a dendrite module, i.e. fp16 overflowed under the loss scale. Clears the record."""
    if not _OVERFLOW_FLAGS:
        return False
    found = bool(torch.stack(_OVERFLOW_FLAGS).any())
    _OVERFLOW_FLAGS.clear()
    return found


def register_diagnostics(model):
    """Keep a reference to the last input of each dendrite module and watch the gradient at
    its output in dendrite phases. Call again after PAI restructures the model, since it may
    rebuild modules."""
    for handle in DIAG['hooks']:
        handle.remove()
    DIAG['hooks'], DIAG['inputs'] = [], {}
    _OVERFLOW_FLAGS.clear()
    for module in GPA.pai_tracker.neuron_module_vector:
        DIAG['hooks'].append(module.register_forward_pre_hook(
            lambda m, inputs: DIAG['inputs'].__setitem__(id(m), inputs[0])))
        DIAG['hooks'].append(module.register_forward_hook(_watch_output_grad))


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
# 'ln_mlp' wraps the pre-norm ln_2 together with the whole MLP; 'ln_fc1' wraps ln_2 with the
# first Linear only (dendrites on the 4D hidden units before the GELU).
# 'qk' and 'qkv' wrap the pre-norm ln_1 with the attention input projections (see SplitAttention).
TARGETS = {'mlp': None, 'ln_mlp': None, 'ln_fc1': 0, 'fc1': 0, 'fc2': 3, 'qk': None, 'qkv': None}
ATTENTION_TARGETS = ('qk', 'qkv')


class SplitAttention(nn.Module):
    """A block's ln_1 and nn.MultiheadAttention, with the fused in_proj split into Linears PAI can wrap.

    nn.MultiheadAttention keeps q, k and v in one in_proj_weight that it passes to a functional
    call, so there is no Linear for PAI to put dendrites on. Like ln_mlp/ln_fc1, the wrapped unit
    includes the pre-norm (PAI's guidance), so the block's ln_1 moves in here and the block's own
    ln_1 becomes Identity:
      'qk':  self.qk = sequential([ln_1, Linear(D, 2D)]) for queries and keys (the only inputs of
             the token-token scores); v = self.v(self.ln_1(x)) with self.ln_1 the same LayerNorm
             object as self.qk's, so q, k and v keep one shared ln_1 as in the plain model.
      'qkv': self.qkv = sequential([ln_1, Linear(D, 3D)]).
    Same math as the modules it replaces (batch_first self-attention, called as
    attention(x, x, x, need_weights=False) on the block input). out_proj is copied into a plain Linear.
    """

    def __init__(self, attention, ln_1, target, sequential):
        super().__init__()
        if target not in ATTENTION_TARGETS:
            raise ValueError(f'SplitAttention target must be one of {ATTENTION_TARGETS}, not {target}')
        if not (attention.batch_first and attention._qkv_same_embed_dim and attention.in_proj_bias is not None
                and attention.bias_k is None and not attention.add_zero_attn):
            raise ValueError('SplitAttention supports batch_first self-attention with in_proj bias only')
        dim = attention.embed_dim
        self.target, self.num_heads, self.dropout = target, attention.num_heads, attention.dropout

        def linear(weight, bias):
            layer = nn.Linear(weight.shape[1], weight.shape[0], device=weight.device, dtype=weight.dtype)
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
            return layer

        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        if target == 'qk':
            self.qk = sequential([ln_1, linear(weight[:2 * dim], bias[:2 * dim])])
            # Second reference to the LayerNorm inside self.qk, for the v path. Keep it registered
            # after self.qk: PAI keeps the first path it finds for a shared module, and this order
            # makes ln_1 part of the perforated block rather than a separately tracked module
            # (both work; prints list ln_1 after qk although it runs first).
            self.ln_1 = ln_1
            self.v = linear(weight[2 * dim:], bias[2 * dim:])
        else:
            self.qkv = sequential([ln_1, linear(weight, bias)])
        # A plain Linear: MultiheadAttention's out_proj is a Linear subclass
        # (NonDynamicallyQuantizableLinear) that PAI's modules_to_track does not match, and PAI
        # stops at optimizer setup on parameters that are neither tracked nor perforated.
        self.out_proj = linear(attention.out_proj.weight, attention.out_proj.bias)

    def forward(self, query, key, value, need_weights=False):
        torch._assert(query is key and key is value and not need_weights,
                      'SplitAttention does self-attention only, without attention weights')
        batch, tokens, _ = query.shape
        if self.target == 'qk':
            qkv = torch.cat([self.qk(query), self.v(self.ln_1(query))], dim=-1)
        else:
            qkv = self.qkv(query)
        # [batch, tokens, 3D] -> 3 x [batch, heads, tokens, head_dim], q/k/v in in_proj's order.
        q, k, v = qkv.view(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(batch, tokens, -1)), None


def wrap_target(block, target, sequential):
    """Wrap one encoder block's --pai-target in `sequential` (GPA.PAISequential in perforate)."""
    if target in ATTENTION_TARGETS:
        # forward() computes self_attention(ln_1(x)); ln_1 moves into SplitAttention.
        block.self_attention = SplitAttention(block.self_attention, block.ln_1, target, sequential)
        block.ln_1 = nn.Identity()
        return
    mlp = block.mlp
    index = TARGETS[target]
    if target == 'ln_mlp':
        # forward() computes mlp(ln_2(x)), so moving ln_2 inside the wrapper keeps the math.
        block.mlp = sequential([block.ln_2, mlp])
        block.ln_2 = nn.Identity()
    elif target == 'ln_fc1':
        # Likewise: mlp[0](ln_2(x)) with ln_2 moved into the wrapper around the first Linear.
        mlp[0] = sequential([block.ln_2, mlp[0]])
        block.ln_2 = nn.Identity()
    elif index is None:
        block.mlp = sequential([mlp])
    else:
        mlp[index] = sequential([mlp[index]])


def add_arguments(parser):
    group = parser.add_argument_group('PerforatedAI')
    group.add_argument('--pai-layers', nargs='*', type=int, default=None, metavar='I',
                       help='0-indexed encoder layers whose MLP gets dendrites, e.g. 3 7 11. '
                            'Enables PAI; fine-tunes --pai-init if given, else trains from scratch.')
    group.add_argument('--pai-target', default='ln_mlp', choices=sorted(TARGETS),
                       help='perforate the MLP with its pre-norm ln_2 (default; PAI wants norm '
                            'layers inside blocks), the whole MLP block, only its first/second Linear, '
                            'or the attention projections: qk (queries and keys, which set the attention '
                            'pattern) or qkv (all three)')
    group.add_argument('--pai-gd', action='store_true',
                       help="train dendrites by gradient descent together with the neurons (PAI's "
                            "open-source mode) instead of Perforated Backpropagation's cascade-"
                            "correlation phases, even when perforatedbp is installed")
    group.add_argument('--pai-dendrite-loss', default='default', choices=['default', 'covariance'],
                       help="Perforated Backpropagation candidate loss: perforatedbp's default "
                            "(ImprovedCovarianceDendriteLoss; its scores only stayed 0 under "
                            "perforatedai 4.0.0's pai_improvement_threshold_raw of 1e-4, see perforate()) "
                            "or the older CovarianceDendriteLoss")
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
    group.add_argument('--pai-new-schedule', action='store_true',
                       help='with --resume: keep the weights, dendrites and tracker but build a fresh '
                            'optimizer and LR scheduler from this run\'s --lr, --wd and '
                            '--pai-cosine-epochs instead of restoring the saved ones (e.g. a new '
                            'neuron phase after a dendrite-phase ablation)')
    group.add_argument('--pai-force-switch', action='store_true',
                       help='with --resume of a neuron-phase checkpoint: validate the loaded model and '
                            'force PAI\'s next switch at step 0, as --pai-init does, so the run starts in '
                            'a new dendrite phase (e.g. a second dendrite set; needs --max-dendrites above '
                            'the sets already installed)')
    group.add_argument('--pai-warmup', default=0, type=int, metavar='STEPS',
                       help='linear LR warmup from 0 to --lr over the first STEPS optimizer steps. '
                            'PAI owns the scheduler, so --warmup does not apply to PAI runs.')
    group.add_argument('--max-dendrites', default=1, type=int,
                       help='maximum dendrite rounds PAI may add')
    group.add_argument('--pai-fixed-switch', default=None, type=int, metavar='N',
                       help='switch on a fixed schedule (PAI DOING_FIXED_SWITCH) instead of when '
                            'validation stops improving: dendrite phases start after N epochs of '
                            'neuron training and last --pai-p-epochs-to-switch epochs')
    group.add_argument('--pai-n-epochs-to-switch', default=10, type=int,
                       help='minimum validations in a neuron phase before PAI may switch')
    group.add_argument('--pai-p-epochs-to-switch', default=2, type=int,
                       help='minimum validations in a dendrite phase before PAI may switch')
    group.add_argument('--pai-initial-correlation-batches', default=None, type=int,
                       help="training batches before PAI scores dendrites (PAI default 100); "
                            "must be below --log-steps")
    group.add_argument('--pai-patience', default=2, type=int,
                       help='ReduceLROnPlateau patience, in validations (--pai-schedule plateau)')
    group.add_argument('--pai-schedule', default='plateau', choices=['plateau', 'cosine'],
                       help='LR scheduler PAI builds for every phase: ReduceLROnPlateau, or '
                            'CosineAnnealingLR from the phase\'s starting LR to 0 over '
                            '--pai-cosine-epochs validations. PAI rebuilds and steps it (once per '
                            'validation), as in PAI\'s ResNet example; each phase restarts it.')
    group.add_argument('--pai-p-lr-scaling', default='none', choices=['none', 'linear', 'sqrt'],
                       help='with --accum-freq k, dendrite (p) phases step once per microbatch of '
                            'batch/k, so scale their LR by 1/k (linear) or 1/sqrt(k) (sqrt). '
                            'none keeps --lr, as PAI runs did before this option.')
    group.add_argument('--pai-cosine-epochs', default=None, type=int, metavar='N',
                       help='T_max of --pai-schedule cosine, in validations (one per epoch with '
                            '--log-steps = steps per epoch)')
    group.add_argument('--pai-candidate-grad-clipping', default=None, type=float,
                       help='clip candidate dendrite gradients to this norm (PAI default: off)')
    group.add_argument('--pai-p-loss-scale', default=1.0, type=float, metavar='S',
                       help='multiply the loss by S in dendrite (p) phase backward passes (after Vamsi, '
                            'branch pai-clean-pphase). The batch-mean loss makes the error each sample '
                            'sends to a candidate ~1/microbatch, possibly small enough that AdamW\'s eps '
                            'slows candidate learning; S = the microbatch size (batch/accum) restores '
                            'per-sample magnitudes. The PB score is a correlation, so S changes learning, '
                            'not the metric. Neuron phases are unaffected.')


def check_args(parser, args):
    args.pai = args.pai_layers is not None
    if not args.pai:
        return
    if not args.pai_layers:
        parser.error('--pai-layers needs at least one layer index')
    if args.pai_dendrite_loss != 'default' and (args.pai_gd or args.pai_control):
        parser.error('--pai-dendrite-loss applies only to Perforated Backpropagation runs')
    if args.pai_warmup < 0:
        parser.error('--pai-warmup must be non-negative')
    if args.pai_fixed_switch is not None and args.pai_fixed_switch < 1:
        parser.error('--pai-fixed-switch must be at least 1')
    if args.pai_schedule == 'cosine':
        if not args.pai_cosine_epochs or args.pai_cosine_epochs < 1:
            parser.error('--pai-schedule cosine needs --pai-cosine-epochs >= 1')
        if args.pai_warmup:
            # warmup_lr overwrites the LR, and CosineAnnealingLR computes each LR from the last.
            parser.error('--pai-warmup is not supported with --pai-schedule cosine')
    elif args.pai_cosine_epochs is not None:
        parser.error('--pai-cosine-epochs applies only to --pai-schedule cosine')
    if args.pai_p_loss_scale <= 0:
        parser.error('--pai-p-loss-scale must be positive')
    if args.pai_resume and args.pai_init:
        parser.error('--pai-resume and --pai-init are mutually exclusive')
    if not all(0 <= i < args.num_layers for i in args.pai_layers):
        parser.error(f'--pai-layers must be in [0, {args.num_layers})')
    if args.distributed or args.ngpus_per_node > 1:
        parser.error('PAI runs support a single GPU only; pass --gpu 0')
    if args.resume and (args.pai_resume or args.pai_init):
        parser.error('--resume continues a PAI run; it excludes --pai-resume and --pai-init')
    if args.pai_new_schedule and not args.resume:
        parser.error('--pai-new-schedule applies only to --resume')
    if args.pai_force_switch and (not args.resume or args.pai_control):
        parser.error('--pai-force-switch applies only to --resume, without --pai-control')


def plain_key(key, target):
    """Map one key of a perforated model's state_dict to the plain model's key.

    Returns None for PAI bookkeeping (tracker_string, dendrite_module.*, shape buffers). Neuron
    weights only: dendrite weights, if any, are dropped too. For 'qk' the q/k and v projections
    map to the slices '<...>.in_proj_weight:qk' and ':v'; plain_state_dict concatenates them.
    Two keys can map to the same plain key for a shared module ('qk': ln_1).
    """
    if key == 'tracker_string' or '.dendrite_module.' in key or key.endswith(
            ('.this_output_dimensions', '.this_node_index')):
        return None
    key = key.replace('.main_module.', '.')
    # The perforated target is wrapped as PAISequential([...]) -> '<module>.model.<i>.'.
    if target in ATTENTION_TARGETS:
        # SplitAttention: '<target>.model.0' is the block's ln_1 ('qk' also has it as
        # self_attention.ln_1, the same tensors; plain_state_dict keeps one), '<target>.model.1'
        # the in_proj rows.
        key = re.sub(rf'\.self_attention\.(?:{target}\.model\.0|ln_1)\.', '.ln_1.', key)
        suffix = ':qk' if target == 'qk' else ''
        key = re.sub(rf'\.self_attention\.{target}\.model\.1\.(weight|bias)$', rf'.self_attention.in_proj_\1{suffix}', key)
        key = re.sub(r'\.self_attention\.v\.(weight|bias)$', r'.self_attention.in_proj_\1:v', key)
    elif target == 'mlp':
        key = re.sub(r'\.mlp\.model\.0\.', '.mlp.', key)
    elif target == 'ln_mlp':
        # PAISequential([ln_2, mlp]) replaced mlp, and ln_2 became Identity.
        key = re.sub(r'\.mlp\.model\.0\.', '.ln_2.', key)
        key = re.sub(r'\.mlp\.model\.1\.', '.mlp.', key)
    elif target == 'ln_fc1':
        # PAISequential([ln_2, fc1]) replaced mlp[0], and ln_2 became Identity.
        key = re.sub(r'\.mlp\.0\.model\.0\.', '.ln_2.', key)
        key = re.sub(r'\.mlp\.0\.model\.1\.', '.mlp.0.', key)
    else:  # fc1 / fc2: mlp[i] = PAISequential([Linear])
        key = re.sub(r'\.mlp\.(\d+)\.model\.0\.', r'.mlp.\1.', key)
    return key


def check_shared_ln(model):
    """Raise unless each 'qk' block's ln_1 is still one LayerNorm in its q/k and v paths.

    SplitAttention holds the LayerNorm both inside the perforated self.qk and as self.ln_1; this
    catches PAI copying either reference when it wraps modules or adds dendrites.
    """
    params = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if '.self_attention.' not in name or '.dendrite_module.' in name:
            continue
        plain = plain_key('.' + name, 'qk')
        if plain and re.search(r'\.ln_1\.(weight|bias)$', plain):
            params.setdefault(plain, []).append(param)
    broken = [k for k, found in params.items() if len(found) != 2 or found[1] is not found[0]]
    if broken:
        raise RuntimeError(f"'qk' target: ln_1 is no longer shared between q/k and v: {broken}")


def detect_target(state_dict):
    """The --pai-target a perforated state_dict was saved with."""
    keys = [k.replace('.main_module.', '.') for k in state_dict]
    if any('.self_attention.qk.' in k for k in keys):
        return 'qk'
    if any('.self_attention.qkv.' in k for k in keys):
        return 'qkv'
    if any('.mlp.0.model.1.' in k for k in keys):
        return 'ln_fc1'
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
            # The 'qk' target's shared ln_1 appears under two keys with the same tensor.
            if torch.equal(plain[new], value):
                continue
            raise ValueError(f'two perforated keys map to {new}')
        plain[new] = value
    # 'qk': in_proj = [q; k] rows from the wrapped Linear(D, 2D), then v from Linear(D, D).
    for key in [k for k in plain if k.endswith(':qk')]:
        base = key.removesuffix(':qk')
        plain[base] = torch.cat([plain.pop(key), plain.pop(base + ':v')])
    return plain


def split_in_proj(plain):
    """plain_state_dict's inverse for 'qk': add the ':qk' and ':v' slices of every in_proj."""
    split = dict(plain)
    for key, value in plain.items():
        if key.endswith(('.self_attention.in_proj_weight', '.self_attention.in_proj_bias')):
            dim = value.shape[0] // 3
            split[key + ':qk'], split[key + ':v'] = value[:2 * dim], value[2 * dim:]
    return split


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


def load_continue(model, path, args):
    """Continue a PAI run from its own main.py checkpoint (--resume), after perforate().

    The checkpoint may have dendrites: PAI's load_net_from_dict rebuilds them from the saved
    cycle count, then the tracker is restored as for --pai-resume. Only neuron-phase ('n')
    checkpoints: in a dendrite phase Perforated Backpropagation keeps candidate statistics that
    the state_dict does not hold. Returns (model, checkpoint) for restore_training.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    string = tracker_string(checkpoint)
    mode = tracker_value(string, 'mode')
    if mode != 'n':
        raise ValueError(f"--resume needs a PAI checkpoint saved in a neuron phase; '{path}' is in "
                         f"mode {mode}")
    loaded = UPA.load_net_from_dict(model, checkpoint['state_dict'])
    model = loaded if loaded is not None else model
    restore_tracker(model, checkpoint, args)
    print(f"=> continuing PAI run from '{path}' (step {checkpoint['step']}, "
          f"{tracker_value(string, 'num_dendrites_added')} dendrite sets, "
          f"epoch {tracker_value(string, 'num_epochs_run')}, "
          f"last improved epoch {tracker_value(string, 'epoch_last_improved')})")
    return model, checkpoint


def resize_module_lists(string, modules):
    """Resize a tracker string's per-module lists (best_scores, current_scores) to `modules`.

    The tracker stores one line per perforated module, then 'end'. A --pai-resume source may
    have perforated a different number of modules (e.g. layer 11 only, for a layers-6..11 arm).
    Before the first switch every list is empty, so they can be resized; otherwise PAI indexes
    past the end when it plots at the first switch.
    """
    lines = string.split('\n')
    for name in ('best_scores', 'current_scores'):
        start = lines.index(name + ',') + 1
        end = lines.index('end', start)
        if end - start == modules:
            continue
        if any(lines[start:end]):
            raise ValueError(f"tracker has {name} for {end - start} perforated modules and this run "
                             f"has {modules}; only empty ones (before the first switch) can be resized")
        lines[start:end] = [''] * modules
    return '\n'.join(lines)


def restore_tracker(model, checkpoint, args):
    """Restore PAI's tracker from the checkpoint, after perforate().

    Follows UPA.load_system, minus its reset of current_best_validation_score and
    epoch_last_improved: those restart PAI's countdown to the next switch, while this resume
    continues it, so PAI switches when the original run would have.
    """
    tracker = GPA.pai_tracker
    tracker.reset_module_vector(model, False)
    tracker.from_string(resize_module_lists(tracker_string(checkpoint), len(tracker.neuron_module_vector)))
    # Run identity comes from this run's flags, not the source run (e.g. a control arm).
    tracker.member_vars['doing_pai'] = not args.pai_control
    if getattr(args, 'pai_fixed_switch', None):
        tracker.member_vars['switch_mode'] = GPA.pc.DOING_FIXED_SWITCH
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
    best = split_in_proj(plain_state_dict(source))
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
    # PAI also reloads the tracker stored in best_model.pt at a switch; give it this run's identity.
    string = resize_module_lists(''.join(map(chr, source['tracker_string'].tolist())),
                                 len(GPA.pai_tracker.neuron_module_vector))
    run_settings = {'doing_pai': str(not args.pai_control)}
    if getattr(args, 'pai_fixed_switch', None):
        run_settings['switch_mode'] = str(GPA.pc.DOING_FIXED_SWITCH)
    lines = string.split('\n')
    for i, line in enumerate(lines):
        name = line.split(',', 1)[0]
        if name in run_settings and ',' in line:
            lines[i] = f"{name},{run_settings[name]}"
    state['tracker_string'] = torch.tensor([ord(c) for c in '\n'.join(lines)], dtype=torch.uint8)
    os.makedirs('pai', exist_ok=True)
    if GPA.pc.get_using_safe_tensors():
        UPA._save_file_deduped(state, 'pai/best_model.pt')
    else:
        raise NotImplementedError('PAI is set to pickle whole models; write_best_model needs safetensors')
    print(f"=> wrote pai/best_model.pt from '{resume_best_model_path(args.pai_resume)}': {copied} "
          f"weights, best epoch {tracker_value(string, 'num_epochs_run')} "
          f"(score {float(tracker_value(string, 'current_best_validation_score')):.3f})")


def prepare_forced_switch(model, resume_path):
    """Write pai/best_model.pt for --pai-force-switch. Call after the first training backward pass
    (PAI creates its dendrite buffers there), with the run's log directory as cwd.

    PAI reloads pai/best_model.pt, weights and tracker, at every switch, and this run's new log
    directory has none. Write the loaded model with the tracker string exactly as the --resume
    checkpoint saved it: restore_tracker has since advanced the tracker by one epoch
    (start_epoch), and a best model carrying that state leaves PAI's epoch counter one ahead of
    its history after the switch (its graphs then fail with an IndexError). This is the state PAI
    itself would reload when the checkpoint's epoch was the source run's best; otherwise PAI
    would have gone back to that best epoch, while this adds the next set to the loaded model.
    """
    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    state['tracker_string'] = checkpoint['state_dict']['tracker_string'].clone()
    os.makedirs('pai', exist_ok=True)
    if not GPA.pc.get_using_safe_tensors():
        raise NotImplementedError('PAI is set to pickle whole models; prepare_forced_switch needs safetensors')
    UPA._save_file_deduped(state, 'pai/best_model.pt')
    string = tracker_string(checkpoint)
    print(f"=> wrote pai/best_model.pt from the loaded model for --pai-force-switch (tracker at epoch "
          f"{tracker_value(string, 'num_epochs_run')}, last improved epoch "
          f"{tracker_value(string, 'epoch_last_improved')})")


def before_first_switch():
    """True while PAI is still in its first neuron phase: mode 'n' and no dendrites added yet."""
    member_vars = GPA.pai_tracker.member_vars
    return member_vars.get('mode') == 'n' and not member_vars.get('num_dendrites_added', 0)


def archive_pre_switch(args):
    """Keep the run's latest pre-switch state in <run>/archive/pre_switch/, for --pai-resume.

    Call after each validation's checkpoint save. While PAI is still in its first neuron phase
    (mode 'n', no dendrites), copy checkpoints/checkpoint.pth.tar and pai/best_model.pt (which
    PAI reloads at a switch) into the archive; from the first switch on, leave it alone, so it
    holds the last neuron-only state. Both files are replaced atomically.
    """
    if not before_first_switch():
        return
    dest = os.path.join(args.log_dir, 'archive', 'pre_switch')
    os.makedirs(os.path.join(dest, 'pai'), exist_ok=True)
    for src, dst in ((os.path.join(args.log_dir, 'pai', 'best_model.pt'), os.path.join(dest, 'pai', 'best_model.pt')),
                     (os.path.join(args.checkpoint_path, 'checkpoint.pth.tar'), os.path.join(dest, 'checkpoint.pth.tar'))):
        if os.path.isfile(src):
            shutil.copyfile(src, dst + '.tmp')
            os.replace(dst + '.tmp', dst)


def restore_training(optimizer, scheduler, scaler, checkpoint, args):
    """Restore optimizer, LR scheduler and GradScaler state; set the resume step.

    With --pai-new-schedule the optimizer and scheduler stay as setup_optimizer built them.
    """
    if not getattr(args, 'pai_new_schedule', False):
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
        wrap_target(model.encoder.layers[i], args.pai_target, GPA.PAISequential)

    pc = GPA.pc
    if getattr(args, 'pai_gd', False):
        # Importing perforatedbp switches PAI to Perforated Backpropagation; restore the
        # open-source defaults so dendrites are trained by gradient descent with the neurons.
        pc.set_perforated_backpropagation(False)
        pc.set_no_extra_n_modes(True)
    if getattr(args, 'pai_dendrite_loss', 'default') == 'covariance':
        import perforatedbp.tracker_pbp as TPB
        import perforatedbp.dendrite_losses.dendrite_loss_pbp as DLP
        TPB.setup_new_global_dendrite_mode(
            DLP.CovarianceDendriteLoss, TPB.best_cascor_score_improved_this_epoch,
            dendrite_learn_mode=True)
    pc.set_testing_dendrite_capacity(False)
    pc.set_max_dendrites(args.max_dendrites)
    pc.set_n_epochs_to_switch(args.pai_n_epochs_to_switch)
    pc.set_p_epochs_to_switch(args.pai_p_epochs_to_switch)
    if args.pai_schedule == 'cosine':
        # PAI's default LR search after structural changes (find_best_lr) would override the
        # cosine's LRs; PAI also says it only suits step-based schedulers.
        pc.set_find_best_lr(False)
    if getattr(args, 'pai_fixed_switch', None):
        pc.set_switch_mode(pc.DOING_FIXED_SWITCH)
        pc.set_n_fixed_switch_num(args.pai_fixed_switch)
        pc.set_first_fixed_switch_num(args.pai_fixed_switch)
        pc.set_p_fixed_switch_num(args.pai_p_epochs_to_switch)
    if args.pai_initial_correlation_batches is not None:
        pc.set_initial_correlation_batches(args.pai_initial_correlation_batches)
    if args.pai_candidate_grad_clipping is not None:
        pc.set_candidate_grad_clipping(args.pai_candidate_grad_clipping)
    # A candidate dendrite counts as improving when its raw correlation score rises by more than
    # this amount. ImprovedCovarianceDendriteLoss scores are tiny (~1e-12 on ViT-Tiny), so
    # perforatedai 4.0.0's default of 1e-4 meant no candidate ever improved and the PB scores
    # stayed 0. 4.0.2 defaults to 0; set it explicitly so a changed default cannot bring that back.
    # scripts/smoke_test.py checks that candidate scores register.
    pc.set_pai_improvement_threshold_raw(0)
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
    check_shared_ln(model)
    return model


def decays(name, param):
    """Whether a parameter gets weight decay: main.weight_decay_param's rule (weight matrices)."""
    return param.ndim >= 2 and name.endswith('weight')


def setup_optimizer(model, args):
    # PAI builds the optimizer itself so it can swap to dendrite-only parameters in dendrite
    # phases, and builds and steps the scheduler (from validation scores for ReduceLROnPlateau,
    # once per validation for CosineAnnealingLR); main.py must not step it. Called again after
    # every restructure, so each phase gets a fresh scheduler starting at its full LR.
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    # Weight decay on weight matrices only, as main.py's baseline optimizer does: biases,
    # LayerNorm gains and the like get none. PAI keeps parameter groups (minus parameters that
    # do not require grad in the current phase). Dendrite phases use no weight decay on the
    # candidates (as the p-phase LR ablations ran them, with --wd 0); --wd is for neuron phases.
    named = list(model.named_parameters())
    wd = 0.0 if in_dendrite_phase() else args.weight_decay
    groups = [{'params': [p for n, p in named if decays(n, p)], 'weight_decay': wd},
              {'params': [p for n, p in named if not decays(n, p)], 'weight_decay': 0.0}]
    lr = args.lr * p_lr_factor(args) if in_dendrite_phase() else args.lr
    opt_args = {'params': groups, 'lr': lr, 'betas': (args.beta1, args.beta2),
                'weight_decay': wd}
    if args.pai_schedule == 'cosine':
        # Decays to 0 like main.py's baseline cosine (eta_min 0), but per validation, not per step.
        GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR)
        sched_args = {'T_max': args.pai_cosine_epochs, 'eta_min': 0.0}
    else:
        GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        sched_args = {'mode': 'max', 'patience': args.pai_patience}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, opt_args, sched_args)
    # PAI adds dendrite parameters to the groups itself; log what each group ends up with.
    phase = 'dendrite' if in_dendrite_phase() else 'neuron'
    print(f"=> {phase}-phase optimizer (--lr {args.lr:.3g}, --pai-p-lr-scaling {args.pai_p_lr_scaling}, "
          f"--accum-freq {args.accum_freq}): " + "; ".join(
              f"group {i}: {sum(p.numel() for p in g['params'])} params, lr {g['lr']:.3g}, "
              f"weight_decay {g['weight_decay']:.3g}" for i, g in enumerate(optimizer.param_groups)))
    return optimizer, scheduler


def p_lr_factor(args):
    """LR factor for dendrite phases, where each microbatch (batch / accum_freq) is a step."""
    if args.pai_p_lr_scaling == 'linear':
        return 1 / args.accum_freq
    if args.pai_p_lr_scaling == 'sqrt':
        return 1 / math.sqrt(args.accum_freq)
    return 1.0


def add_score(model, acc1, args, device, force_switch=False):
    """Report a validation score. Returns (model, restructured, training_complete)."""
    # Percent, so PAI's improvement thresholds behave as they did in the HW1 experiments.
    # Actual return order is (net, restructured, training_complete); the docstring lists it differently.
    # PAI appends this epoch's PB scores inside add_validation_score and, when it ends the dendrite
    # phase, replaces its score lists with new empty ones, so hold on to the old lists.
    global _pb_scores
    best_scores = GPA.pai_tracker.member_vars.get('best_scores') if in_dendrite_phase() else None
    _hook_neuron_pb_scores()
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        100 * acc1, model, force_switch=force_switch)
    _pb_scores = {}
    for module, scores in zip(GPA.pai_tracker.neuron_module_vector, best_scores or []):
        if len(scores):
            _pb_scores[f'pai/pb_best/{short_module_name(module.name)}'] = float(scores[-1])
    if restructured:
        register_diagnostics(model)
        check_shared_ln(model)
    return model.to(device), bool(restructured), training_complete


_neuron_dump_count = 0


def _hook_neuron_pb_scores():
    """Make PAI save per-neuron PB scores right after it updates them at each dendrite-phase
    validation. add_validation_score calls TPB.update_pb_scores (looked up on the module at call
    time) before it decides on a switch, and a switch discards the per-neuron values, so wrapping
    it is the only place that sees the last dendrite-phase epoch. Never raises."""
    try:
        from perforatedbp import tracker_pbp as TPB
        if getattr(TPB.update_pb_scores, '_dumps_neurons', False):
            return
        original = TPB.update_pb_scores

        def update_pb_scores(*args, **kwargs):
            result = original(*args, **kwargs)
            if in_dendrite_phase():
                save_neuron_pb_scores()
            return result

        update_pb_scores._dumps_neurons = True
        TPB.update_pb_scores = update_pb_scores
    except Exception as e:  # noqa: BLE001
        print(f"=> could not hook per-neuron PB scores: {e!r}")


def save_neuron_pb_scores(folder='pai_neurons'):
    """Save every neuron's running-best |Pearson| (PB score) per perforated module to
    <log dir>/pai_neurons/val<k>.npz. Called from _hook_neuron_pb_scores after PAI's update at
    the k-th dendrite-phase validation, so PAI's module score (pai/pb_best/<module>) is the max
    over neurons of the saved values. Never raises: a failure here must not stop a run."""
    global _neuron_dump_count
    try:
        import numpy as np
        arrays = {}
        for module in GPA.pai_tracker.neuron_module_vector:
            values = module.dendrite_module.dendrite_values[0]
            arrays[short_module_name(module.name)] = (
                values.best_pearson_correlation.detach().float().abs().flatten().cpu().numpy())
        _neuron_dump_count += 1
        os.makedirs(folder, exist_ok=True)
        np.savez(os.path.join(folder, f'val{_neuron_dump_count}.npz'), **arrays)
    except Exception as e:  # noqa: BLE001
        print(f"=> could not save per-neuron PB scores: {e!r}")


def warmup_lr(optimizer, args, step):
    # ReduceLROnPlateau only lowers the LR at validations, and PAI rebuilds the optimizer only
    # after a phase switch, so overwriting the LR for the first steps is safe as long as the
    # warmup ends before the first plateau or switch (warmup < pai_n_epochs_to_switch epochs).
    for group in optimizer.param_groups:
        group['lr'] = args.lr * step / args.pai_warmup


def in_dendrite_phase():
    return GPA.pai_tracker.member_vars.get('mode') == 'p'


# Best PB score so far of each perforated module after the latest dendrite-phase validation
# (the numbers in PAI's Best PBScores csv and plot), keyed like 'pai/pb_best/layer11.mlp'.
_pb_scores = {}


def short_module_name(name):
    return name.removeprefix('.').removeprefix('encoder.layers.').replace('encoder_layer_', 'layer')


def stats():
    member_vars = GPA.pai_tracker.member_vars
    return {
        'pai/mode': 1 if in_dendrite_phase() else 0,
        'pai/dendrites_added': member_vars.get('num_dendrites_added', 0),
        # PAI's count excludes candidate-dendrite scaffolding, so it is the real model size.
        'pai/params': (member_vars.get('param_counts') or [0])[-1],
        **_pb_scores,
    }
