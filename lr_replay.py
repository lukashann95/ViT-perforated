"""--replay-lr: train the plain model (no PAI) on the LR schedule a PAI run followed.

A control for a Perforated Backpropagation run. It separates what the dendrites added from what
PAI's warm restarts added: at each restart PAI reloads the best model and starts a fresh
optimizer at a high LR. The schedule (a JSON file, built by scripts/build_replay_lr.py from the
PAI run's W&B history) lists the neuron-phase stretches of the PAI run that led to its final
model, in order:

    {"source_run": "<entity>/<project>/<run>", "start_step": 61000, "log_steps": 500,
     "phases": [{"load_best": true, "segments": [[63500, 64000, 0.0005], ...]}, ...]}

Each segment [a, b, lr] is the PAI run's steps a+1..b at that LR. Each phase starts with a fresh
AdamW state. A phase with load_best first reloads the best model of the cycle that just ended,
as PAI does when it switches. A phase without it keeps the weights: PAI resets the optimizer
when its LR search commits to a trial and continues from that trial's last state. Dendrite
phases (the neurons are frozen there) and abandoned LR-search trials are left out, so the
control trains for fewer steps than the PAI run.
"""
import bisect
import json
import os

import torch


class LRReplay:
    def __init__(self, path):
        with open(path) as f:
            spec = json.load(f)
        self.path = os.path.abspath(path)
        self.source_run = spec['source_run']
        self.start_step = spec['start_step']
        self.log_steps = spec['log_steps']
        self.load_best = {}  # first step of each phase -> whether it reloads the cycle's best model
        self._first, self._rows = [], []  # per segment: first step; (last step, lr, source offset)
        step = self.start_step
        for phase in spec['phases']:
            self.load_best[step + 1] = phase['load_best']
            for start, end, lr in phase['segments']:
                if end <= start:
                    raise ValueError(f'{path}: empty segment {[start, end, lr]}')
                self._first.append(step + 1)
                self._rows.append((step + end - start, lr, start - step))
                step += end - start
        if any((first - 1 - self.start_step) % self.log_steps for first in self.load_best):
            raise ValueError(f'{path}: phases must start right after a validation '
                             f'(every {self.log_steps} steps)')
        self.total_steps = step
        # Best model of the current cycle: (acc@1, state_dict on the CPU).
        self.cycle_best = None

    def _row(self, step):
        i = bisect.bisect_right(self._first, step) - 1
        if i < 0 or step > self._rows[i][0]:
            raise ValueError(f'step {step} is outside the replayed schedule')
        return self._rows[i]

    def lr(self, step):
        return self._row(step)[1]

    def source_step(self, step):
        return step + self._row(step)[2]

    def start_phase(self, step, model, optimizer):
        """At the first step of a phase: reload the cycle's best model if it asks to, and reset
        the optimizer. A no-op at other steps."""
        if step not in self.load_best:
            return
        if self.load_best[step]:
            if self.cycle_best is None:
                raise RuntimeError(f'step {step}: no validation in the cycle before this restart')
            acc1, state = self.cycle_best
            model.load_state_dict(state)
            self.cycle_best = None
            print(f'=> replay: step {step} (PAI step {self.source_step(step)}) restarts from the '
                  f"cycle's best model (acc@1 {acc1:.3f}) with a fresh optimizer")
        else:
            print(f'=> replay: step {step} (PAI step {self.source_step(step)}) resets the optimizer')
        # A fresh AdamW: its state is created again at the next step.
        optimizer.state.clear()

    def set_lr(self, step, optimizer):
        for group in optimizer.param_groups:
            group['lr'] = self.lr(step)

    def record(self, acc1, model):
        """After a validation: keep the model if it is the best of the current cycle."""
        if self.cycle_best is None or acc1 > self.cycle_best[0]:
            self.cycle_best = (acc1, {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()})

    def state_dict(self):
        return {'cycle_best': self.cycle_best}

    def load_state_dict(self, state):
        self.cycle_best = state['cycle_best']


def load_weights(model, path):
    """Load only the weights: a main.py checkpoint (plain, or a PAI run's before its first
    switch) or PAI's pai/best_model.pt."""
    import pai_setup
    if path.endswith('best_model.pt'):
        state, step = pai_setup._load_pai_file(path), None
    else:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state, step = checkpoint['state_dict'], checkpoint.get('step')
    if 'tracker_string' in state:
        state = pai_setup.plain_state_dict(state)
    model.load_state_dict(state, strict=True)
    print(f"=> loaded the weights of '{path}'" + (f' (step {step})' if step is not None else ''))
