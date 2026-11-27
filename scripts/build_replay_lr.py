"""Build a --replay-lr schedule (see lr_replay.py) from a PAI run's logged LR.

Name the neuron-phase stretches of the PAI run that led to its final model, in order, as PAI
steps A-B (steps A+1..B). 'restart:A-B' reloads the best model of the cycle before it and resets
the optimizer, as PAI does at a switch; 'continue:A-B' only resets the optimizer, as PAI does when
its LR search commits to a trial and continues from that trial's last state. The first stretch is
'continue' (there is no earlier cycle). Find the stretches in the run's log: '=> PAI restructured
the model at step N' marks every switch, LR-search trial and commit.

The LR of each validation interval is read from the run's W&B history ('lr', logged every
--print-freq steps); PAI only changes it at validations.

    python scripts/build_replay_lr.py IdlF26Group40/ViT-perforated-plateau/<run> --log-steps 500 \\
        --start-step 61000 continue:63500-89500 continue:103000-115500 restart:128000-136500 \\
        continue:140500-144000 -o scripts/replay_lr/<run>.json
"""
import argparse
import json
import os

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', help='<entity>/<project>/<run id> of the PAI run')
    parser.add_argument('stretches', nargs='+', metavar='restart|continue:A-B')
    parser.add_argument('--log-steps', type=int, required=True, help="the PAI run's --log-steps")
    parser.add_argument('--start-step', type=int, default=0,
                        help="the replay run's first step is this + 1 (default 0); e.g. the PAI "
                             "run's resume step, to share its x axis and copied history")
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    stretches = []
    for token in args.stretches:
        kind, _, span = token.partition(':')
        start, _, end = span.partition('-')
        if kind not in ('restart', 'continue') or not start.isdigit() or not end.isdigit():
            parser.error(f'bad stretch {token!r}; expected restart:A-B or continue:A-B')
        start, end = int(start), int(end)
        if end <= start or start % args.log_steps or end % args.log_steps:
            parser.error(f'{token}: needs A < B, both multiples of --log-steps {args.log_steps}')
        stretches.append((kind == 'restart', start, end))
    if stretches[0][0]:
        parser.error('the first stretch must be continue: there is no cycle before it')

    api = wandb.Api(api_key=os.environ.get('WANDB_API_TOKEN') or None, timeout=120)
    history = api.run(args.run).scan_history(keys=['_step', 'lr'], page_size=10000)
    logged = {}
    for row in history:
        if row.get('lr') is not None:
            logged[row['_step']] = row['lr']

    phases = []
    for load_best, start, end in stretches:
        segments = []
        for v in range(start, end, args.log_steps):
            lrs = {lr for step, lr in logged.items() if v < step <= v + args.log_steps}
            if len(lrs) != 1:
                raise SystemExit(f'steps {v + 1}-{v + args.log_steps}: expected one logged LR, got {sorted(lrs)}')
            lr = lrs.pop()
            if segments and segments[-1][2] == lr:
                segments[-1][1] = v + args.log_steps
            else:
                segments.append([v, v + args.log_steps, lr])
        phases.append({'load_best': load_best, 'segments': segments})

    spec = {'source_run': args.run, 'start_step': args.start_step, 'log_steps': args.log_steps,
            'stretches': args.stretches, 'phases': phases}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(spec, f, indent=1)
        f.write('\n')
    steps = sum(end - start for _, start, end in stretches)
    print(f'=> wrote {args.output}: {len(phases)} phases, {steps} steps')
    for phase in phases:
        print(('restart' if phase['load_best'] else 'continue') + ': ' + ', '.join(
            f'{a}-{b} lr {lr:.3g}' for a, b, lr in phase['segments']))


if __name__ == '__main__':
    main()
