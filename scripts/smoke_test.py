"""Check that this machine can run PAI training, without starting a real run.

    python scripts/smoke_test.py [--data DIR] [--layers 11] [--target ln_mlp] [--offline]

1. GPU: torch sees it, this torch build has kernels for it, and an FP32 forward/backward runs.
2. Credentials: WANDB_API_TOKEN, PAIEMAIL and PAITOKEN are set; the W&B key is accepted.
3. Perforation: ViT-Tiny is perforated exactly as main.py does it, and the report lists which
   encoder blocks got dendrites, so you can confirm the layer choice.
4. Dataset (with --data): DIR/train and DIR/val exist with matching class folders, e.g. after a
   job has unpacked the parquet shards to local disk.

Expects the credentials in the environment (setup_machine.sh exports .env before calling this).
Exits non-zero on the first failed check.
"""
import argparse
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def section(title):
    print(f'\n=== {title} ===', flush=True)


def fail(message):
    sys.exit(f'SMOKE TEST FAILED: {message}')


def check_gpu():
    import torch
    section('GPU')
    if not torch.cuda.is_available():
        fail('torch.cuda.is_available() is False (no GPU, or driver/torch mismatch)')
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f'sm_{major}{minor}'
    print(f'{name}, {arch}, torch {torch.__version__} (CUDA {torch.version.cuda})')
    if arch not in torch.cuda.get_arch_list():
        fail(f'this torch build has no {arch} kernels (it has {torch.cuda.get_arch_list()}); '
             're-run setup_machine.sh so it picks a matching torch build')
    # Training runs in FP32 with TF32 tensor cores (main.py --tf32), as the run script does.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    x = torch.randn(256, 256, device='cuda', requires_grad=True)
    y = (x @ x).relu().sum()
    y.backward()
    torch.cuda.synchronize()
    print('FP32/TF32 forward/backward ok')


def check_credentials(offline):
    section('Credentials')
    missing = [k for k in ('WANDB_API_TOKEN', 'PAIEMAIL', 'PAITOKEN') if not os.environ.get(k)]
    if missing:
        fail(f'not set: {", ".join(missing)} (fill them in .env)')
    print('WANDB_API_TOKEN, PAIEMAIL, PAITOKEN are set')
    if offline:
        print('--offline: skipping the W&B key check')
        return
    import wandb
    # Api().viewer checks the key without writing it to ~/.netrc, unlike wandb.login().
    try:
        user = wandb.Api(api_key=os.environ['WANDB_API_TOKEN']).viewer.username
    except Exception as e:
        fail(f'W&B rejected WANDB_API_TOKEN ({type(e).__name__})')
    print(f'W&B key accepted: user {user}, logging to entity {os.environ.get("WANDB_ENTITY") or "(default)"}')


def check_perforation(layers, target):
    import torch
    import main
    import pai_setup
    from simple_vit import SimpleVisionTransformer
    section(f'Perforation: --pai-layers {" ".join(map(str, layers))} --pai-target {target}')
    args = main.parser.parse_args([
        '--tf32', '--mlp-head', '--hidden-dim', '192', '--num-layers', '12', '--num-heads', '3',
        '--pai-layers', *map(str, layers), '--pai-target', target])
    model = SimpleVisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=3,
                                    hidden_dim=192, mlp_dim=768, representation_size=192)
    # PAI writes ./pai (config dump); keep it out of the repo. Its license check happens here.
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            model = pai_setup.perforate(model, args)
        finally:
            os.chdir(cwd)

    perforated = []
    for i, block in enumerate(model.encoder.layers):
        dendrite_modules = [n for n, m in block.named_modules() if type(m).__name__ == 'PAIDendriteModule']
        if dendrite_modules:
            perforated.append(i)
        wrapped = ', '.join(n.split('.')[0] for n in dendrite_modules) or '-'
        print(f'  encoder layer {i:2d}: {"DENDRITES on " + wrapped if dendrite_modules else "tracked only"}')
    if perforated != sorted(layers):
        fail(f'expected dendrites on layers {sorted(layers)}, got {perforated}')
    print(f'\nWrapped unit of layer {layers[0]}:')
    print(model.encoder.layers[layers[0]].mlp if target in ('mlp', 'ln_mlp') else model.encoder.layers[layers[0]])

    model = model.cuda().train()
    x = torch.randn(8, 3, 224, 224, device='cuda')
    y = torch.randint(0, 1000, (8,), device='cuda')
    _, loss = model(x, 1.0, y, y)
    loss.backward()
    print(f'perforated forward/backward ok (loss {loss.item():.3f})')


def check_data(path):
    section(f'Dataset: {path}')
    classes = {}
    for split in ('train', 'val'):
        d = os.path.join(path, split)
        if not os.path.isdir(d):
            fail(f'{d} is missing')
        classes[split] = sorted(e.name for e in os.scandir(d) if e.is_dir())
        n_images = sum(len(os.listdir(os.path.join(d, c))) for c in classes[split])
        print(f'{split}: {len(classes[split])} classes, {n_images} images')
    if classes['train'] != classes['val']:
        fail('train and val have different class folders, so ImageFolder labels would not match')
    if len(classes['val']) != 1000:
        fail(f'expected 1000 classes, found {len(classes["val"])}')


def main_():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help='ImageFolder root with train/ and val/ (optional)')
    parser.add_argument('--layers', type=int, nargs='+', default=[11])
    parser.add_argument('--target', default='ln_mlp')
    parser.add_argument('--offline', action='store_true', help='skip the W&B key check')
    args = parser.parse_args()

    check_gpu()
    check_credentials(args.offline)
    check_perforation(args.layers, args.target)
    if args.data:
        check_data(args.data)
    print('\nSMOKE TEST PASSED')


if __name__ == '__main__':
    main_()
