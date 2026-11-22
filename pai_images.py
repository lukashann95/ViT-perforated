"""Preserve PAI plot revisions alongside run outputs and optionally upload to W&B."""
import hashlib
from pathlib import Path
import warnings


class PAIImages:
    def __init__(self, log_dir):
        self.root = Path(log_dir).resolve()
        self.source = self.root / 'pai'
        self.archive = self.root / 'pai_images'
        self.seen = {}
        self.live_csvs = set()

    def collect(self, step, wandb=None):
        """Return W&B media for changed plots; archive even when W&B is disabled.

        PAI's csv files (PB scores, scores, learning rates, ...) are uploaded to W&B as live
        files under pai/, re-synced whenever PAI rewrites them."""
        media = {}
        raster = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp'}
        for path in sorted(self.source.rglob('*')):
            if wandb is not None and path.is_file() and path.suffix.lower() == '.csv' \
                    and path not in self.live_csvs:
                try:
                    wandb.save(str(path), base_path=str(self.root), policy='live')
                    self.live_csvs.add(path)
                except Exception as exc:
                    warnings.warn(f'Could not upload PAI csv {path}: {exc}')
                continue
            if not path.is_file() or path.suffix.lower() not in raster | {'.svg', '.pdf'}:
                continue
            relative = path.relative_to(self.source)
            try:
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                if self.seen.get(relative) == digest:
                    continue
                snapshot = self.archive / f'step_{step:08d}' / relative
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(content)
                if wandb is not None:
                    wandb.save(str(snapshot), base_path=str(self.root), policy='now')
                    if path.suffix.lower() in raster:
                        media[f'pai/images/{relative.as_posix()}'] = wandb.Image(
                            str(snapshot), caption=f'{relative.as_posix()} (step {step})')
                self.seen[relative] = digest
            except Exception as exc:
                # Plot logging must not abort a training job. Retry on the next collection.
                warnings.warn(f'Could not archive/upload PAI plot {path}: {exc}')
        return media
