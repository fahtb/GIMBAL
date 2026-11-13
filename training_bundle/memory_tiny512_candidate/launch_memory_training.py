"""Explicit opt-in candidate launcher; delegates training to the official runner."""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True,help='SAMURAI repo containing sam2/training')
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',action='store_true',help='without this, validate paths/config only')
    args=p.parse_args()
    if args.output.exists():p.error('Use a new experiment output; never overwrite a checkpoint')
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    splits={r['split'] for r in manifest}
    if splits!={'train','val'}:p.error('Reviewed, video-disjoint train/val data required')
    groups={}
    for row in manifest:
        key=row['source_id']
        if key in groups and groups[key]!=row['split']:p.error('Video leakage')
        groups[key]=row['split']
    if not args.checkpoint.is_file():p.error('Missing checkpoint')
    from omegaconf import OmegaConf
    sys.path.insert(0,str(args.repo.resolve()/'sam2'))
    cfg=OmegaConf.load(Path(__file__).with_name('candidate.yaml'))
    cfg.dataset.img_folder=str(args.dataset.resolve()/'JPEGImages')
    cfg.dataset.gt_folder=str(args.dataset.resolve()/'Annotations')
    cfg.dataset.file_list_txt=str(args.dataset.resolve()/'train.txt')
    cfg.launcher.experiment_log_dir=str(args.output.resolve())
    cfg.launcher.num_gpus=1;cfg.launcher.num_nodes=1
    cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path=str(args.checkpoint.resolve())
    print('Candidate config checked; training/validation accuracy and VRAM are NOT yet verified.')
    if not args.run:return
    if sys.platform=='win32':p.error('This candidate uses official NCCL runner: export to Linux, do not replace the live .venv')
    from training.train import single_node_runner,register_omegaconf_resolvers
    register_omegaconf_resolvers()
    single_node_runner(cfg,29517)


if __name__=='__main__':main()
