"""Prepare a candidate official SAM2 training config; no training or downloads."""
import argparse,json,shutil,zipfile
from pathlib import Path
import yaml


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-zip',type=Path,default=Path('D:/segment-anything-2-real-time-main.zip'))
    args=p.parse_args()
    if args.output.exists():p.error('Use a new bundle directory')
    base=root/'sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml'
    cfg=yaml.safe_load(base.read_text())
    with zipfile.ZipFile(args.reference_zip) as archive:
        prefix='segment-anything-2-real-time-main/'
        tiny=yaml.safe_load(archive.read(prefix+'sam2/configs/sam2.1/sam2.1_hiera_t_512.yaml'))['model']
        license_text=archive.read(prefix+'LICENSE')
    # Reuse architecture only. Gy920 compile_* extensions are NOT passed to the
    # installed SAMURAI constructor. Keep official training prompt simulation.
    for key,value in tiny.items():
        if key!='_target_' and not key.startswith('compile_'):
            cfg['trainer']['model'][key]=value
    cfg['trainer']['model'].update(_target_='memory_train_adapter.MemoryOnlySAM2Train',freeze_image_encoder=True)
    cfg['scratch'].update(resolution=512,train_batch_size=1,num_train_workers=0,num_frames=4,
                          max_num_objects=1,base_lr=1e-6,vision_lr=1e-6,num_epochs=5)
    cfg['trainer']['optim']['param_group_modifiers']=[]
    cfg['trainer']['optim']['options']['lr']=cfg['trainer']['optim']['options']['lr'][:1]
    cfg['trainer']['checkpoint']['save_freq']=1
    cfg['trainer']['checkpoint']['model_weight_initializer']['state_dict']['checkpoint_path']='SET_BY_LAUNCHER'
    cfg['dataset'].update(img_folder='SET_BY_LAUNCHER',gt_folder='SET_BY_LAUNCHER',file_list_txt='SET_BY_LAUNCHER',multiplier=1)
    args.output.mkdir(parents=True)
    (args.output/'candidate.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (args.output/'LICENSE.Apache-2.0').write_bytes(license_text)
    for name in ('memory_train_adapter.py','launch_memory_training.py'):
        shutil.copy2(root/'scripts/training_tools'/name,args.output/name)
    (args.output/'STATUS.json').write_text(json.dumps(dict(status='NOT_TRAINED_NOT_VALIDATED',
        source_config=str(base.relative_to(root)),reference='https://github.com/Gy920/segment-anything-2-real-time',
        license='Apache-2.0',notes=['No checkpoint or dataset copied.','No gradient accumulation or validation early-stop implemented in this candidate.',
                                  'Use a separate Linux training environment; benchmark VRAM before a full run.']),indent=2))
    print(args.output)


if __name__=='__main__':main()
