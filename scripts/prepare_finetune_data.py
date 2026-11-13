"""Export only explicitly reviewed target masks, with video-disjoint splits."""
import argparse,json,shutil
from pathlib import Path
import cv2
import numpy as np


def validate(folders):
    sources={};items=[]
    for folder in folders:
        meta=json.loads((folder/'review.json').read_text())
        split=meta['split'];source=meta['source_id']
        if split not in ('train','val'):raise ValueError('Invalid split')
        if source in sources and sources[source]!=split:raise ValueError('Video leakage between train and val')
        sources[source]=split
        if len(meta['frames'])<4:raise ValueError('Each sequence needs >=4 reviewed frames')
        present=absent=0
        for row in meta['frames']:
            if any(Path(row[k]).name!=row[k] for k in ('image','mask')):raise ValueError('Annotation filenames must be basenames')
            if row.get('reviewed') is not True:raise ValueError('Unreviewed mask')
            image=cv2.imread(str(folder/'JPEGImages'/row['image']))
            mask=cv2.imread(str(folder/'Annotations'/row['mask']),cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape[:2]!=mask.shape:raise ValueError('Missing or mismatched annotation')
            if not set(np.unique(mask))<={0,1}:raise ValueError('Single-target mask must contain only IDs 0 and 1')
            if bool(mask.any())==bool(row['absent']):raise ValueError('Absence label contradicts mask')
            present+=bool(mask.any());absent+=not mask.any()
        if not present:raise ValueError('Sequence needs a positive prompt frame as well as negatives')
        items.append((folder,meta,present,absent))
    if {v[1]['split'] for v in items}!={'train','val'}:raise ValueError('Need separate train and val videos')
    if not any(x[3] for x in items):raise ValueError('Need reviewed absent-target frames')
    if not any(x[1]['kind']=='person' for x in items if x[1]['split']=='train'):raise ValueError('Need reviewed PERSON training sequences against forgetting')
    if not any(x[1]['kind']=='person' for x in items if x[1]['split']=='val'):raise ValueError('Need held-out PERSON regression sequence')
    return items


def main():
    p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();items=validate(args.folders)
    if args.output.exists():p.error('Output must be new; never overwrite a dataset')
    args.output.mkdir(parents=True)
    lists={'train':[],'val':[]};summary=[]
    for i,(folder,meta,present,absent) in enumerate(items):
        name=f'{meta["kind"]}_{i:04d}';lists[meta['split']].append(name)
        for sub in ('JPEGImages','Annotations'):
            dest=args.output/sub/name;dest.mkdir(parents=True)
            for row in meta['frames']:
                file=row['image' if sub=='JPEGImages' else 'mask']
                shutil.copy2(folder/sub/file,dest/file)
        summary.append(dict(sequence=name,split=meta['split'],source_id=meta['source_id'],kind=meta['kind'],present=present,absent=absent))
    for split,names in lists.items():(args.output/(split+'.txt')).write_text('\n'.join(names)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__=='__main__':main()
