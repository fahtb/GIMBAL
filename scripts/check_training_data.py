import json,tempfile,unittest
from pathlib import Path
import cv2
import numpy as np
from prepare_finetune_data import validate


class Tests(unittest.TestCase):
    def folder(self,root,name,split,source,empty=False):
        folder=root/name
        (folder/'JPEGImages').mkdir(parents=True);(folder/'Annotations').mkdir()
        rows=[]
        for i in range(4):
            mask=np.zeros((8,8),np.uint8)
            absent=empty and i==3
            if not absent:mask[2:5,2:5]=1
            cv2.imwrite(str(folder/'JPEGImages'/f'{i}.jpg'),np.zeros((8,8,3),np.uint8))
            cv2.imwrite(str(folder/'Annotations'/f'{i}.png'),mask)
            rows.append(dict(frame=i,reviewed=True,absent=absent,image=f'{i}.jpg',mask=f'{i}.png'))
        (folder/'review.json').write_text(json.dumps(dict(source_id=source,kind='person',split=split,frames=rows)))
        return folder

    def test_video_disjoint_masks_and_explicit_absence(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            a=self.folder(root,'a','train','video_a',True);b=self.folder(root,'b','val','video_b')
            self.assertEqual(len(validate([a,b])),2)
            meta=json.loads((b/'review.json').read_text());meta['source_id']='video_a'
            (b/'review.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'leakage'):validate([a,b])

    def test_unreviewed_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);a=self.folder(root,'a','train','a',True);b=self.folder(root,'b','val','b')
            meta=json.loads((a/'review.json').read_text());meta['frames'][0]['reviewed']=False
            (a/'review.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Unreviewed'):validate([a,b])


if __name__=='__main__':unittest.main()
