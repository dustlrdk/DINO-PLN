import os
from pathlib import Path
from datasets.data_util import preparing_dataset
from datasets.coco import get_aux_target_hacks_list, CocoDetection, make_coco_transforms


def build(image_set, args):
    root = Path(args.coco_path)
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / 'train.json'),
        "train_reg": (root / "train", root / 'train.json'),
        "val": (root / "val", root / 'val.json'),
        "eval_debug": (root / "val", root / 'val.json'),
        # "test": (root / "test", root / 'image_info_test-dev2017.json' ),
    }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    dataset = CocoDetection(img_folder, ann_file,
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args),
            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
        )

    return dataset