## Setup
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install PyYAML
```

## Convert annotation from YOLO-OBB to DOTA format
```bash
python3 convert_yolo_obb_to_dota.py
```

Default split ratio is `train/val/test = 0.7/0.15/0.15`.

Custom split ratio:
```bash
python3 convert_yolo_obb_to_dota.py --split-ratios 0.8,0.1,0.1
```

## Duplicate annotation txt (`*-0.txt -> *-1.txt, *-2.txt`)
```bash
./duplicate_label_variants.sh
```

Duplicate in converted split folders (`output/train|val|test/labelTxt`):
```bash
./duplicate_label_variants.sh ./output
```
