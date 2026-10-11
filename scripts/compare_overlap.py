import json, sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
R = str(Path(__file__).resolve().parents[1] / 'results' / 'runs')
def get(run, ds):
    try: m = json.load(open(f'{R}/{run}/result.json'))['test'][ds]; return m
    except FileNotFoundError: return None
rows = []
print(f'{"held-out":9s} | {"plain B (full)":>14s} | {"B sized to strict":>17s} | {"B-strict":>9s} | IoU_landslide, SegFormer-B0, seed 0')
for ds in ('cas','gdcld','hrgldd','l4s','lmhld'):
    full = get(f'xd_B_segformer_b0__holdout_{ds}__seed0', ds); sized = get(f'xd_Bsized_segformer_b0__holdout_{ds}__seed0', ds)
    strict = get(f'xd_Bstrict_fixbgr_segformer_b0__holdout_hrgldd__seed0' if ds=='hrgldd' else f'xd_Bstrict_segformer_b0__holdout_{ds}__seed0', ds)
    f = lambda m: f'{100*m["iou_landslide"]:.1f}' if m else 'n/a'
    print(f'{ds:9s} | {f(full):>14s} | {f(sized):>17s} | {f(strict):>9s}')
print()
for ds in ('cas','gdcld','hrgldd','l4s','lmhld'):
    s = get(f'xd_Bsized_segformer_b0__holdout_{ds}__seed0', ds)
    if s: print(ds, 'sized: mIoU %.1f F1 %.1f P %.1f R %.1f PR-AUC %.3f ECE %.3f' % (100*s['miou'],100*s['f1'],100*s['precision'],100*s['recall'],s['pr_auc'],s['ece']))
    if s and 'by_group' in s:
        print('   by group IoU_ls:', {k: round(100*v['iou_landslide'],1) for k,v in s['by_group'].items()})

