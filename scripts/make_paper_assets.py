"""Generate everything the manuscript needs from the result files: LaTeX macros (every number in the prose), tables, bibliography.

Outputs (paper/generated/): numbers.tex, numbers_provenance.json, tab_*.tex, refs.tex
Run: python -I scripts/make_paper_assets.py
No number in paper/manuscript.tex is typed by hand: results come from results/runs/*/result.json (via resultsdb.py), data facts from
results/data_summary.csv and configs/dataset_events.json, latency from results/tables/latency_cpu.csv, settings from configs/*.yaml,
numbers quoted from other papers from paper/literature_values.json.
"""
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resultsdb import DS, ROOT, load, mean_std  # noqa: E402

R, INFO = load()
GEN = ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
MAC = {}
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def W(s: str) -> str:
    return "".join(DIG.get(c, c) for c in s if c.isalnum())


def put(name, value):
    MAC[W(name)] = str(value)


def sg(x, nd=1):
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$")


def num(x, nd=1):
    s = f"{x:.{nd}f}"
    return s.replace("-", "$-$")


DSN = {"l4s": "Lfs", "hrgldd": "Hrg", "gdcld": "Gdc", "lmhld": "Lmh", "cas": "Cas"}
DSL = {"l4s": "L4S", "hrgldd": "HR-GLDD", "gdcld": "GDCLD", "lmhld": "LMHLD", "cas": "CAS"}
MN = {"unet": "Unet", "seg": "Seg", "adp": "Adp", "kd": "Kd"}
MET = {"iou_landslide": ("Iou", 100, 1), "f1": ("Fone", 100, 1), "precision": ("Prec", 100, 1), "recall": ("Rec", 100, 1),
       "miou": ("Miou", 100, 1), "ece": ("Ece", 1, 3), "pr_auc": ("Pr", 1, 3)}


def vals(fam, meth, ds, key, scale):
    return {s: scale * m[key] for s, m in R.get((fam, meth, ds), {}).items()}


def all_per_seed(fam, meth, key, scale):
    seeds = set.intersection(*[set(R.get((fam, meth, d), {})) for d in DS]) if all((fam, meth, d) in R for d in DS) else set()
    return {s: float(np.mean([scale * R[(fam, meth, d)][s][key] for d in DS])) for s in sorted(seeds)}


def stat_macros(prefix, v, nd):
    """v: {seed: value}. Emits M, S, N, P and Z (seed 0) macros."""
    if not v:
        return
    m, sd, n = mean_std(v.values())
    put(prefix + "M", f"{m:.{nd}f}")
    put(prefix + "S", f"{sd:.{nd}f}" if sd is not None else "n/a")
    put(prefix + "N", n)
    put(prefix + "P", f"{m:.{nd}f} $\\pm$ {sd:.{nd}f}" if sd is not None else f"{m:.{nd}f}")
    if 0 in v:
        put(prefix + "Z", f"{v[0]:.{nd}f}")


# ---------------------------------------------------------------- B-strict macros
for meth in MN:
    for key, (mn, scale, nd) in MET.items():
        for ds in DS:
            stat_macros(f"{MN[meth]}{mn}{DSN[ds]}", vals("strict", meth, ds, key, scale), nd)
        stat_macros(f"{MN[meth]}{mn}All", all_per_seed("strict", meth, key, scale), nd)

# teacher (seed 0 only; stored inside the kd runs)
teach = {}
for ds in DS:
    i = INFO.get(("strict", "kd", ds), {}).get(0)
    if i and i["teacher"]:
        teach[ds] = i["teacher"]
        for key, (mn, scale, nd) in MET.items():
            put(f"Teach{mn}{DSN[ds]}Z", f"{scale * i['teacher']['test'][ds][key]:.{nd}f}")
if len(teach) == 5:
    for key, (mn, scale, nd) in MET.items():
        put(f"Teach{mn}AllZ", f"{np.mean([scale * teach[d]['test'][d][key] for d in DS]):.{nd}f}")
    t_all = float(np.mean([100 * teach[d]["test"][d]["iou_landslide"] for d in DS]))
    s_all0 = all_per_seed("strict", "seg", "iou_landslide", 100).get(0)
    put("TeachMinusSegZ", sg(t_all - s_all0))
    t0 = teach["l4s"]
    put("TeachParamsM", f"{t0['n_params'] / 1e6:.1f}")
    put("TeachTrainableM", f"{t0['n_trainable'] / 1e6:.2f}")
    put("TeachTrainPct", f"{100 * t0['n_trainable'] / t0['n_params']:.1f}")
    put("TeachLoraRank", t0["lora_rank"])
    seg_p = INFO[("strict", "seg", "l4s")][0]["n_params"]
    put("SegParamsM", f"{seg_p / 1e6:.1f}")
    put("ParamRatioTeachSeg", f"{t0['n_params'] / seg_p:.1f}")
    put("TeachMedTrainS", f"{statistics.median(teach[d]['train_seconds'] for d in DS):.0f}")

# paired differences of seed-means per dataset
def seedmean(meth, ds):
    v = vals("strict", meth, ds, "iou_landslide", 100)
    return float(np.mean(list(v.values()))) if v else None


def diffs(a, b):
    d = {ds: seedmean(a, ds) - seedmean(b, ds) for ds in DS if seedmean(a, ds) is not None and seedmean(b, ds) is not None}
    key = f"Diff{MN[a]}{MN[b]}"
    for ds, x in d.items():
        put(f"{key}{DSN[ds]}", sg(x))
    if d:
        put(f"{key}Mean", sg(float(np.mean(list(d.values())))))
        put(f"{key}Pos", sum(x > 0 for x in d.values()))
        put(f"{key}Cnt", len(d))
    return d


D_SEG_UNET, D_ADP_SEG, D_KD_SEG = diffs("seg", "unet"), diffs("adp", "seg"), diffs("kd", "seg")

# ---------------------------------------------------------------- overlap control (single seed 0)
for fam, tag in (("plainB", "Plain"), ("sized", "Sized")):
    for key, (mn, scale, nd) in MET.items():
        for ds in DS:
            v = vals(fam, "seg", ds, key, scale)
            if 0 in v:
                put(f"{tag}{mn}{DSN[ds]}Z", f"{v[0]:.{nd}f}")
OVL = {}
HAVE_COMP = all(("comp", "seg", d) in R for d in DS)
for key, (mn, scale, nd) in MET.items():                         # composition-matched control (seeds 0..2)
    for ds in DS:
        stat_macros(f"Comp{mn}{DSN[ds]}", vals("comp", "seg", ds, key, scale), nd)
    stat_macros(f"Comp{mn}All", all_per_seed("comp", "seg", key, scale), nd)
for ds in DS:
    p = vals("plainB", "seg", ds, "iou_landslide", 100).get(0)
    z = vals("sized", "seg", ds, "iou_landslide", 100).get(0)
    c = vals("comp", "seg", ds, "iou_landslide", 100).get(0)
    st = vals("strict", "seg", ds, "iou_landslide", 100).get(0)
    OVL[ds] = (p, z, c, st)
    if None not in (p, z, st):
        put(f"OvlTotal{DSN[ds]}", f"{p - st:.1f}")
        put(f"OvlData{DSN[ds]}", f"{p - z:.1f}")
    if z is not None and st is not None:
        put(f"OvlSizedMinusStrict{DSN[ds]}", sg(z - st))
    if None not in (z, c):
        put(f"OvlSource{DSN[ds]}", num(z - c))                     # seed 0: sized plain B minus composition-matched = effect of losing the other source datasets
    if None not in (c, st):
        put(f"OvlEvents{DSN[ds]}", sg(c - st))                     # seed 0: composition-matched -> B-strict = removing event-sharing tiles only
    cm, sm = vals("comp", "seg", ds, "iou_landslide", 100), vals("strict", "seg", ds, "iou_landslide", 100)
    if cm and sm:                                                  # seed means (all available seeds)
        d_ = float(np.mean(list(cm.values()))) - float(np.mean(list(sm.values())))
        put(f"OvlEventsSeeds{DSN[ds]}", sg(d_))
        put(f"OvlEventsSeedsAbs{DSN[ds]}", f"{abs(d_):.1f}")
if HAVE_COMP:
    pairs = {ds: float(np.mean(list(vals("comp", "seg", ds, "iou_landslide", 100).values()))) - float(np.mean(list(vals("strict", "seg", ds, "iou_landslide", 100).values()))) for ds in DS}
    put("OvlEventsSeedsPos", sum(v > 0 for v in pairs.values()))
    put("OvlEventsSeedsMean", sg(float(np.mean(list(pairs.values())))))

# ---------------------------------------------------------------- in-dataset (protocol A, single seed)
IND = {"l4s14_unet": ("Lfs", "Fourteen", "Unet"), "l4s14_deeplab": ("Lfs", "Fourteen", "Deeplab"), "l4s14_seg": ("Lfs", "Fourteen", "Seg"),
       "l4srgb_unet": ("Lfs", "Rgb", "Unet"), "l4srgb_seg": ("Lfs", "Rgb", "Seg"), "l4sbank_adp": ("Lfs", "Bank", "Adp"),
       "gdcld_unet": ("Gdc", "Rgb", "Unet"), "gdcld_deeplab": ("Gdc", "Rgb", "Deeplab"), "gdcld_seg": ("Gdc", "Rgb", "Seg"),
       "gdcld_seg60": ("Gdc", "RgbSixty", "Seg")}
for meth, (d, inp, mod) in IND.items():
    ds = "l4s" if d == "Lfs" else "gdcld"
    if ("inds", meth, ds) not in R or 0 not in R[("inds", meth, ds)]:
        continue
    m = R[("inds", meth, ds)][0]
    i = INFO[("inds", meth, ds)][0]
    for key, (mn, scale, nd) in MET.items():
        put(f"In{d}{inp}{mod}{mn}", f"{scale * m[key]:.{nd}f}")
    put(f"In{d}{inp}{mod}ParamsM", f"{i['n_params'] / 1e6:.1f}")
    put(f"In{d}{inp}{mod}Gflops", f"{i['gflops']:.1f}")
    put(f"In{d}{inp}{mod}ValIou", f"{100 * i['best_val_iou']:.1f}")
    put(f"In{d}{inp}{mod}BestEpoch", i["best_epoch"])
    miou, iou = m["miou"], m["iou_landslide"]
    iou_bg = 2 * miou - iou
    f1_bg = 2 * iou_bg / (1 + iou_bg)
    put(f"In{d}{inp}{mod}MacroFone", f"{100 * (m['f1'] + f1_bg) / 2:.1f}")        # derived: mean of landslide and background F1
    put(f"In{d}{inp}{mod}BgIou", f"{100 * iou_bg:.1f}")

# ---------------------------------------------------------------- data facts
dsum = pd.read_csv(ROOT / "results" / "data_summary.csv").set_index("dataset")
put("TilesAll", f"{int(dsum.tiles.sum()):,}".replace(",", "{,}"))
for ds in DS:
    put(f"Tiles{DSN[ds]}", f"{int(dsum.loc[ds, 'tiles']):,}".replace(",", "{,}"))
    put(f"Pos{DSN[ds]}", f"{dsum.loc[ds, 'pos_pixel_ratio_pct']:.1f}")
ev = json.loads((ROOT / "configs" / "dataset_events.json").read_text(encoding="utf-8"))
EVS = {d: (set(ev[d]["group_to_event"].values()) if ev[d]["tile_events"] else set(ev[d]["events"])) for d in DS}
allev = sorted(set().union(*EVS.values()))
cnt = {e: sum(e in EVS[d] for d in DS) for e in allev}
put("EvTotal", len(allev))
put("EvTwoPlus", sum(c >= 2 for c in cnt.values()))
put("EvThreePlus", sum(c >= 3 for c in cnt.values()))
put("EvAllFive", sum(c == 5 for c in cnt.values()))
bs = json.loads((ROOT / "splits" / "protocol_B_strict.json").read_text())
bp = json.loads((ROOT / "splits" / "protocol_B.json").read_text())
for ds in DS:
    put(f"TrainStrict{DSN[ds]}", f"{len(bs['holdout_' + ds]['train']):,}".replace(",", "{,}"))
    put(f"TrainPlain{DSN[ds]}", f"{len(bp['holdout_' + ds]['train']):,}".replace(",", "{,}"))
    put(f"TestN{DSN[ds]}", f"{len(bs['holdout_' + ds]['test']):,}".replace(",", "{,}"))
put("TrainStrictMin", f"{min(len(bs['holdout_' + d]['train']) for d in DS):,}".replace(",", "{,}"))
put("TrainStrictMax", f"{max(len(bs['holdout_' + d]['train']) for d in DS):,}".replace(",", "{,}"))
put("TrainPlainMin", f"{min(len(bp['holdout_' + d]['train']) for d in DS):,}".replace(",", "{,}"))
put("TrainPlainMax", f"{max(len(bp['holdout_' + d]['train']) for d in DS):,}".replace(",", "{,}"))

# ---------------------------------------------------------------- efficiency
lat = pd.read_csv(ROOT / "results" / "tables" / "latency_cpu.csv")
LATN = {"U-Net R34": "Unet", "DeepLabV3+ R34": "Deeplab", "SegFormer-B0": "Seg", "SegFormer-B0 + band adapter": "Adp", "SegFormer-B3 (teacher)": "Teach"}
tmax = int(lat.threads.max())
for _, r in lat.iterrows():
    tag = "Four" if int(r.threads) == tmax else "One"
    put(f"Lat{LATN[r.model]}Px{int(r.input_px)}T{tag}", f"{r.median_ms:.0f}")
    if int(r.threads) == tmax:
        put(f"Gfl{LATN[r.model]}Px{int(r.input_px)}", f"{r.GFLOPs:.2f}")
        put(f"Par{LATN[r.model]}", f"{r.params_M:.1f}")
        put(f"Mb{LATN[r.model]}", f"{r.fp32_MB:.0f}")
put("LatRatioUnetSeg", f"{float(lat[(lat.model == 'U-Net R34') & (lat.input_px == 128) & (lat.threads == tmax)].median_ms.iloc[0]) / float(lat[(lat.model == 'SegFormer-B0') & (lat.input_px == 128) & (lat.threads == tmax)].median_ms.iloc[0]):.1f}")
put("LatRatioTeachSeg", f"{float(lat[(lat.model == 'SegFormer-B3 (teacher)') & (lat.input_px == 128) & (lat.threads == tmax)].median_ms.iloc[0]) / float(lat[(lat.model == 'SegFormer-B0') & (lat.input_px == 128) & (lat.threads == tmax)].median_ms.iloc[0]):.1f}")
put("LatRuns", int(lat.runs.min()))

# training time per fold (B-strict, seed-level median of fold times)
for meth in ("seg", "unet", "adp", "kd"):
    t = [i["train_seconds"] for ds in DS for i in INFO.get(("strict", meth, ds), {}).values()]
    if t:
        put(f"Time{MN[meth]}MedS", f"{statistics.median(t):.0f}")

# ---------------------------------------------------------------- GDCLD validation vs test (ours) and literature
for meth, mn in (("gdcld_unet", "Unet"), ("gdcld_deeplab", "Deeplab"), ("gdcld_seg", "Seg")):
    i = INFO[("inds", meth, "gdcld")][0]
    t = R[("inds", meth, "gdcld")][0]["iou_landslide"]
    put(f"GdcValTest{mn}Gap", f"{100 * (i['best_val_iou'] - t):.1f}")
lit = json.loads((ROOT / "paper" / "literature_values.json").read_text(encoding="utf-8"))
e = lit["essd_gdcld"]
for k in ("val_segformer_miou", "val_segformer_f1", "val_unet_miou", "val_unet_f1", "test_segformer_miou", "test_segformer_f1",
          "test_segformer_precision", "test_segformer_recall", "test_miou_min", "epochs", "patch_px"):
    put("LitEssd" + "".join(p.capitalize() for p in k.split("_")), e[k])
put("LitEssdValTestGap", f"{e['val_segformer_miou'] - e['test_segformer_miou']:.2f}")
d = float(MAC[W('InGdcRgbSegMiou')]) - e["test_segformer_miou"]
put("GdcOursMinusEssdTestMiou", sg(d))
put("GdcOursMinusEssdTestMioutwo", sg(d))
for row in lit["li2025_table3_l4s_test"]["rows"]:
    put("LitLi" + W(row["model"]) + "Miou", row["miou"])
    put("LitLi" + W(row["model"]) + "Fone", row["f1"])
for row in lit["szwarcman_table7_l4s_test"]["rows"]:
    put("LitPrithviSeven" + W(row["model"].split()[0]) + "Miou", row["miou"])
    put("LitPrithviSeven" + W(row["model"].split()[0]) + "Fone", row["f1"])
put("LitVuFone", lit["vu2026_clay"]["f1_unet_clay_lora"])
put("LitVuStd", lit["vu2026_clay"]["f1_std"])
put("LitVuBase", lit["vu2026_clay"]["f1_unet"])
put("LitNavaEvents", lit["nava2026_sar"]["train_events"])
put("LitNavaF", lit["nava2026_sar"]["f1_max_unseen"])
put("LitPrakashMcc", f"{lit['prakash2021']['mcc_unseen_regions']:.2f}")
put("LitKowsherLora", lit["kowsher2026"]["lora_trainable_reduction_pct_max"])

# ---------------------------------------------------------------- per-region (size-matched run, seed 0)
REG = {}
for ds in ("cas", "gdcld", "lmhld"):
    i = INFO.get(("sized", "seg", ds), {}).get(0)
    if i and i["by_group"]:
        REG[ds] = {g: 100 * m["iou_landslide"] for g, m in i["by_group"].items()}
        for g, v in REG[ds].items():
            put(f"Reg{DSN[ds]}{W(g)}", f"{v:.1f}")
        rest = sorted(REG[ds].values())[1:]                     # all but the lowest region/event
        put(f"Reg{DSN[ds]}RestMin", f"{min(rest):.1f}")
        put(f"Reg{DSN[ds]}RestMax", f"{max(rest):.1f}")
        put(f"Reg{DSN[ds]}Lowest", f"{min(REG[ds].values()):.1f}")

# superseded runs (wrong HR-GLDD channel order), kept only to quantify the effect of the fix
for tag, run in (("Seg", "xd_Bstrict_segformer_b0__holdout_hrgldd__seed0"), ("Unet", "xd_Bstrict_unet__holdout_hrgldd__seed0")):
    f = ROOT / "results" / "runs_superseded" / run / "result.json"
    if f.exists():
        r_ = json.loads(f.read_text())["test"]["hrgldd"]
        put(f"OldOrder{tag}IouHrgZ", f"{100 * r_['iou_landslide']:.1f}")

# downscale ceiling and example-tile facts (recorded values with provenance, see the JSON files)
ceil = json.loads((ROOT / "paper" / "generated" / "downscale_ceiling.json").read_text(encoding="utf-8-sig"))
put("CeilIou", f"{ceil['iou_landslide']:.1f}")
put("CeilMiou", f"{ceil['miou']:.1f}")
exs = json.loads((ROOT / "paper" / "generated" / "example_tiles.json").read_text(encoding="utf-8-sig"))
put("ExWorstMinPct", f"{100 * min(exs['worst_six']['landslide_share']):.1f}")
put("ExWorstMaxPct", f"{100 * max(exs['worst_six']['landslide_share']):.1f}")
put("ExBestIouMin", f"{min(exs['best_six']['tile_iou']):.2f}")
put("ExBestIouMax", f"{max(exs['best_six']['tile_iou']):.2f}")
put("ExWorstIouMax", f"{max(exs['worst_six']['tile_iou']):.2f}")

# ---------------------------------------------------------------- settings from configs
def cfg(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


c_seg, c_unet, c_kd = cfg("xd_Bstrict_segformer_b0__holdout_l4s.yaml"), cfg("xd_Bstrict_unet__holdout_l4s.yaml"), cfg("xd_Bstrict_kd_segformer_b0__holdout_l4s.yaml")
c_l4s, c_gdc = cfg("baseline_l4s_unet.yaml"), cfg("baseline_gdcld_unet.yaml")
put("XdEpochs", c_seg["epochs"]); put("XdBatch", c_seg["batch_size"]); put("XdCrop", c_seg["crop"]); put("XdMaxVal", f"{c_seg['max_val']:,}".replace(",", "{,}"))
put("XdLrSeg", c_seg["lr"]); put("XdLrUnet", c_unet["lr"]); put("XdWdSeg", c_seg["weight_decay"]); put("XdWdUnet", c_unet["weight_decay"])
put("KdAlpha", c_kd["kd_alpha"]); put("KdTemp", c_kd["kd_temp"]); put("KdRank", c_kd["teacher_lora_rank"]); put("KdTeacherEpochs", c_kd["teacher_epochs"])
put("IndEpochsLfs", c_l4s["epochs"]); put("IndEpochsGdc", c_gdc["epochs"]); put("IndBatch", c_l4s["batch_size"])
put("IndLrUnet", c_l4s["lr"]); put("IndWdUnet", c_l4s["weight_decay"])
seeds_all = sorted(set.intersection(*[set(R[("strict", "seg", d)]) for d in DS]))
put("NumSeeds", len(seeds_all))
put("NumRunsTotal", sum(len(v) for v in R.values()))

# ---------------------------------------------------------------- write numbers.tex
lines = ["% GENERATED by scripts/make_paper_assets.py -- do not edit by hand."]
for k in sorted(MAC):
    lines.append(f"\\newcommand{{\\{k}}}{{{MAC[k]}}}")
(GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
(GEN / "numbers_provenance.json").write_text(json.dumps(MAC, indent=1), encoding="utf-8")
print(len(MAC), "macros")


# ================================================================ tables
def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def cellstat(v, nd=1):
    if not v:
        return "n/a"
    m, sd, n = mean_std(v.values())
    if n == 1:
        return f"{m:.{nd}f}$^\\dagger$"
    return f"{m:.{nd}f}$\\pm${sd:.{nd}f}" + ("$^\\ddagger$" if n == 2 else "")


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")


METH_LAB = [("unet", "U-Net (ResNet34)"), ("seg", "SegFormer-B0"), ("adp", "SegFormer-B0 + band adapter"), ("kd", "SegFormer-B0 distilled student")]
for key, tag, scale, nd in (("iou_landslide", "iou", 100, 1), ("f1", "f1", 100, 1), ("ece", "ece", 1, 3), ("precision", "prec", 100, 1), ("recall", "rec", 100, 1)):
    t = ["\\begin{tabular}{lcccccc}", "\\toprule", "Method & " + " & ".join(DSL[d] for d in DS) + " & Mean \\\\", "\\midrule"]
    for m, lab in METH_LAB:
        cells = [cellstat(vals("strict", m, d, key, scale), nd) for d in DS]
        t.append(f"{lab} & " + " & ".join(cells) + f" & {cellstat(all_per_seed('strict', m, key, scale), nd)} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    write(f"tab_strict_{tag}.tex", "\n".join(t) + "\n")

# overlap controls: four training conditions. Plain B and size-matched are single seed (0); the other two have three seeds.
t = ["\\begin{tabular}{lcccc}", "\\toprule",
     "Held-out & Plain B & Plain B, size-matched & Composition-matched & B-strict \\\\",
     " & (all 4 other datasets) & (all 4 other datasets) & (LMHLD + CAS only) & (LMHLD + CAS only) \\\\",
     " & events kept, seed 0 & events kept, seed 0 & events kept & events removed \\\\", "\\midrule"]
f = lambda x: "n/a" if x is None else num(x)  # noqa: E731
for ds in DS:
    p, z, c, st = OVL[ds]
    t.append(f"{DSL[ds]} & {f(p)} & {f(z)} & {cellstat(vals('comp', 'seg', ds, 'iou_landslide', 100))} & {cellstat(vals('strict', 'seg', ds, 'iou_landslide', 100))} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_overlap.tex", "\n".join(t) + "\n")
t = ["\\begin{tabular}{lcccc}", "\\toprule", "Held-out & Events kept (composition-matched) & Events removed (B-strict) & Difference of means & Seed std (kept / removed) \\\\", "\\midrule"]
for ds in DS:
    cm, sm = vals("comp", "seg", ds, "iou_landslide", 100), vals("strict", "seg", ds, "iou_landslide", 100)
    if cm and sm:
        dm = float(np.mean(list(cm.values()))) - float(np.mean(list(sm.values())))
        t.append(f"{DSL[ds]} & {np.mean(list(cm.values())):.1f} & {np.mean(list(sm.values())):.1f} & {sg(dm)} & {np.std(list(cm.values()), ddof=1):.1f} / {np.std(list(sm.values()), ddof=1):.1f} \\\\")
    else:
        t.append(f"{DSL[ds]} & n/a & n/a & n/a & \\\\")
if HAVE_COMP:
    t.append("\\midrule")
    t.append(f"Mean over datasets & {MAC.get('CompIouAllM', 'n/a')} & {MAC.get('SegIouAllM', 'n/a')} & {MAC.get('OvlEventsSeedsMean', 'n/a')} & {MAC.get('CompIouAllS', 'n/a')} / {MAC.get('SegIouAllS', 'n/a')} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_overlap_seeds.tex", "\n".join(t) + "\n")

# teacher vs student
t = ["\\begin{tabular}{lcccc}", "\\toprule", "Held-out & Teacher B3+LoRA (seed 0) & Student, KD & Student, no KD (seed 0) & Student, no KD (mean of seeds) \\\\", "\\midrule"]
for ds in DS:
    tv = f"{100 * teach[ds]['test'][ds]['iou_landslide']:.1f}" if ds in teach else "n/a"
    t.append(f"{DSL[ds]} & {tv} & {cellstat(vals('strict', 'kd', ds, 'iou_landslide', 100))} & "
             f"{vals('strict', 'seg', ds, 'iou_landslide', 100).get(0, float('nan')):.1f} & {cellstat(vals('strict', 'seg', ds, 'iou_landslide', 100))} \\\\")
t.append("\\midrule")
t.append(f"Mean & {MAC.get('TeachIouAllZ', 'n/a')} & {cellstat(all_per_seed('strict', 'kd', 'iou_landslide', 100))} & "
         f"{MAC.get('SegIouAllZ', 'n/a')} & {cellstat(all_per_seed('strict', 'seg', 'iou_landslide', 100))} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_kd.tex", "\n".join(t) + "\n")

# in-dataset
t = ["\\begin{tabular}{llllrrrrrrrr}", "\\toprule", "Dataset & Input & Model & Params (M) & GFLOPs & mIoU & IoU & F1 & Prec. & Rec. & PR-AUC & ECE \\\\", "\\midrule"]
ORDER = [("l4s14_unet", "L4S", "14 bands", "U-Net R34"), ("l4s14_deeplab", "L4S", "14 bands", "DeepLabV3+ R34"), ("l4s14_seg", "L4S", "14 bands", "SegFormer-B0"),
         ("l4srgb_unet", "L4S", "RGB", "U-Net R34"), ("l4srgb_seg", "L4S", "RGB", "SegFormer-B0"), ("l4sbank_adp", "L4S", "bank (9 of 11 slots)", "SegFormer-B0 + adapter"),
         ("gdcld_unet", "GDCLD", "RGB", "U-Net R34"), ("gdcld_deeplab", "GDCLD", "RGB", "DeepLabV3+ R34"), ("gdcld_seg", "GDCLD", "RGB", "SegFormer-B0"),
         ("gdcld_seg60", "GDCLD", "RGB", "SegFormer-B0, 60 epochs")]
for meth, dl, inp, lab in ORDER:
    ds = "l4s" if dl == "L4S" else "gdcld"
    if ("inds", meth, ds) not in R:
        continue
    m, i = R[("inds", meth, ds)][0], INFO[("inds", meth, ds)][0]
    t.append(f"{dl} & {inp} & {lab} & {i['n_params'] / 1e6:.1f} & {i['gflops']:.1f} & " + " & ".join(f"{100 * m[k]:.1f}" for k in ("miou", "iou_landslide", "f1", "precision", "recall"))
             + f" & {m['pr_auc']:.3f} & {m['ece']:.3f} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_inds.tex", "\n".join(t) + "\n")

# latency
t = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Model & Params (M) & Weights fp32 (MB) & GFLOPs, 128 px & 128 px, 1 thread (ms) & 128 px, 4 threads (ms) & 256 px, 4 threads (ms) \\\\", "\\midrule"]
for lab, tag in LATN.items():
    t.append(f"{esc(lab)} & {MAC[W('Par' + tag)]} & {MAC[W('Mb' + tag)]} & {MAC[W('Gfl' + tag + 'Px128')]} & {MAC[W('Lat' + tag + 'Px128TOne')]} & {MAC[W('Lat' + tag + 'Px128TFour')]} & {MAC[W('Lat' + tag + 'Px256TFour')]} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_latency.tex", "\n".join(t) + "\n")

# data summary
def clean_split(s):
    return "none (own region split)" if "none" in s else s.split(" ")[0]


def clean_ev(s):
    s = str(s)
    return "not exposed" if s.startswith("unknown") else ("4 (test only)" if "test crops" in s else s)


CITE = {"l4s": "ghorbanzadeh2022", "hrgldd": "meena2023hrgldd", "gdcld": "fang2024gdcld", "lmhld": "liu2025lmhld", "cas": "xu2024cas"}
LIC = {d: re.sub(r"\s*\(.*\)", "", str(dsum.loc[d, "licence"])).replace("-", " ").replace("BY NC", "BY-NC") for d in DS}
t = ["\\begin{tabular}{llrrlrlll}", "\\toprule", "Dataset & Source & Tiles & Tile (px) & Bands & Official split & Landslide px (\\%) & Events/regions & Licence \\\\", "\\midrule"]
for ds in DS:
    r = dsum.loc[ds]
    bands = f"{int(r.channels)}"
    t.append(f"{DSL[ds]} & \\cite{{{CITE[ds]}}} & {MAC[W('Tiles' + DSN[ds])]} & {r.tile_px.replace('x', '$\\times$')} & {bands} & {esc(clean_split(r.official_split))} & {r.pos_pixel_ratio_pct:.1f} & {esc(clean_ev(r.n_events_or_regions))} & {esc(LIC[ds])} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_data.tex", "\n".join(t) + "\n")

# per-seed values
t = ["\\begin{tabular}{llccc}", "\\toprule", "Method & Held-out & Seed 0 & Seed 1 & Seed 2 \\\\", "\\midrule"]
for m, lab in METH_LAB:
    for ds in DS:
        v = vals("strict", m, ds, "iou_landslide", 100)
        t.append(f"{lab} & {DSL[ds]} & " + " & ".join(f"{v[s]:.1f}" if s in v else "--" for s in (0, 1, 2)) + " \\\\")
    t.append("\\midrule")
t[-1] = "\\bottomrule"
t.append("\\end{tabular}")
write("tab_perseed.tex", "\n".join(t) + "\n")

# literature context
t = ["\\begin{tabular}{llllrr}", "\\toprule", "Source & Dataset / split & Model & Input & mIoU & F1 \\\\", "\\midrule"]
for r in lit["li2025_table3_l4s_test"]["rows"]:
    if r["model"] in ("U-Net", "SegFormer", "Prithvi-EO-2.0-300M"):
        t.append(f"\\cite{{li2025prithvi}} & L4S test & {esc(r['model'])} & see source & {r['miou']:.2f} & {r['f1']:.2f} \\\\")
t.append(f"\\cite{{szwarcman2024prithvi}} & L4S test & U-Net (ResNet50) & 6 bands & {lit['szwarcman_table7_l4s_test']['rows'][0]['miou']:.1f} & {lit['szwarcman_table7_l4s_test']['rows'][0]['f1']:.1f} \\\\")
e = lit["essd_gdcld"]
t.append(f"\\cite{{fang2024gdcld}} & GDCLD validation & SegFormer & RGB 1024 px & {e['val_segformer_miou']:.2f} & {e['val_segformer_f1']:.2f} \\\\")
t.append(f"\\cite{{fang2024gdcld}} & GDCLD validation & U-Net & RGB 1024 px & {e['val_unet_miou']:.2f} & {e['val_unet_f1']:.2f} \\\\")
t.append(f"\\cite{{fang2024gdcld}} & GDCLD test (4 events) & SegFormer & RGB 1024 px & {e['test_segformer_miou']:.2f} & {e['test_segformer_f1']:.2f} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_literature.tex", "\n".join(t) + "\n")

# event assignment matrix
t = ["\\begin{tabular}{l" + "c" * len(DS) + "}", "\\toprule", "Event & " + " & ".join(DSL[d] for d in DS) + " \\\\", "\\midrule"]
for e_ in sorted(allev, key=lambda x: (-cnt[x], x)):
    cells = []
    for d in DS:
        cells.append(("T" if ev[d]["tile_events"] else "D") if e_ in EVS[d] else "")
    t.append(f"{esc(e_)} & " + " & ".join(cells) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_events.tex", "\n".join(t) + "\n")

# training composition per fold (B and B-strict)
from collections import Counter  # noqa: E402
t = ["\\begin{tabular}{llrrrrr}", "\\toprule", "Protocol & Held-out & \\multicolumn{5}{c}{Training tiles from source dataset} \\\\",
     " & & " + " & ".join(DSL[d] for d in DS) + " \\\\", "\\midrule"]
for lab, spl in (("B", bp), ("B-strict", bs)):
    for ds in DS:
        c = Counter(r[0] for r in spl["holdout_" + ds]["train"])
        t.append(f"{lab} & {DSL[ds]} & " + " & ".join(f"{c.get(d, 0):,}".replace(",", "{,}") if c.get(d, 0) else "--" for d in DS) + " \\\\")
    if lab == "B":
        t.append("\\midrule")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_composition.tex", "\n".join(t) + "\n")
srcs = {ds: sorted({r[0] for r in bs["holdout_" + ds]["train"]}) for ds in DS}
put("StrictOnlyLmhCas", "yes" if all(set(v) <= {"lmhld", "cas"} for v in srcs.values()) else "no")

# hyper-parameters
t = ["\\begin{tabular}{lll}", "\\toprule", "Setting & Cross-dataset runs (B, B-strict) & In-dataset runs (A) \\\\", "\\midrule",
     f"Epochs & {c_seg['epochs']} & {c_l4s['epochs']} (L4S), {c_gdc['epochs']} (GDCLD) \\\\",
     f"Batch size & {c_seg['batch_size']} & {c_l4s['batch_size']} \\\\",
     f"Input & RGB, random {c_seg['crop']} px crops at native scale & native channels (L4S), RGB (GDCLD); full tiles \\\\",
     f"Optimiser & AdamW & AdamW \\\\",
     f"Learning rate / weight decay, SegFormer-B0 & {c_seg['lr']} / {c_seg['weight_decay']} & {cfg('baseline_l4s_segformer_b0.yaml')['lr']} / {cfg('baseline_l4s_segformer_b0.yaml')['weight_decay']} \\\\",
     f"Learning rate / weight decay, U-Net & {c_unet['lr']} / {c_unet['weight_decay']} & {c_l4s['lr']} / {c_l4s['weight_decay']} \\\\",
     "Schedule & cosine annealing to zero & cosine annealing to zero \\\\",
     "Loss & cross-entropy + soft Dice & cross-entropy + soft Dice \\\\",
     "Augmentation & flips, 90$^\\circ$ rotations, random crop & flips, 90$^\\circ$ rotations \\\\",
     f"Validation set (checkpoint selection) & $\\leq${MAC[W('XdMaxVal')]} random tiles of the training datasets' validation splits & official validation split \\\\",
     "Precision & mixed (fp16 autocast) & mixed (fp16 autocast) \\\\",
     f"Teacher / distillation & \\multicolumn{{2}}{{l}}{{SegFormer-B3, LoRA rank {c_kd['teacher_lora_rank']} on 140 linear layers, {c_kd['teacher_epochs']} epochs; $\\alpha={c_kd['kd_alpha']}$, $T={c_kd['kd_temp']}$}} \\\\",
     "\\bottomrule", "\\end{tabular}"]
write("tab_hparams.tex", "\n".join(t) + "\n")

# channel statistics
cs = json.loads((ROOT / "paper" / "generated" / "channel_stats.json").read_text(encoding="utf-8-sig"))   # written by PowerShell (BOM)
t = ["\\begin{tabular}{llrrrrl}", "\\toprule", "Scene & Sensor & ch.\\ 1 & ch.\\ 2 & ch.\\ 3 & NIR & Order of first three channels \\\\", "\\midrule"]
hm = cs["hrgldd"]["means_array_channels_0_to_3"]
t.append(f"HR-GLDD (array order) & PlanetScope & {hm[0]:.3f} & {hm[1]:.3f} & {hm[2]:.3f} & {hm[3]:.3f} & ch.1 $<$ ch.3 $<$ ch.2 \\\\")
for g, v in cs["lmhld"].items():
    b, gr, rd, nir = v["means_BGRNIR"]
    order = sorted([("B", b), ("G", gr), ("R", rd)], key=lambda x: x[1])
    t.append(f"LMHLD {esc(g.split('_')[0])} (B,G,R) & {v['sensor']} & {b:.0f} & {gr:.0f} & {rd:.0f} & {nir:.0f} & {' $<$ '.join(x[0] for x in order)} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_channels.tex", "\n".join(t) + "\n")

# per-region
t = ["\\begin{tabular}{ll}", "\\toprule", "Held-out dataset & Landslide IoU by region/event (ascending) \\\\", "\\midrule"]
for ds in ("cas", "gdcld", "lmhld"):
    if ds in REG:
        items = ", ".join(f"{esc(g.replace('_', ' '))} {v:.1f}" for g, v in sorted(REG[ds].items(), key=lambda kv: kv[1]))
        t.append(f"{DSL[ds]} & {items} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("tab_regions.tex", "\n".join(t) + "\n")

# bibliography
refs = json.loads((ROOT / "paper" / "references.json").read_text(encoding="utf-8"))
b = ["\\begin{thebibliography}{99}"]
for r in refs:
    b.append(f"\\bibitem{{{r['key']}}} {r['text']}")
b.append("\\end{thebibliography}")
write("refs.tex", "\n".join(b) + "\n")
print("tables written:", sorted(p.name for p in GEN.glob("tab_*.tex")))
