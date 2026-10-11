"""Streaming segmentation metrics for the binary landslide task (no per-pixel storage).

Accumulates a confusion matrix at threshold 0.5 and 2 histograms of P(landslide) (for positives and negatives) from which
PR-AUC and the top-label expected calibration error (ECE, 15 equal-width confidence bins) are computed.
"""
from __future__ import annotations

import numpy as np

NBINS = 2000


class BinaryMeter:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.hist_pos = np.zeros(NBINS, np.int64)
        self.hist_neg = np.zeros(NBINS, np.int64)

    def update(self, prob: np.ndarray, target: np.ndarray) -> None:
        """prob: P(landslide) float array, target: {0,1} array of the same shape."""
        p, t = prob.reshape(-1), target.reshape(-1).astype(bool)
        pred = p >= 0.5
        self.tp += int((pred & t).sum()); self.fp += int((pred & ~t).sum())
        self.fn += int((~pred & t).sum()); self.tn += int((~pred & ~t).sum())
        b = np.minimum((p * NBINS).astype(np.int64), NBINS - 1)
        self.hist_pos += np.bincount(b[t], minlength=NBINS)
        self.hist_neg += np.bincount(b[~t], minlength=NBINS)

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-12
        iou_l, iou_b = tp / (tp + fp + fn + eps), tn / (tn + fp + fn + eps)
        prec, rec = tp / (tp + fp + eps), tp / (tp + fn + eps)
        out = dict(iou_landslide=iou_l, miou=(iou_l + iou_b) / 2, f1=2 * prec * rec / (prec + rec + eps), precision=prec, recall=rec,
                   n_pixels=int(tp + fp + fn + tn), pos_pixel_ratio=(tp + fn) / max(1, tp + fp + fn + tn))
        # PR-AUC: sweep the threshold from high to low over the histogram bins (step-wise integration of precision over recall)
        cp, cn = np.cumsum(self.hist_pos[::-1]), np.cumsum(self.hist_neg[::-1])
        P = max(1, int(self.hist_pos.sum()))
        recall_c, precision_c = cp / P, cp / np.maximum(cp + cn, 1)
        out["pr_auc"] = float(np.sum(np.diff(np.concatenate([[0.0], recall_c])) * precision_c))
        # ECE on the winning class: confidence = max(p, 1-p); accuracy = prediction correct. Uses histogram bin centres.
        centres = (np.arange(NBINS) + 0.5) / NBINS
        conf = np.maximum(centres, 1 - centres)
        n_bin = self.hist_pos + self.hist_neg
        correct = np.where(centres >= 0.5, self.hist_pos, self.hist_neg)
        edges = np.minimum(((conf - 0.5) / 0.5 * 15).astype(int), 14)
        n_tot = max(1, int(n_bin.sum()))
        ece = 0.0
        for b in range(15):
            sel = edges == b
            nb = int(n_bin[sel].sum())
            if nb:
                ece += nb / n_tot * abs(correct[sel].sum() / nb - (conf[sel] * n_bin[sel]).sum() / nb)
        out["ece"] = float(ece)
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}

    def reliability(self, nbins: int = 15) -> dict:
        """Reliability-diagram data for the winning class, using the same 15 confidence bins as the ECE. Lists of length nbins:
        pixel count, mean confidence and accuracy per bin (None where a bin is empty). ECE = sum(count/total * |acc - conf|)."""
        centres = (np.arange(NBINS) + 0.5) / NBINS
        conf = np.maximum(centres, 1 - centres)
        n_bin = self.hist_pos + self.hist_neg
        correct = np.where(centres >= 0.5, self.hist_pos, self.hist_neg)
        edges = np.minimum(((conf - 0.5) / 0.5 * nbins).astype(int), nbins - 1)
        count, acc, mconf = [], [], []
        for b in range(nbins):
            sel = edges == b
            nb = int(n_bin[sel].sum())
            count.append(nb)
            acc.append(float(correct[sel].sum() / nb) if nb else None)
            mconf.append(float((conf[sel] * n_bin[sel]).sum() / nb) if nb else None)
        return dict(count=count, accuracy=acc, confidence=mconf)
