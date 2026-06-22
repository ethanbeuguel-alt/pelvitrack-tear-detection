import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.signal import find_peaks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE = r"C:\Users\ethan\OneDrive\Documents\ecoles\2IA\mission r et d"

VIDEOS = [
    {"path": rf"{BASE}\C1MA.avi", "tear_start": 1340},
    {"path": rf"{BASE}\C1VA.avi", "tear_start": 1574},
    {"path": rf"{BASE}\C1VB.avi", "tear_start": 1607},
    {"path": rf"{BASE}\C2MA.avi", "tear_start": 393},
    {"path": rf"{BASE}\C3MA.avi", "tear_start": 972},
    {"path": rf"{BASE}\C3MB.avi", "tear_start": 1074},
    {"path": rf"{BASE}\C3MC.avi", "tear_start": 692},
    {"path": rf"{BASE}\C3MD.avi", "tear_start": 687},
    {"path": rf"{BASE}\C3SA.avi", "tear_start": 2031},
    {"path": rf"{BASE}\C3SB.avi", "tear_start": 2006},
    {"path": rf"{BASE}\C3VA.avi", "tear_start": 312},
    {"path": rf"{BASE}\C3VB.avi", "tear_start": 537},
    {"path": rf"{BASE}\C3VC.avi", "tear_start": 356},
]

FRAME_SKIP = 5
THRESHOLD  = 0.50
SMOOTH_WIN = 9
ROLL_WIN   = 15
CYCLE_HW   = 5

CACHE_DIR  = "cache_features_v6"
PROM_WIN   = 41
GRAPHS_DIR = "graph_finaux2"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

ACTIVE_FEATURES = [
    "img1", "img2", "entropy",
    "frames_since_end", "cycle_period",
    "diff_mean", "delta_img2",
    "roll_std_img1", "roll_std_diff",
    "edge_mean", "edge_std", "laplacian_var",
    "optical_flow", "pos_rel",
    "zone_tl", "zone_tr", "zone_bl", "zone_br",
    "img1_prom", "img1_rise",
]


def normalize(sig):
    return (sig - sig.mean()) / (sig.std() + 1e-8)

def smooth(sig, w):
    return np.convolve(sig, np.ones(w) / w, mode="same")

def diff1(sig):
    d = np.zeros_like(sig)
    d[1:] = sig[1:] - sig[:-1]
    return d

def rolling_stats(sig, w):
    padded = np.pad(sig, (w - 1, 0), mode="edge")
    wins = np.lib.stride_tricks.sliding_window_view(padded, w)
    return wins.mean(axis=1), wins.std(axis=1)

def rolling_min(sig, w):
    padded = np.pad(sig, (w - 1, 0), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, w).min(axis=1)


def frame_entropy(gray):
    h = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    h = h / (h.sum() + 1e-8)
    h = h[h > 0]
    return -np.sum(h * np.log2(h))

def edge_features(gray):
    mag = np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) ** 2 +
                  cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) ** 2)
    return mag.mean(), mag.std(), cv2.Laplacian(gray, cv2.CV_64F).var()

def optical_flow_mag(prev, curr):
    sp = cv2.resize(prev, (0, 0), fx=0.25, fy=0.25)
    sc = cv2.resize(curr, (0, 0), fx=0.25, fy=0.25)
    flow = cv2.calcOpticalFlowFarneback(sp, sc, None, 0.5, 2, 10, 2, 5, 1.2, 0)
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()


def detect_cycle_peaks(sig):
    peaks, _ = find_peaks(smooth(sig, 10), distance=80, prominence=0.05)
    if len(peaks) == 0:
        return peaks
    refined = []
    for p in peaks:
        lo = max(0, p - CYCLE_HW)
        hi = min(len(sig), p + CYCLE_HW + 1)
        refined.append(lo + int(np.argmax(sig[lo:hi])))
    return np.array(refined, dtype=int)

def build_cycle_features(img1, cycle_ends, N):
    phase, f_since, period_arr = np.zeros(N), np.zeros(N), np.zeros(N)
    prev_end, prev_period = 0, 0
    for k, ce in enumerate(cycle_ends):
        if ce >= N:
            break
        period = ce - prev_end
        idx = np.arange(prev_end, min(ce + 1, N))
        f_since[idx]    = idx - prev_end
        period_arr[idx] = prev_period if k > 0 else 0
        phase[idx]      = (idx - prev_end) / (period + 1e-8)
        prev_end, prev_period = ce, period
    tail = np.arange(prev_end, N)
    if len(tail):
        f_since[tail]    = tail - prev_end
        period_arr[tail] = prev_period
        phase[tail]      = (tail - prev_end) / (prev_period + 1e-8)
    return normalize(phase), normalize(f_since), normalize(period_arr)


def extract_features(video_path, tear_start):
    cache_file = os.path.join(CACHE_DIR, os.path.basename(video_path) + ".npz")
    if os.path.exists(cache_file):
        print(f"  [Cache] {os.path.basename(video_path)}")
        d = np.load(cache_file)
        return d["X"], d["y"], d["img1"], d["diff_mean"], d["diff_delta"], d["cycle_ends"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir : {video_path}")

    keys = ["m", "s", "ent", "diff", "em", "es", "lv", "flow", "ztl", "ztr", "zbl", "zbr"]
    raw = {k: [] for k in keys}
    prev = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mh, mw = g.shape[0] // 2, g.shape[1] // 2

        raw["m"].append(g.mean())
        raw["s"].append(g.std())
        raw["ent"].append(frame_entropy(g))
        raw["diff"].append(cv2.absdiff(g, prev).mean() if prev is not None else 0.0)
        raw["flow"].append(optical_flow_mag(prev, g)   if prev is not None else 0.0)
        raw["ztl"].append(g[:mh, :mw].mean())
        raw["ztr"].append(g[:mh, mw:].mean())
        raw["zbl"].append(g[mh:, :mw].mean())
        raw["zbr"].append(g[mh:, mw:].mean())
        em, es, lv = edge_features(g)
        raw["em"].append(em); raw["es"].append(es); raw["lv"].append(lv)
        prev = g.copy()
        frame_idx += 1

    cap.release()

    n = lambda x: normalize(np.array(x, dtype=float))
    img1      = n(raw["m"]);  img2      = n(raw["s"]);   entropy   = n(raw["ent"])
    diff_mean = n(raw["diff"]); flow    = n(raw["flow"])
    edge_mean = n(raw["em"]); edge_std  = n(raw["es"]); laplacian = n(raw["lv"])
    zone_tl   = n(raw["ztl"]); zone_tr = n(raw["ztr"])
    zone_bl   = n(raw["zbl"]); zone_br = n(raw["zbr"])
    N = len(img1)
    pos_rel = normalize(np.linspace(0, 1, N))

    cycle_ends                  = detect_cycle_peaks(img1)
    _, f_since, period_arr      = build_cycle_features(img1, cycle_ends, N)
    diff_delta                  = diff1(diff_mean)
    _, roll_s1                  = rolling_stats(img1, ROLL_WIN)
    _, roll_sd                  = rolling_stats(diff_mean, ROLL_WIN)
    img1_prom = normalize(img1 - smooth(img1, PROM_WIN))
    img1_rise = normalize(img1 - rolling_min(img1, ROLL_WIN))

    X = np.column_stack((
        img1, img2, entropy,
        f_since, period_arr,
        diff_mean, diff1(img2),
        roll_s1, roll_sd,
        edge_mean, edge_std, laplacian,
        flow, pos_rel,
        zone_tl, zone_tr, zone_bl, zone_br,
        img1_prom, img1_rise,
    ))

    y = np.zeros(N, dtype=int)
    scaled_tear = tear_start // FRAME_SKIP
    if scaled_tear >= N:
        raise ValueError(f"tear_start dépasse la vidéo : {video_path}")
    y[scaled_tear:] = 1

    np.savez(cache_file, X=X, y=y, img1=img1, diff_mean=diff_mean,
             diff_delta=diff_delta, cycle_ends=cycle_ends)

    n0, n1 = np.sum(y == 0), np.sum(y == 1)
    print(f"  {N} frames | {n0} normales / {n1} rupture | {len(cycle_ends)} cycles")
    return X, y, img1, diff_mean, diff_delta, cycle_ends

def detect_tear(y_proba, img1):
    proba_sm = smooth(y_proba, SMOOTH_WIN)

    img1_sm = smooth(img1, PROM_WIN // 2)
    d1 = diff1(img1_sm)
    d1_sm = smooth(d1, 7)

    infl_peaks, _ = find_peaks(d1_sm, prominence=0.002, distance=10)

    refined_idx = None
    for cp in sorted(infl_peaks):
        if proba_sm[int(cp)] >= THRESHOLD:
            refined_idx = int(cp)
            break

    if refined_idx is None:
        above = np.where(proba_sm >= THRESHOLD)[0]
        refined_idx = int(above[0]) if len(above) else None

    y_pred = np.zeros(len(y_proba), dtype=int)
    if refined_idx is not None:
        y_pred[refined_idx:] = 1
    return proba_sm, y_pred, refined_idx, refined_idx


def train_weights(y, cycle_ends):
    n0, n1 = np.sum(y == 0), np.sum(y == 1)
    w = np.where(y == 1, n0 / (n1 + 1e-8), 1.0)
    for ce in cycle_ends:
        lo, hi = max(0, ce - CYCLE_HW), min(len(y), ce + CYCLE_HW + 1)
        w[lo:hi] *= 3.0
    return w


all_data = []
for v in VIDEOS:
    name = os.path.basename(v["path"])
    print(f"\n{name}")
    if not os.path.exists(v["path"]):
        print(f"  IGNORÉ — fichier introuvable")
        continue
    X, y, img1, diff_mean, diff_delta, cycle_ends = extract_features(v["path"], v["tear_start"])
    all_data.append({"name": name, "X": X, "y": y, "img1": img1,
                     "diff_mean": diff_mean, "diff_delta": diff_delta, "cycle_ends": cycle_ends})

print(f"\n{len(all_data)} vidéos | {all_data[0]['X'].shape[1]} features")


accs, f1s, delays = [], [], []

for test_idx in range(len(all_data)):
    train = [d for i, d in enumerate(all_data) if i != test_idx]

    X_train = np.nan_to_num(np.vstack([d["X"] for d in train]))
    y_train = np.hstack([d["y"] for d in train])
    weights = np.hstack([train_weights(d["y"], d["cycle_ends"]) for d in train])
    X_test  = np.nan_to_num(all_data[test_idx]["X"])
    y_test  = all_data[test_idx]["y"]

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, max_features=0.6,
        learning_rate=0.05, subsample=0.8, min_samples_leaf=3, random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=weights)

    y_proba = model.predict_proba(X_test)[:, 1]
    proba_sm, y_pred, stable_idx, refined_idx = detect_tear(
        y_proba, all_data[test_idx]["img1"]
    )

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    accs.append(acc); f1s.append(f1)

    true_idx = int(np.argmax(y_test == 1)) if np.any(y_test == 1) else None
    delay = (refined_idx - true_idx) if (refined_idx is not None and true_idx is not None) else None
    if delay is not None:
        delays.append(delay)

    name = all_data[test_idx]["name"]
    print(f"\n{'='*60}")
    print(f"Test : {name}  |  Acc {acc:.4f}  |  F1 {f1:.4f}")
    print(f"Vrai début : frame {true_idx}  |  Détection : frame {refined_idx}")
    if delay is not None:
        print(f"Retard : {delay:+d} frames")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion :", confusion_matrix(y_test, y_pred))
    feat_names = ACTIVE_FEATURES if len(ACTIVE_FEATURES) == X_test.shape[1] else [f"f{i}" for i in range(X_test.shape[1])]
    print("Importance des features :")
    for nm, imp in sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {nm:<22}: {imp:.4f}")

    frames = np.arange(len(all_data[test_idx]["img1"]))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    title = (f"{name} — LOO | Acc {acc:.3f} | F1 {f1:.3f} | Retard {delay:+d} fr"
             if delay is not None else f"{name} — Leave-One-Out")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(frames, all_data[test_idx]["img1"],     label="Luminosité", alpha=0.7)
    ax.plot(frames, all_data[test_idx]["diff_mean"], label="Mouvement",  alpha=0.7)
    _lum_sm = smooth(all_data[test_idx]["img1"], 5)
    _lum_pk, _ = find_peaks(_lum_sm, prominence=0.25, distance=15)
    if len(_lum_pk):
        ax.scatter(_lum_pk, all_data[test_idx]["img1"][_lum_pk],
                   color="orange", s=25, zorder=5, label="Pic lumière")
    for ce in all_data[test_idx]["cycle_ends"]:
        ax.axvline(ce, color="purple", lw=0.6, alpha=0.5,
                   label="Pic cycle" if ce == all_data[test_idx]["cycle_ends"][0] else "")
    if true_idx is not None:
        ax.axvspan(true_idx, len(frames), alpha=0.10, color="red", label="Zone rupture GT")
        ax.axvline(true_idx,    color="red",   linestyle=":",  lw=1.5, label="Vrai début")
    if refined_idx is not None:
        ax.axvline(refined_idx, color="green", linestyle="-.", lw=1.5, label="Détection")
    ax.set_ylabel("z-score"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.fill_between(frames, y_proba, alpha=0.2, color="steelblue")
    ax.plot(frames, y_proba,  color="steelblue", alpha=0.6, label="Prob. brute")
    ax.plot(frames, proba_sm, color="navy",      lw=2,      label="Prob. lissée")
    ax.axhline(THRESHOLD, color="gray", linestyle="--", lw=1, label=f"Seuil {THRESHOLD}")
    if true_idx is not None:    ax.axvline(true_idx,    color="red",   linestyle=":",  lw=1.5)
    if refined_idx is not None: ax.axvline(refined_idx, color="green", linestyle="-.", lw=1.5)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("P(déchirure)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.step(frames, y_test, where="post", color="red",   lw=2,              label="Label réel")
    ax.step(frames, y_pred, where="post", color="green", lw=1.5, linestyle="--", label="Label prédit")
    if true_idx is not None:    ax.axvline(true_idx,    color="red",   linestyle=":",  lw=1.5)
    if refined_idx is not None: ax.axvline(refined_idx, color="green", linestyle="-.", lw=1.5)
    ax.set_ylim(-0.1, 1.3); ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Rupture"])
    ax.set_xlabel("Frames"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\n{'='*60}")
print(f"RESULTATS GLOBAUX ({len(all_data)} vidéos)")
print(f"Accuracy moy : {np.mean(accs):.4f}  (min {np.min(accs):.4f} / max {np.max(accs):.4f})")
print(f"F1 moy       : {np.mean(f1s):.4f}  (min {np.min(f1s):.4f} / max {np.max(f1s):.4f})")
if delays:
    print(f"Retard moy   : {np.mean(delays):+.1f} frames")
    print(f"Retard min   : {np.min(delays):+d} | max : {np.max(delays):+d}")
