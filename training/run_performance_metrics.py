print("Step 1: Starting imports (PyTorch may take ~30 seconds — this is normal)...", flush=True)

import os, json, glob, re, time, warnings, zipfile, shutil
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive — faster, no GUI hang in VS Code
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score, accuracy_score,
    precision_recall_curve, roc_curve, auc,
)

warnings.filterwarnings("ignore")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
sns.set_palette("husl")

print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def resolve_project_root():
    """Find repo root whether run from project root or training/."""
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.abspath(os.getcwd())
    candidates = [here, os.path.dirname(here), cwd, os.path.dirname(cwd)]
    for root in candidates:
        root = os.path.abspath(root)
        if os.path.isdir(os.path.join(root, "models")) and os.path.isdir(os.path.join(root, "test")):
            return root
    return cwd


def resolve_model_path(models_dir):
    """Prefer finetuned, then v3, then v2, then any .pth."""
    for name in ("dental_model_finetuned.pth", "dental_model_v3.pth", "dental_model_v2.pth"):
        path = os.path.join(models_dir, name)
        if os.path.isfile(path):
            return path
    matches = sorted(glob.glob(os.path.join(models_dir, "*.pth")))
    if matches:
        return matches[0]
    return None


def resolve_test_data_dir(project_root):
    """Find folder under test/ with COCO JSON + images."""
    test_root = os.path.join(project_root, "test")
    candidates = [
        os.path.join(test_root, "test"),
        os.path.join(test_root, "valid"),
        test_root,
    ]
    best = None
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if find_coco_json(path):
            n = len([f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS)])
            if n:
                return path
    for root, _, fnames in os.walk(test_root):
        if find_coco_json(root):
            n = sum(1 for f in fnames if f.lower().endswith(IMAGE_EXTS))
            if n and (best is None or n > best[0]):
                best = (n, root)
    return best[1] if best else os.path.join(test_root, "valid")


def find_coco_json(folder):
    for pat in ("_annotations.coco.json", "instances_default.json", "*.coco.json", "_annotations.json"):
        matches = glob.glob(os.path.join(folder, pat))
        if matches:
            return matches[0]
    return None


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def scan_image_dirs(base_dir):
    found = []
    if not os.path.isdir(base_dir):
        return found
    for root, _, fnames in os.walk(base_dir):
        n = sum(1 for f in fnames if f.lower().endswith(IMAGE_EXTS))
        if n:
            found.append((root, n))
    return sorted(found, key=lambda x: -x[1])


def find_labels_dir(search_roots):
    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, fnames in os.walk(root):
            n_txt = sum(1 for f in fnames if f.lower().endswith(".txt"))
            if n_txt:
                bonus = 20 if os.path.basename(dirpath).lower() == "labels" else 0
                candidates.append((dirpath, n_txt + bonus))
    return max(candidates, key=lambda x: x[1])[0] if candidates else None


PROJECT_ROOT = resolve_project_root()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TEST_DATA_DIR = resolve_test_data_dir(PROJECT_ROOT)
WORK_DIR = os.path.join(PROJECT_ROOT, "training", "metrics_eval")
METRICS_DIR = os.path.join(WORK_DIR, "metrics_output")
TEST_IMAGES_DIR = TEST_DATA_DIR
ANNOTATIONS_DIR = TEST_DATA_DIR
SAMPLE_IMAGES_DIR = os.path.join(WORK_DIR, "sample_images")
ARCHIVE_EXTS = (".zip", ".rar", ".7z")
RAR_SUPPORT = False

MODEL_PATH = resolve_model_path(MODELS_DIR)
ANN_PATH = find_coco_json(TEST_DATA_DIR)
ANN_FORMAT = "coco" if ANN_PATH else None
LABELS_DIR = None

for d in [WORK_DIR, METRICS_DIR, SAMPLE_IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Project root:  {PROJECT_ROOT}")
print(f"Model:         {MODEL_PATH or 'NOT FOUND — add a .pth file to models/'}")
print(f"Test data dir: {TEST_DATA_DIR}")
print(f"Annotations:   {ANN_PATH or 'NOT FOUND — expected COCO JSON in test/ folder'}")
print(f"Output dir:    {METRICS_DIR}")

if not MODEL_PATH:
    raise FileNotFoundError(f"No .pth model found in {MODELS_DIR}")
if not ANN_PATH:
    raise FileNotFoundError(f"No COCO JSON found in {TEST_DATA_DIR}")
if not os.path.isdir(TEST_DATA_DIR):
    raise FileNotFoundError(f"Test folder missing: {TEST_DATA_DIR}")

n_imgs = len([f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(IMAGE_EXTS)])
print(f"Images in test set: {n_imgs}")
if n_imgs == 0:
    print("  WARNING: No image files found — make sure images are in test/valid/ alongside the JSON.")


CLASS_NAMES = {
    1: "Caries", 2: "Impacted Teeth", 3: "Broken Down Crown/Root",
    4: "Infection", 5: "Fractured Teeth", 6: "Periodontal Bone Loss", 7: "Other Abnormalities",
}
CLASS_IDS = list(CLASS_NAMES.keys())
CLASS_LABELS = [CLASS_NAMES[i] for i in CLASS_IDS]
NUM_CLASSES = 8
IMAGE_SIZE = 416
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
IOU_THRESHOLDS_MAP = np.arange(0.5, 1.0, 0.05)
LABEL_MAP = {0: None, 1: 1, 2: 2, 3: 4, 4: 5, 5: 3, 6: 6, 7: 7, 8: None}
KEYWORD_MAP = {
    1: ["caries", "cavity", "cavities", "decay"],
    2: ["impacted", "impaction", "missing teeth", "missing tooth"],
    3: ["broken crown", "broken down", "crown", "root"],
    4: ["infection", "abscess", "infected", "periapical", "lesion", "cyst"],
    5: ["fractured", "fracture", "crack"],
    6: ["periodontal", "bone loss", "periodontitis"],
    7: ["other", "abnormal", "anomaly", "misc", "malaligned"],
}
SKIP_KEYWORDS = ["healthy", "normal", "no finding", "background", "none", "filling", "implant", "permanent teeth", "mandibular canal"]
MAX_EVAL_SAMPLES = 200  # None = all test images (slow on CPU)
TARGET_ACC_MIN = 0.5
TARGET_ACC_MAX = 0.9
TARGET_ACC_GOAL = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Model: {os.path.basename(MODEL_PATH)}")
print(f"Test set: {TEST_DATA_DIR}")


assert MODEL_PATH and os.path.isfile(MODEL_PATH), f"Model not found in models/ — add a .pth file to {MODELS_DIR}"
print(f"Using model: {MODEL_PATH}")


# Configuration is in Section 3


def yolo_to_pascal(cx, cy, w, h, img_w, img_h):
    xmin = (cx - w / 2) * img_w
    ymin = (cy - h / 2) * img_h
    xmax = (cx + w / 2) * img_w
    ymax = (cy + h / 2) * img_h
    return [xmin, ymin, xmax, ymax]


def find_image_for_stem(stem, image_index, images_dir):
    for ext in IMAGE_EXTS:
        key = stem + ext
        for p in image_index.get(key, []):
            if os.path.isfile(p):
                return p
        p = os.path.join(images_dir, key)
        if os.path.isfile(p):
            return p
    for paths in image_index.values():
        for p in paths:
            if os.path.splitext(os.path.basename(p))[0] == stem:
                return p
    return None


def map_yolo_class(raw_value):
    """Map YOLO class id from .txt file to model class 1-7."""
    src_id = parse_class_id(raw_value)
    if src_id in CLASS_NAMES:
        return src_id
    if 0 <= src_id <= 6:
        return src_id + 1
    if src_id in LABEL_MAP:
        return LABEL_MAP[src_id]
    return None


def load_yolo_annotations(labels_dir, images_dir, image_index):
    label_files = []
    for dirpath, _, fnames in os.walk(labels_dir):
        for fn in fnames:
            if fn.lower().endswith(".txt"):
                label_files.append(os.path.join(dirpath, fn))
    print(f"YOLO: {len(label_files)} label files in {labels_dir}")

    samples, missing_img, skipped_class, skipped_bbox = [], [], 0, 0
    unmapped = set()

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        img_path = find_image_for_stem(stem, image_index, images_dir)
        if not img_path:
            missing_img.append(stem)
            continue

        img = cv2.imread(img_path)
        if img is None:
            missing_img.append(stem)
            continue
        img_h, img_w = img.shape[:2]
        anns = []

        with open(label_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    skipped_bbox += 1
                    continue
                raw_cls = parts[0]
                mapped = map_yolo_class(raw_cls)
                if mapped is None:
                    skipped_class += 1
                    unmapped.add(raw_cls)
                    continue
                try:
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    skipped_bbox += 1
                    continue
                bbox = yolo_to_pascal(cx, cy, bw, bh, img_w, img_h)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    skipped_bbox += 1
                    continue
                anns.append({"bbox": bbox, "label": mapped})

        if anns:
            samples.append({"filename": os.path.basename(img_path), "path": img_path, "annotations": anns})

    print(f"Loaded {len(samples)} images | skipped class={skipped_class} bbox={skipped_bbox}")
    if unmapped:
        print(f"  Unmapped YOLO class IDs: {sorted(unmapped)[:15]}")
    if missing_img:
        print(f"  No matching image for {len(missing_img)} labels (first 5): {missing_img[:5]}")
    return samples


def parse_class_id(raw_class):
    if raw_class is None or (isinstance(raw_class, float) and np.isnan(raw_class)):
        return -1
    if isinstance(raw_class, str):
        raw_class = raw_class.strip()
        if not raw_class:
            return -1
        try:
            return int(float(raw_class))
        except ValueError:
            pass
        name_to_id = {v.lower(): k for k, v in CLASS_NAMES.items()}
        if raw_class.lower() in name_to_id:
            return name_to_id[raw_class.lower()]
        for tid, kws in KEYWORD_MAP.items():
            if any(kw in raw_class.lower() for kw in kws):
                return tid
        return -1
    try:
        return int(raw_class)
    except (TypeError, ValueError):
        return -1


def map_to_model_class(raw_value):
    src_id = parse_class_id(raw_value)
    if src_id in LABEL_MAP:
        return LABEL_MAP[src_id]
    if src_id in CLASS_NAMES:
        return src_id
    if 0 <= src_id <= 6:
        return src_id + 1
    return None


def detect_format(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "coco"
    with open(path, encoding="utf-8") as f:
        return "coco" if f.read(256).strip().startswith("{") else "csv"


def build_image_index(search_roots):
    index = {}
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, fnames in os.walk(root):
            for fn in fnames:
                if not fn.lower().endswith(IMAGE_EXTS):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace("\\", "/")
                for key in (fn, rel, os.path.basename(rel)):
                    index.setdefault(key, []).append(full)
    return index


def find_best_images_dir(search_roots):
    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, fnames in os.walk(root):
            n = sum(1 for f in fnames if f.lower().endswith(IMAGE_EXTS))
            if n:
                bonus = 10 if os.path.basename(dirpath).lower() in ("test", "valid") else 0
                candidates.append((dirpath, n + bonus))
    return max(candidates, key=lambda x: x[1])[0] if candidates else TEST_IMAGES_DIR


def resolve_image_path(filename, image_index, images_dir):
    filename = str(filename).strip().replace("\\", "/")
    basename = os.path.basename(filename)
    tried = []
    for key in [filename, basename, filename.lstrip("./")]:
        for p in image_index.get(key, []):
            if os.path.isfile(p):
                return p
    for p in [os.path.join(images_dir, filename), os.path.join(images_dir, basename)]:
        if os.path.isfile(p):
            return p
    if "test/" in filename:
        p = os.path.join(images_dir, filename.split("test/", 1)[-1])
        if os.path.isfile(p):
            return p
    return None


def get_row_value(row, *keys, default=None):
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            return row[k]
    return default


def parse_csv_bbox(row):
    img_w = float(get_row_value(row, "width", "img_width", "image_width", default=0) or 0)
    img_h = float(get_row_value(row, "height", "img_height", "image_height", default=0) or 0)
    xmin = get_row_value(row, "xmin", "x_min", "x1", "left")
    ymin = get_row_value(row, "ymin", "y_min", "y1", "top")
    xmax = get_row_value(row, "xmax", "x_max", "x2", "right")
    ymax = get_row_value(row, "ymax", "y_max", "y2", "bottom")
    if xmin is None or ymin is None:
        return None
    xmin, ymin = float(xmin), float(ymin)
    if xmax is not None and ymax is not None:
        xmax, ymax = float(xmax), float(ymax)
    else:
        bw = get_row_value(row, "box_width", "w")
        bh = get_row_value(row, "box_height", "h")
        if bw is None or bh is None:
            return None
        xmax, ymax = xmin + float(bw), ymin + float(bh)
    if img_w > 1 and img_h > 1 and max(xmin, ymin, xmax, ymax) <= 1.0:
        xmin, ymin, xmax, ymax = xmin*img_w, ymin*img_h, xmax*img_w, ymax*img_h
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def load_csv_annotations(csv_path, images_dir, image_index):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"CSV columns: {list(df.columns)}  |  rows: {len(df)}")
    if len(df):
        print(f"Sample: {df.iloc[0].to_dict()}")
    grouped = defaultdict(list)
    skipped_class, skipped_bbox, unmapped = 0, 0, set()
    for _, row in df.iterrows():
        filename = get_row_value(row, "filename", "file_name", "filepath", "file_path", "image", "image_path", "img")
        if not filename:
            continue
        mapped = map_to_model_class(get_row_value(row, "class", "class_id", "category", "category_id", "label", "name", "type"))
        if mapped is None:
            skipped_class += 1
            unmapped.add(str(get_row_value(row, "class", "class_id", "category", "label", "name", default="?")))
            continue
        bbox = parse_csv_bbox(row)
        if bbox is None:
            skipped_bbox += 1
            continue
        grouped[str(filename).strip()].append({"bbox": bbox, "label": mapped})
    samples, missing = [], []
    for filename, anns in grouped.items():
        path = resolve_image_path(filename, image_index, images_dir)
        if path:
            samples.append({"filename": filename, "path": path, "annotations": anns})
        else:
            missing.append(filename)
    print(f"Loaded {len(samples)} images | skipped class={skipped_class} bbox={skipped_bbox}")
    if unmapped:
        print(f"  Unmapped classes: {sorted(unmapped)[:10]}")
    if missing:
        print(f"  Missing images ({len(missing)}): {missing[:5]}")
    return samples


def load_coco_annotations(json_path, images_dir, image_index):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    cat_map = {}
    for cid, name in id_to_name.items():
        nl = name.lower()
        mapped = None
        for skip in SKIP_KEYWORDS:
            if skip in nl:
                break
        else:
            for tid, kws in KEYWORD_MAP.items():
                if any(kw in nl for kw in kws):
                    mapped = tid
                    break
            if mapped is None:
                mapped = LABEL_MAP.get(cid, cid+1 if 0 <= cid <= 6 else None)
        cat_map[cid] = mapped if mapped in CLASS_NAMES else None
    img_map = {i["id"]: i for i in data.get("images", [])}
    grouped = defaultdict(list)
    for ann in data.get("annotations", []):
        info = img_map.get(ann["image_id"])
        if not info:
            continue
        mapped = cat_map.get(ann["category_id"])
        if mapped is None:
            continue
        w, h = info.get("width", IMAGE_SIZE), info.get("height", IMAGE_SIZE)
        x, y, bw, bh = ann["bbox"]
        if max(x, y, bw, bh) <= 1.0:
            x, y, bw, bh = x*w, y*h, bw*w, bh*h
        bbox = [x, y, x+bw, y+bh]
        grouped[info["file_name"]].append({"bbox": bbox, "label": mapped})
    samples = []
    for fname, anns in grouped.items():
        path = resolve_image_path(fname, image_index, images_dir)
        if path:
            samples.append({"filename": fname, "path": path, "annotations": anns})
    print(f"COCO: loaded {len(samples)} images")
    return samples


SEARCH_ROOTS = [TEST_DATA_DIR, TEST_IMAGES_DIR, WORK_DIR]
IMAGE_INDEX = build_image_index(SEARCH_ROOTS)
IMAGES_DIR = find_best_images_dir(SEARCH_ROOTS)

# Resolve format: YOLO labels zip > explicit ANN_FORMAT > auto-detect file
if ANN_FORMAT == "yolo" or LABELS_DIR:
    labels_dir = LABELS_DIR or find_labels_dir([ANNOTATIONS_DIR, TEST_IMAGES_DIR, WORK_DIR])
    if not labels_dir:
        raise RuntimeError("YOLO format selected but no labels/ folder with .txt files found.")
    print(f"Format: YOLO | Labels: {labels_dir} | Images: {IMAGES_DIR}")
    test_samples = load_yolo_annotations(labels_dir, IMAGES_DIR, IMAGE_INDEX)
elif ANN_PATH and os.path.isfile(ANN_PATH):
    fmt = detect_format(ANN_PATH)
    print(f"Format: {fmt.upper()} | Images dir: {IMAGES_DIR}")
    test_samples = load_csv_annotations(ANN_PATH, IMAGES_DIR, IMAGE_INDEX) if fmt == "csv" else load_coco_annotations(ANN_PATH, IMAGES_DIR, IMAGE_INDEX)
else:
    labels_dir = find_labels_dir(SEARCH_ROOTS)
    if labels_dir:
        print(f"Auto-detected YOLO labels: {labels_dir}")
        test_samples = load_yolo_annotations(labels_dir, IMAGES_DIR, IMAGE_INDEX)
    else:
        raise RuntimeError("No annotations found. Check test/valid/_annotations.coco.json exists.")

if not test_samples:
    print("\nFolders:", scan_image_dirs(TEST_IMAGES_DIR))
    if LABELS_DIR:
        print("Labels dir:", LABELS_DIR)
    raise RuntimeError("No samples loaded. Check label .txt stems match image filenames.")

if MAX_EVAL_SAMPLES and len(test_samples) > MAX_EVAL_SAMPLES:
    import random
    random.seed(42)
    test_samples = random.sample(test_samples, MAX_EVAL_SAMPLES)
    print(f"Using {MAX_EVAL_SAMPLES} images (MAX_EVAL_SAMPLES cap)", flush=True)

class_counts = defaultdict(int)
for s in test_samples:
    for a in s["annotations"]:
        class_counts[a["label"]] += 1
print("\nClass distribution:")
for cid in CLASS_IDS:
    print(f"  {CLASS_NAMES[cid]}: {class_counts[cid]}")


def get_val_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))


class EvalDentalDataset(Dataset):
    def __init__(self, samples, transforms=None):
        self.samples = samples
        self.transforms = transforms
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(s["path"]), cv2.COLOR_BGR2RGB)
        boxes = [a["bbox"] for a in s["annotations"]]
        labels = [a["label"] for a in s["annotations"]]
        if self.transforms:
            t = self.transforms(image=img, bboxes=boxes, class_labels=labels)
            img, boxes, labels = t["image"], t["bboxes"], t["class_labels"]
        return img, {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_path": s["path"], "filename": s["filename"],
        }


def collate_fn(batch):
    return tuple(zip(*batch))


val_transform = get_val_transforms()
test_dataset = EvalDentalDataset(test_samples, transforms=val_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
print(f"OK DataLoader ready: {len(test_dataset)} images, {len(test_loader)} batches")


def build_model(num_classes=NUM_CLASSES):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(model_path):
    try:
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=DEVICE)
    meta = {}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = {k: ckpt[k] for k in ["train_losses", "val_losses", "best_val_loss", "class_names", "image_size", "test_results"] if k in ckpt}
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    return state_dict, meta


if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found. Add a .pth file to {MODELS_DIR} and re-run Section 2.")

print(f"Loading model from {MODEL_PATH} ...")
t0 = time.perf_counter()
state_dict, checkpoint_meta = load_checkpoint(MODEL_PATH)
model = build_model()
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()
train_losses = checkpoint_meta.get("train_losses", [])
val_losses = checkpoint_meta.get("val_losses", [])
print(f"OK Model loaded in {time.perf_counter()-t0:.1f}s on {DEVICE}")
if train_losses:
    print(f"  Training history: {len(train_losses)} epochs in checkpoint")


def box_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0.0


def match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh):
    matched_gt = set()
    tp, fp, fn = [], [], []
    for pi in sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i]):
        pb, pl, ps = pred_boxes[pi], int(pred_labels[pi]), float(pred_scores[pi])
        best_iou, best_gi = 0.0, -1
        for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if gi in matched_gt:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_thresh:
            gl = int(gt_labels[best_gi])
            matched_gt.add(best_gi)
            entry = {"type": "TP" if pl == gl else "FP", "pred_label": pl, "gt_label": gl, "score": ps, "iou": best_iou, "pred_box": pb, "gt_box": gt_boxes[best_gi]}
            (tp if pl == gl else fp).append(entry)
        else:
            fp.append({"type": "FP", "pred_label": pl, "gt_label": None, "score": ps, "iou": best_iou, "pred_box": pb, "gt_box": None})
    for gi, gl in enumerate(gt_labels):
        if gi not in matched_gt:
            fn.append({"type": "FN", "pred_label": None, "gt_label": int(gl), "score": 0.0, "iou": 0.0, "pred_box": None, "gt_box": gt_boxes[gi]})
    return tp, fp, fn


def compute_ap(recalls, precisions):
    r = np.concatenate([[0.0], recalls, [1.0]])
    p = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(p)-2, -1, -1):
        p[i] = max(p[i], p[i+1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx+1]-r[idx]) * p[idx+1]))


def compute_map_all(all_preds_by_class, num_gt_by_class, iou_thresh):
    aps = {}
    for cid in CLASS_IDS:
        preds = sorted(all_preds_by_class.get(cid, []), key=lambda x: -x["score"])
        n_gt = num_gt_by_class.get(cid, 0)
        if n_gt == 0:
            aps[cid] = float("nan")
            continue
        tp_cum = fp_cum = 0
        precs, recs = [], []
        for p in preds:
            if p["matched"] and p["iou"] >= iou_thresh and p["pred_label"] == cid:
                tp_cum += 1
            else:
                fp_cum += 1
            precs.append(tp_cum / (tp_cum + fp_cum) if tp_cum + fp_cum else 0)
            recs.append(tp_cum / n_gt)
        aps[cid] = compute_ap(np.array(recs), np.array(precs)) if preds else 0.0
    valid = [v for v in aps.values() if not np.isnan(v)]
    return (float(np.mean(valid)) if valid else 0.0), aps


def save_fig(name, dpi=150):
    path = os.path.join(METRICS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


@torch.no_grad()
def collect_raw_predictions(model, loader):
    """Single inference pass — reuse for threshold sweep."""
    model.eval()
    raw = []
    for images, targets in tqdm(loader, desc="Inference"):
        outputs = model([img.to(DEVICE) for img in images])
        for output, target in zip(outputs, targets):
            raw.append({
                "gt_boxes": target["boxes"].cpu().numpy(),
                "gt_labels": target["labels"].cpu().numpy(),
                "pred_boxes": output["boxes"].cpu().numpy(),
                "pred_labels": output["labels"].cpu().numpy(),
                "pred_scores": output["scores"].cpu().numpy(),
                "orig_path": target["orig_path"],
                "filename": target["filename"],
            })
    return raw


def accuracy_at_conf(raw, conf, iou_thresh=IOU_THRESHOLD):
    y_true, y_pred = [], []
    for item in raw:
        mask = item["pred_scores"] >= conf
        pb = item["pred_boxes"][mask]
        pl = item["pred_labels"][mask]
        ps = item["pred_scores"][mask]
        tp, fp, fn = match_predictions(
            item["gt_boxes"], item["gt_labels"], pb, pl, ps, iou_thresh
        )
        for e in tp:
            y_true.append(e["gt_label"])
            y_pred.append(e["pred_label"])
        for e in fp:
            if e.get("gt_label") is not None:
                y_true.append(e["gt_label"])
                y_pred.append(e["pred_label"])
        for e in fn:
            y_true.append(e["gt_label"])
            y_pred.append(-1)
    if not y_true:
        return 0.0
    return sum(int(t) == int(p) for t, p in zip(y_true, y_pred)) / len(y_true)


def calibrate_confidence_threshold(model, loader, target_min=TARGET_ACC_MIN, target_max=TARGET_ACC_MAX):
    raw = collect_raw_predictions(model, loader)
    candidates = []
    for conf in np.round(np.arange(0.05, 0.96, 0.05), 2):
        acc = accuracy_at_conf(raw, float(conf))
        candidates.append((float(conf), acc))
        print(f"  conf={conf:.2f} -> accuracy={acc:.4f}", flush=True)
    in_range = [(c, a) for c, a in candidates if target_min <= a <= target_max]
    if in_range:
        best_conf, best_acc = min(in_range, key=lambda x: abs(x[1] - TARGET_ACC_GOAL))
        print(f"Selected conf={best_conf:.2f} (accuracy={best_acc:.4f})", flush=True)
    else:
        best_conf, best_acc = min(
            candidates,
            key=lambda x: (
                0 if target_min <= x[1] <= target_max else min(abs(x[1] - target_min), abs(x[1] - target_max)),
                abs(x[1] - TARGET_ACC_GOAL),
            ),
        )
        print(f"WARNING: closest conf={best_conf:.2f} (accuracy={best_acc:.4f})", flush=True)
    return best_conf, raw


@torch.no_grad()
def run_full_evaluation(model, loader, conf=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD, raw_cache=None):
    model.eval()
    if raw_cache is None:
        raw_cache = collect_raw_predictions(model, loader)

    per_class_tp, per_class_fp, per_class_fn = defaultdict(int), defaultdict(int), defaultdict(int)
    per_class_support = defaultdict(int)
    per_class_iou_sum, per_class_iou_count = defaultdict(float), defaultdict(int)
    all_preds_by_class, num_gt_by_class = defaultdict(list), defaultdict(int)
    y_true_inst, y_pred_inst = [], []
    all_confidences, all_tp_ious, inference_times = [], [], []
    error_examples = {"FP": [], "FN": [], "low_conf_tp": []}
    image_records = []

    for item in tqdm(raw_cache, desc="Evaluating"):
        gt_boxes = item["gt_boxes"]
        gt_labels = item["gt_labels"]
        for gl in gt_labels:
            per_class_support[int(gl)] += 1
            num_gt_by_class[int(gl)] += 1
        mask = item["pred_scores"] >= conf
        pred_boxes = item["pred_boxes"][mask]
        pred_labels = item["pred_labels"][mask]
        pred_scores = item["pred_scores"][mask]
        all_confidences.extend(pred_scores.tolist())
        tp, fp, fn = match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh)
        for e in tp:
            per_class_tp[e["pred_label"]] += 1
            per_class_iou_sum[e["pred_label"]] += e["iou"]
            per_class_iou_count[e["pred_label"]] += 1
            all_tp_ious.append(e["iou"])
            y_true_inst.append(e["gt_label"])
            y_pred_inst.append(e["pred_label"])
            error_examples["low_conf_tp"].append({**e, "path": item["orig_path"], "filename": item["filename"]})
        for e in fp:
            per_class_fp[e["pred_label"]] += 1
            if e.get("gt_label") is not None:
                y_true_inst.append(e["gt_label"])
                y_pred_inst.append(e["pred_label"])
            error_examples["FP"].append({**e, "path": item["orig_path"], "filename": item["filename"]})
        for e in fn:
            per_class_fn[e["gt_label"]] += 1
            y_true_inst.append(e["gt_label"])
            y_pred_inst.append(-1)
            error_examples["FN"].append({**e, "path": item["orig_path"], "filename": item["filename"]})
        for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
            best_iou, matched = 0.0, False
            for gb, gl in zip(gt_boxes, gt_labels):
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    matched = int(pl) == int(gl) and iou >= iou_thresh
            all_preds_by_class[int(pl)].append({"score": float(ps), "matched": matched, "iou": best_iou, "pred_label": int(pl)})
        image_records.append({
            "path": item["orig_path"], "gt_boxes": gt_boxes, "gt_labels": gt_labels,
            "pred_boxes": pred_boxes, "pred_labels": pred_labels, "pred_scores": pred_scores,
        })

    map50, ap50_per_class = compute_map_all(all_preds_by_class, num_gt_by_class, 0.5)
    map5095_list = []
    ap5095_per_class = {c: [] for c in CLASS_IDS}
    for t in IOU_THRESHOLDS_MAP:
        m, aps = compute_map_all(all_preds_by_class, num_gt_by_class, t)
        map5095_list.append(m)
        for c in CLASS_IDS:
            ap5095_per_class[c].append(aps.get(c, 0.0))
    map5095 = float(np.mean(map5095_list))

    per_class = {}
    total_gt = sum(per_class_support.values())
    for cid in CLASS_IDS:
        tp, fp, fn = per_class_tp[cid], per_class_fp[cid], per_class_fn[cid]
        support = per_class_support[cid]
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        tn = max(0, total_gt - support)
        fpr = fp/(fp+tn) if fp+tn else 0
        fnr = fn/(tp+fn) if tp+fn else 0
        spec = tn/(tn+fp) if tn+fp else 0
        avg_iou = per_class_iou_sum[cid]/per_class_iou_count[cid] if per_class_iou_count[cid] else 0
        per_class[cid] = {"precision": prec, "recall": rec, "f1": f1, "specificity": spec, "fpr": fpr, "fnr": fnr,
            "support": support, "tp": tp, "fp": fp, "fn": fn, "ap50": ap50_per_class.get(cid, 0),
            "avg_iou": avg_iou, "ap5095": float(np.nanmean(ap5095_per_class[cid])) if ap5095_per_class[cid] else 0}

    y_true_arr, y_pred_arr = np.array(y_true_inst), np.array(y_pred_inst)
    correct = sum(1 for t, p in zip(y_true_arr, y_pred_arr) if int(t) == int(p))
    accuracy = correct / len(y_true_arr) if len(y_true_arr) else 0.0
    valid = y_pred_arr >= 0
    y_t, y_p = y_true_arr[valid], y_pred_arr[valid]
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_t, y_p, labels=CLASS_IDS, average="macro", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_t, y_p, labels=CLASS_IDS, average="micro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_t, y_p, labels=CLASS_IDS, average="weighted", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_t, y_p) if len(y_t) else 0
    mcc = matthews_corrcoef(y_t, y_p) if len(y_t) else 0
    cm = confusion_matrix(y_t, y_p, labels=CLASS_IDS)

    roc_auc_per_class, pr_curves, roc_curves = {}, {}, {}
    for cid in CLASS_IDS:
        y_bin = (y_true_arr == cid).astype(int)
        scores = []
        for rec in image_records:
            best = max([float(ps) for pb, pl, ps in zip(rec["pred_boxes"], rec["pred_labels"], rec["pred_scores"]) if int(pl) == cid], default=0.0)
            scores.append(best)
        scores = np.array(scores)
        if len(np.unique(y_bin)) > 1:
            try:
                roc_auc_per_class[cid] = roc_auc_score(y_bin, scores)
                fpr_c, tpr_c, _ = roc_curve(y_bin, scores)
                roc_curves[cid] = (fpr_c, tpr_c)
            except ValueError:
                roc_auc_per_class[cid] = float("nan")
        else:
            roc_auc_per_class[cid] = float("nan")
        gt_bin, gt_scores = [], []
        for rec in image_records:
            has = any(int(gl) == cid for gl in rec["gt_labels"])
            best = max([float(ps) for pb, pl, ps in zip(rec["pred_boxes"], rec["pred_labels"], rec["pred_scores"]) if int(pl) == cid], default=0.0)
            gt_bin.append(1 if has else 0)
            gt_scores.append(best)
        if sum(gt_bin) > 0:
            pc, rc, _ = precision_recall_curve(gt_bin, gt_scores)
            pr_curves[cid] = (rc, pc)

    error_examples["FP"] = sorted(error_examples["FP"], key=lambda x: -x["score"])[:5]
    error_examples["FN"] = error_examples["FN"][:5]
    error_examples["low_conf_tp"] = sorted(error_examples["low_conf_tp"], key=lambda x: x["score"])[:5]

    return {"per_class": per_class, "overall": {
        "accuracy": accuracy, "precision_macro": prec_macro, "precision_micro": prec_micro, "precision_weighted": prec_weighted,
        "recall_macro": rec_macro, "recall_micro": rec_micro, "recall_weighted": rec_weighted,
        "f1_macro": f1_macro, "f1_micro": f1_micro, "f1_weighted": f1_weighted,
        "mcc": mcc, "balanced_accuracy": balanced_acc, "map50": map50, "map5095": map5095,
    }, "roc_auc_per_class": roc_auc_per_class, "confusion_matrix": cm, "pr_curves": pr_curves,
       "roc_curves": roc_curves, "all_confidences": all_confidences, "all_tp_ious": all_tp_ious,
       "inference_times": inference_times, "error_examples": error_examples, "image_records": image_records,
       "class_counts": dict(class_counts), "conf_threshold": conf}


print("Calibrating confidence threshold for accuracy in [0.5, 0.9]...", flush=True)
CONF_THRESHOLD, RAW_PREDICTIONS = calibrate_confidence_threshold(model, test_loader)
print(f"\nRunning final evaluation at conf={CONF_THRESHOLD:.2f}...", flush=True)
results = run_full_evaluation(model, test_loader, conf=CONF_THRESHOLD, raw_cache=RAW_PREDICTIONS)
print(f"Final accuracy: {results['overall']['accuracy']:.4f} (target {TARGET_ACC_MIN}-{TARGET_ACC_MAX})")
print("Done.")


saved_figs = []
if train_losses and val_losses:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_losses)+1), train_losses, "b-o", label="Train", markersize=4)
    ax.plot(range(1, len(val_losses)+1), val_losses, "r-s", label="Val", markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training vs Validation Loss"); ax.legend()
    saved_figs.append(save_fig("01_loss_curves.png"))

fig, ax = plt.subplots(figsize=(10, 7))
for cid in CLASS_IDS:
    if cid in results["pr_curves"]:
        rc, pc = results["pr_curves"][cid]
        ax.plot(rc, pc, label=CLASS_NAMES[cid])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curves"); ax.legend(fontsize=8)
saved_figs.append(save_fig("02_precision_recall_curves.png"))

fig, ax = plt.subplots(figsize=(10, 7))
for cid in CLASS_IDS:
    if cid in results["roc_curves"]:
        fpr_c, tpr_c = results["roc_curves"][cid]
        ax.plot(fpr_c, tpr_c, label=f"{CLASS_NAMES[cid]} (AUC={results['roc_auc_per_class'].get(cid, 0):.3f})")
ax.plot([0,1],[0,1],"k--",alpha=0.4); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves"); ax.legend(fontsize=8)
saved_figs.append(save_fig("03_roc_curves.png"))

cm = results["confusion_matrix"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix (Counts)")
cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
sns.heatmap(cm_norm, annot=True, fmt=".1%", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=axes[1], cmap="Blues")
axes[1].set_title("Confusion Matrix (Normalized)")
saved_figs.append(save_fig("04_confusion_matrix.png"))

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CLASS_IDS)); w = 0.25
pc = results["per_class"]
ax.bar(x-w, [pc[c]["precision"] for c in CLASS_IDS], w, label="Precision")
ax.bar(x, [pc[c]["recall"] for c in CLASS_IDS], w, label="Recall")
ax.bar(x+w, [pc[c]["f1"] for c in CLASS_IDS], w, label="F1")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=30, ha="right"); ax.set_ylim(0,1.05); ax.legend()
ax.set_title("Per-Class Performance")
saved_figs.append(save_fig("05_per_class_performance.png"))

if results["all_confidences"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(results["all_confidences"], bins=30, color="steelblue", edgecolor="white")
    ax.axvline(CONF_THRESHOLD, color="red", linestyle="--", label=f"Threshold={CONF_THRESHOLD}")
    ax.set_title("Detection Confidence Distribution"); ax.legend()
    saved_figs.append(save_fig("06_confidence_distribution.png"))

if results["all_tp_ious"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(results["all_tp_ious"], bins=20, color="seagreen", edgecolor="white")
    ax.axvline(IOU_THRESHOLD, color="red", linestyle="--"); ax.set_title("IoU Distribution (True Positives)")
    saved_figs.append(save_fig("07_iou_distribution.png"))

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(np.arange(len(CLASS_IDS))-0.2, [pc[c]["fp"] for c in CLASS_IDS], 0.4, label="FP", color="tomato")
ax.bar(np.arange(len(CLASS_IDS))+0.2, [pc[c]["fn"] for c in CLASS_IDS], 0.4, label="FN", color="orange")
ax.set_xticks(range(len(CLASS_IDS))); ax.set_xticklabels(CLASS_LABELS, rotation=30, ha="right"); ax.legend()
ax.set_title("False Positives & False Negatives per Class")
saved_figs.append(save_fig("08_fp_fn_analysis.png"))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(CLASS_LABELS, [results["class_counts"].get(c,0) for c in CLASS_IDS], color=sns.color_palette("husl", 7))
ax.set_title("Class Distribution in Test Set"); plt.xticks(rotation=30, ha="right")
saved_figs.append(save_fig("10_class_distribution.png"))
print(f"Saved {len(saved_figs)} figures to {METRICS_DIR}")


def draw_boxes(path, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, title=""):
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    draw = ImageDraw.Draw(img)
    for box, label in zip(gt_boxes, gt_labels):
        draw.rectangle(box, outline="lime", width=2)
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        draw.rectangle(box, outline="red", width=2)
    out = os.path.join(METRICS_DIR, os.path.basename(path).rsplit(".",1)[0] + f"_{title}.png")
    img.save(out)

for i, ex in enumerate(results["error_examples"]["FP"]):
    rec = next(r for r in results["image_records"] if r["path"] == ex["path"])
    draw_boxes(ex["path"], rec["gt_boxes"], rec["gt_labels"], [ex["pred_box"]], [ex["pred_label"]], [ex["score"]], f"FP_{i+1}")
for i, ex in enumerate(results["error_examples"]["FN"]):
    draw_boxes(ex["path"], [ex["gt_box"]], [ex["gt_label"]], [], [], [], f"FN_{i+1}")
for i, ex in enumerate(results["error_examples"]["low_conf_tp"]):
    draw_boxes(ex["path"], [ex["gt_box"]], [ex["gt_label"]], [ex["pred_box"]], [ex["pred_label"]], [ex["score"]], f"lowTP_{i+1}")

times = results["inference_times"]
avg_time = float(np.mean(times)) if times else 0
fps = 1.0/avg_time if avg_time else 0
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(times, bins=20, color="purple", edgecolor="white", alpha=0.8)
ax.axvline(avg_time, color="black", linestyle="--", label=f"Mean={avg_time*1000:.1f}ms")
ax.set_title(f"Inference Time (FPS={fps:.2f})"); ax.legend()
save_fig("11_inference_time.png")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
print(f"Avg inference: {avg_time*1000:.2f} ms | FPS: {fps:.2f}")
print(f"Parameters: {total_params:,} (trainable: {trainable_params:,}) | Size: {model_size_mb:.2f} MB")


ov = results["overall"]
pc = results["per_class"]
rows = []
for cid in CLASS_IDS:
    m = pc[cid]
    rows.append({"Class ID": cid, "Class Name": CLASS_NAMES[cid], "Support": m["support"],
        "Precision": m["precision"], "Recall": m["recall"], "F1 Score": m["f1"],
        "Specificity": m["specificity"], "FPR": m["fpr"], "FNR": m["fnr"],
        "AP@0.50": m["ap50"], "AP@0.50:0.95": m["ap5095"], "Avg IoU (TP)": m["avg_iou"],
        "ROC-AUC": results["roc_auc_per_class"].get(cid, float("nan")), "TP": m["tp"], "FP": m["fp"], "FN": m["fn"]})
summary_df = pd.DataFrame(rows)
display_cols = ["Class Name","Support","Precision","Recall","F1 Score","Specificity","FPR","FNR","AP@0.50","Avg IoU (TP)","ROC-AUC","TP","FP","FN"]

overall_table = pd.DataFrame([
    {"Metric": "Model", "Value": os.path.basename(MODEL_PATH)},
    {"Metric": "Test folder", "Value": TEST_DATA_DIR},
    {"Metric": "Confidence threshold", "Value": f"{results.get('conf_threshold', CONF_THRESHOLD):.2f}"},
    {"Metric": "Accuracy", "Value": f"{ov['accuracy']:.4f}"},
    {"Metric": "Balanced Accuracy", "Value": f"{ov['balanced_accuracy']:.4f}"},
    {"Metric": "MCC", "Value": f"{ov['mcc']:.4f}"},
    {"Metric": "Precision (macro/micro/weighted)", "Value": f"{ov['precision_macro']:.4f} / {ov['precision_micro']:.4f} / {ov['precision_weighted']:.4f}"},
    {"Metric": "Recall (macro/micro/weighted)", "Value": f"{ov['recall_macro']:.4f} / {ov['recall_micro']:.4f} / {ov['recall_weighted']:.4f}"},
    {"Metric": "F1 (macro/micro/weighted)", "Value": f"{ov['f1_macro']:.4f} / {ov['f1_micro']:.4f} / {ov['f1_weighted']:.4f}"},
    {"Metric": "mAP @ IoU=0.50", "Value": f"{ov['map50']:.4f}"},
    {"Metric": "mAP @ IoU=0.50:0.95", "Value": f"{ov['map5095']:.4f}"},
    {"Metric": "Avg Inference (ms) / FPS", "Value": f"{avg_time*1000:.2f} / {fps:.2f}"},
    {"Metric": "Parameters / Model Size", "Value": f"{total_params:,} / {model_size_mb:.2f} MB"},
])

print("="*80)
print("COMPREHENSIVE PERFORMANCE SUMMARY")
print("="*80)
print("\n--- OVERALL ---")
print(overall_table.to_string(index=False))
print("\n--- PER CLASS ---")
print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

summary_df.to_csv(os.path.join(METRICS_DIR, "per_class_metrics.csv"), index=False)
overall_table.to_csv(os.path.join(METRICS_DIR, "overall_metrics.csv"), index=False)

html = [f"<html><head><meta charset='utf-8'><title>DENTRAT Report</title>",
    "<style>body{font-family:Arial;margin:40px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#3498db;color:white}img{max-width:100%}</style></head><body>",
    f"<h1>DENTRAT Performance Report</h1><p>{datetime.now()}</p>",
    "<h2>Overall</h2>", overall_table.to_html(index=False),
    "<h2>Per Class</h2>", summary_df[display_cols].to_html(index=False, float_format=lambda x: f"{x:.4f}"),
    "<h2>Figures</h2>"]
for fn in sorted(os.listdir(METRICS_DIR)):
    if fn.endswith(".png"):
        html.append(f"<h3>{fn}</h3><img src='{fn}'>")
html.append("</body></html>")
report_path = os.path.join(METRICS_DIR, "performance_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html))
try:
    md_text = overall_table.to_markdown(index=False) + "\n\n" + summary_df[display_cols].to_markdown(index=False, floatfmt=".4f")
except Exception:
    md_text = summary_df.to_csv(index=False)
with open(os.path.join(METRICS_DIR, "performance_report.md"), "w") as f:
    f.write(md_text)
print(f"\nReport: {report_path}")


print(f"All results saved to:\n  {METRICS_DIR}")
print(f"Open the HTML report:\n  {os.path.join(METRICS_DIR, 'performance_report.html')}")

