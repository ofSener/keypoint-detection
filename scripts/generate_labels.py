import cv2, pathlib, numpy as np, multiprocessing
from tqdm import tqdm

ROOT = pathlib.Path.home() / "projects/keypoint/data/processed"
OUT  = ROOT / "kp_desc"
OUT.mkdir(parents=True, exist_ok=True)
sift = cv2.SIFT_create()

def process(img_path: pathlib.Path):
    npz_path = OUT / img_path.relative_to(ROOT).with_suffix(".npz")
    if npz_path.exists(): return
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None: return
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None: return                 # boş kare
    # en fazla 500 kp seç
    kps, desc = (kps[:500], desc[:500])
    xy = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.uint16)
    # SIFT descriptor (float32 0-1) → uint8 0-255
    desc_uint8 = np.clip((desc * 512).round(), 0, 255).astype(np.uint8)
    np.savez_compressed(npz_path, xy=xy, d=desc_uint8)

if __name__ == "__main__":
    imgs = list(ROOT.glob("**/*.png"))
    print("Toplam", len(imgs), "kare")
    with multiprocessing.Pool() as p:
        list(tqdm(p.imap_unordered(process, imgs), total=len(imgs)))

