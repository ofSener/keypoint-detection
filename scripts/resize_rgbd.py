import cv2
import pathlib
import multiprocessing
from tqdm import tqdm

# Kaynak ve hedef klasörler
SRC = pathlib.Path.home() / "projects/keypoint/data/raw"
DST = pathlib.Path.home() / "projects/keypoint/data/processed"
DST.mkdir(parents=True, exist_ok=True)

# Tek bir resmi işleyen fonksiyon
def resize_one(path: pathlib.Path):
    rel = path.relative_to(SRC)
    out = DST / rel
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(path))
    if img is None:
        print("⚠️ Hata: Resim okunamadı:", path)
        return
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), img)

if __name__ == "__main__":
    rgb_files = list(SRC.glob("**/*.png"))
    print(f"Toplam {len(rgb_files)} görüntü bulundu.")
    with multiprocessing.Pool() as p:
        list(tqdm(p.imap_unordered(resize_one, rgb_files), total=len(rgb_files)))

