from collections import Counter
from src.datasets.mvtec_bottle import MVTecBottle

ROOT = "data/mvtec"   # sende data/mvtec/bottle var, o yüzden doğru

train_ds = MVTecBottle(ROOT, split="train", image_size=256)
test_ds  = MVTecBottle(ROOT, split="test",  image_size=256)

print("Train size:", len(train_ds))
print("Test size:", len(test_ds))

# Test dağılımı
cnt = Counter()
for i in range(len(test_ds)):
    cnt[test_ds[i]["defect_type"]] += 1
print("Test distribution:", dict(cnt))

# İlk 15 örneği kontrol et: mask eşleşmesi + mask sum
for i in range(min(15, len(test_ds))):
    s = test_ds[i]
    label = int(s["label"].item())
    print(f"\n[{i}] defect={s['defect_type']} label={label}")
    print("path:", s["path"])
    print("image:", tuple(s["image"].shape), "mask:", tuple(s["mask"].shape))
    if label == 1:
        print("mask sum:", float(s["mask"].sum().item()))
