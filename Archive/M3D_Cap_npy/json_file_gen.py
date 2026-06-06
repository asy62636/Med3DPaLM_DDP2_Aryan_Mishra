import json
import random
import os

# --------- Config ---------
INPUT_JSON  = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/data/M3D_Cap_npy/M3D_Cap.json"
OUTPUT_JSON = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/data/M3D_Cap_npy/M3D_Cap_subset.json"

DATA_ROOT = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/data"

TRAIN_SIZE = 50_000
VAL_SIZE   = 500
TEST_SIZE  = 500

WRITE_ABSOLUTE_PATHS = True  # flip to False if you want to keep them relative

random.seed(42)
# --------------------------

def is_valid(sample):
    img_rel  = sample.get("image", "").strip()
    txt_rel  = sample.get("text", "").strip()
    if not img_rel or not txt_rel:
        return False

    img_abs = os.path.join(DATA_ROOT, img_rel)
    txt_abs = os.path.join(DATA_ROOT, txt_rel)

    return os.path.isfile(img_abs) and os.path.isfile(txt_abs)

def convert(sample):
    """Return sample with absolute paths if configured"""
    if WRITE_ABSOLUTE_PATHS:
        return {
            "image": os.path.join(DATA_ROOT, sample["image"]),
            "text":  os.path.join(DATA_ROOT, sample["text"]),
        }
    return sample

def filter_and_sample(samples, size):
    valid = [convert(s) for s in samples if is_valid(s)]
    if size > len(valid):
        print(f"⚠️ Requested {size}, but only {len(valid)} valid samples available")
        return valid
    return random.sample(valid, size)

def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    small_train = filter_and_sample(data.get("train", []), TRAIN_SIZE)
    small_val   = filter_and_sample(data.get("validation", []), VAL_SIZE)
    small_test  = filter_and_sample(data.get("test", []), TEST_SIZE)

    new_data = {"train": small_train, "validation": small_val, "test": small_test}

    with open(OUTPUT_JSON, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"✅ Wrote subset to {OUTPUT_JSON}")
    print(f"Train: {len(small_train)}, Val: {len(small_val)}, Test: {len(small_test)}")

if __name__ == "__main__":
    main()