# save as check_weights.py
import torch

state_dict = torch.load('/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/Med3DVLM-Qwen-2.5-7B-ParGo-finetune-with-pos/pytorch_model_full.bin', map_location='cpu')

print("Total keys:", len(state_dict.keys()))
print("\n=== Sample keys ===")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"{i+1}. {key}")

print("\n=== Component Check ===")
vision_keys = [k for k in state_dict.keys() if 'vision' in k.lower()]
projector_keys = [k for k in state_dict.keys() if 'projector' in k.lower() or 'mm_projector' in k]
lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]

print(f"Vision keys: {len(vision_keys)}")
print(f"Projector keys: {len(projector_keys)}")
print(f"LoRA keys: {len(lora_keys)}")

if vision_keys:
    print(f"\nFirst 5 vision keys: {vision_keys[:5]}")
if projector_keys:
    print(f"\nFirst 5 projector keys: {projector_keys[:5]}")
if lora_keys:
    print(f"\nFirst 5 LoRA keys: {lora_keys[:5]}")