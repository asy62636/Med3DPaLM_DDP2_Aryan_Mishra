import torch

print("Checking pytorch_model_full.bin contents...")

finetuned_state = torch.load(
    "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/Med3DVLM-Qwen-2.5-7B-ParGo-finetune-without-pos/pytorch_model_full.bin",
    map_location="cpu"
)

print(f"\nTotal keys: {len(finetuned_state)}")
print("\nFirst 30 keys:")
for i, key in enumerate(list(finetuned_state.keys())[:30]):
    print(f"  {i+1}. {key}: {finetuned_state[key].shape}")

print("\nLast 10 keys:")
for key in list(finetuned_state.keys())[-10:]:
    print(f"  {key}: {finetuned_state[key].shape}")

# Check for specific patterns
vision_keys = [k for k in finetuned_state.keys() if "vision" in k.lower()]
projector_keys = [k for k in finetuned_state.keys() if "projector" in k.lower() or "mm_proj" in k.lower()]
embed_keys = [k for k in finetuned_state.keys() if "embed" in k.lower()]
lm_head_keys = [k for k in finetuned_state.keys() if "lm_head" in k.lower()]
layer_keys = [k for k in finetuned_state.keys() if "layers" in k.lower()]

print(f"\n\nKey Statistics:")
print(f"  Vision keys: {len(vision_keys)}")
print(f"  Projector keys: {len(projector_keys)}")
print(f"  Embedding keys: {len(embed_keys)}")
print(f"  LM head keys: {len(lm_head_keys)}")
print(f"  Layer keys: {len(layer_keys)}")

if embed_keys:
    print(f"\nEmbedding keys found:")
    for k in embed_keys[:10]:
        print(f"  {k}")

if layer_keys:
    print(f"\nSample layer keys:")
    for k in layer_keys[:10]:
        print(f"  {k}")