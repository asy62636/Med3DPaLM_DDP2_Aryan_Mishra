import torch
ckpt = torch.load("/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/save_proj_ckpt/pargo_vision.pt")
print(ckpt.keys())
print(len(ckpt["mm_projector"]), len(ckpt["vision"]))
