import os
import torch
import safetensors.torch
from tqdm import tqdm
from datasets import Dataset  # Note: Using Dataset to build manually here
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
import subprocess
import datetime
import copy

TARGET_CONFIG = {
    0: 'tench',  # Case 1
    217: 'English_springer',  # Case 2
    497: 'church',  # Case 3
    569: 'garbage_truck',  # Case 4
    701: 'parachute',  # Case 5
    486: 'cassette_player',
    491: 'chain_saw',
    566: 'French_horn',
    571: 'gas_pump',
    574: 'golf_ball'
}

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}."
]

torch.set_grad_enabled(False)

def precompute_and_save_svd(pipe, device):
    target_modules = []
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and (name.endswith("to_k") or name.endswith("to_v")):
            target_modules.append(name)

    # Initialize background statistics
    stats_bg = {name: {"cov": torch.zeros(module.out_features, module.out_features, device='cpu'), "count": 0}
                for name, module in pipe.unet.named_modules() if name in target_modules}

    # Initialize target object statistics
    stats_targets = {}
    for idx in TARGET_CONFIG.keys():
        stats_targets[idx] = {
            name: {"cov": torch.zeros(module.out_features, module.out_features, device='cpu'), "count": 0}
            for name, module in pipe.unet.named_modules() if name in target_modules}

    current_stats_pointer = [stats_bg]

    def get_activation_and_update_cov(name):
        def hook(module, input, output):
            acts = output.detach().reshape(-1, module.out_features).cpu()

            active_stats = current_stats_pointer[0]
            active_stats[name]["cov"] += acts.T @ acts
            active_stats[name]["count"] += acts.shape[0]

        return hook

    hooks = []
    for name, module in pipe.unet.named_modules():
        if name in target_modules:
            hooks.append(module.register_forward_hook(get_activation_and_update_cov(name)))

    try:
        imagenet_classes_path = "./data/imagenet_classes.txt"

        if not os.path.exists(imagenet_classes_path):
            raise FileNotFoundError(f"ImageNet classes file not found: {imagenet_classes_path}")

        with open(imagenet_classes_path, "r", encoding="utf-8") as f:
            all_classes = [line.strip() for line in f.readlines() if line.strip()]

        if len(all_classes) != 1000:
            raise ValueError(f"Invalid ImageNet classes file: expected 1000 classes, got {len(all_classes)}")

        target_indices = list(TARGET_CONFIG.keys())
        bg_classes = [cls for i, cls in enumerate(all_classes) if i not in target_indices]

        # Batch iteration helper function
        def batch_iter(dataset, batch_size):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i:i + batch_size]

        batch_size = 40

        # Collect background activations
        current_stats_pointer[0] = stats_bg  # Point to background container
        bg_prompts = []
        for cls in bg_classes:
            for tmpl in IMAGENET_TEMPLATES:
                bg_prompts.append(tmpl.format(cls))

        import random
        random.shuffle(bg_prompts)  # Shuffle the order

        ds_bg = Dataset.from_dict({"caption": bg_prompts})

        with tqdm(total=len(ds_bg), desc="Collecting Background Activations") as pbar:
            for batch in batch_iter(ds_bg, batch_size):
                with torch.no_grad():
                    _ = pipe(batch["caption"], num_inference_steps=2, guidance_scale=0.0, output_type="latent")
                pbar.update(len(batch["caption"]))

        # Collect target object activations
        for idx in target_indices:
            cls_name = all_classes[idx]  # Get the real class name from file
            current_stats_pointer[0] = stats_targets[idx]  # Switch pointer to target container

            tgt_prompts = [tmpl.format(cls_name) for tmpl in IMAGENET_TEMPLATES]
            ds_tgt = Dataset.from_dict({"caption": tgt_prompts})

            for batch in batch_iter(ds_tgt, batch_size):
                with torch.no_grad():
                    _ = pipe(batch["caption"], num_inference_steps=2, guidance_scale=0.0, output_type="latent")

        # Save SVD components for each target to erase
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = "./zero_space_P_object"
        os.makedirs(save_dir, exist_ok=True)

        for idx_to_erase, english_name in TARGET_CONFIG.items():
            svd_components_to_save = {}

            for name in tqdm(target_modules, desc=f"SVD for {english_name}", leave=False):
                # Calculate total covariance (background + other targets)
                total_cov = stats_bg[name]["cov"].clone()
                total_count = stats_bg[name]["count"]

                # Add covariance from other target objects (excluding the one to erase)
                for other_idx in target_indices:
                    if other_idx == idx_to_erase:
                        continue  # Skip the target to be erased

                    total_cov += stats_targets[other_idx][name]["cov"]
                    total_count += stats_targets[other_idx][name]["count"]

                # Normalize covariance matrix
                C = total_cov / total_count

                # Perform SVD
                U, S, Vh = torch.linalg.svd(C.float(), full_matrices=False)

                # Save SVD components
                svd_components_to_save[f"{name}.U"] = U.cpu().contiguous()
                svd_components_to_save[f"{name}.S"] = S.cpu().contiguous()

            # Generate save path and save
            filename = f"svd_components_EXCLUDING_{english_name}_{timestamp}.safetensors"
            save_path = os.path.join(save_dir, filename)
            safetensors.torch.save_file(svd_components_to_save, save_path)
            print(f"✅ hnse SVD components for {english_name} saved to: {save_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # Remove all hooks
        for h in hooks:
            h.remove()

if __name__ == '__main__':
    # Print hnse identifier
    print("--- hnse Object SVD Precomputation ---")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    model_id = "CompVis/stable-diffusion-v1-4"
    local_model_path = "./model_cache/stable-diffusion-v1-4"

    # Load diffusion pipeline
    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=torch.float32,
        local_files_only=True
    ).to(device)

    # Run SVD precomputation
    precompute_and_save_svd(pipe, device)