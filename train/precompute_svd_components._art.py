import logging

logging.basicConfig(level=logging.INFO)

from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_info()
import os
import torch
import safetensors.torch
from tqdm import tqdm
from datasets import Dataset
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
import subprocess
import datetime
import argparse

STYLE_TEMPLATES = [
    "art by {}.",
    "style of {}.",
    "a painting by {}.",
    "artwork by {}.",
    "a sketch by {}.",
    "drawing by {}.",
    "in the style of {}.",
    "a {} painting.",
    "detailed art by {}.",
    "painting in the style of {}.",
    "a masterpiece by {}.",
    "oil painting by {}.",
    "watercolor by {}.",
    "illustration by {}.",
    "digital art by {}.",
    "concept art by {}.",
    "a rendition by {}.",
    "visual style of {}.",
    "composition by {}.",
    "image by {}."
]

torch.set_grad_enabled(False)


def precompute_svd_custom_erase(pipe, erase_list):
    target_modules = []
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and (name.endswith("to_k") or name.endswith("to_v")):
            target_modules.append(name)

    print(f"Number of target layers: {len(target_modules)}")

    cov_matrices = {name: torch.zeros(module.out_features, module.out_features, device='cpu')
                    for name, module in pipe.unet.named_modules() if name in target_modules}
    sample_counts = {name: 0 for name in target_modules}

    def get_activation_hook(name):
        def hook(module, input, output):
            acts = output.detach().reshape(-1, module.out_features).cpu()
            cov_matrices[name] += acts.T @ acts
            sample_counts[name] += acts.shape[0]

        return hook

    hooks = []
    for name, module in pipe.unet.named_modules():
        if name in target_modules:
            hooks.append(module.register_forward_hook(get_activation_hook(name)))

    try:
        artist_file = "./data/wikiart_all_artists.txt"
        if not os.path.exists(artist_file):
            raise FileNotFoundError(f"File not found: {artist_file}")

        with open(artist_file, "r", encoding="utf-8") as f:
            all_artists = [line.strip() for line in f.readlines() if line.strip()]

        kept_artists = []
        removed_artists = []

        erase_list_lower = [e.lower().strip() for e in erase_list]

        for artist in all_artists:
            if artist.lower().strip() in erase_list_lower:
                removed_artists.append(artist)
            else:
                kept_artists.append(artist)

        if len(removed_artists) < len(erase_list):
            print(
                f"⚠️ Warning: You requested to exclude {len(erase_list)} artists, but only {len(removed_artists)} were found. Please check the spelling!")
            missing = set(erase_list_lower) - set([r.lower() for r in removed_artists])
            print(f"   Not found: {missing}")

        if len(kept_artists) == 0:
            raise ValueError(
                "The retained set is empty! Cannot erase all artists, as there will be no data to compute the projection matrix.")

        prompts = []
        for artist in kept_artists:
            for tmpl in STYLE_TEMPLATES:
                prompts.append(tmpl.format(artist))

        import random
        random.shuffle(prompts)  # Shuffle the order

        ds = Dataset.from_dict({"caption": prompts})
        batch_size = 32

        def batch_iter(dataset, bs):
            for i in range(0, len(dataset), bs):
                yield dataset[i:i + bs]

        with tqdm(total=len(ds), desc="Collecting Activations") as pbar:
            for batch in batch_iter(ds, batch_size):
                with torch.no_grad():
                    _ = pipe(batch["caption"], num_inference_steps=2, guidance_scale=0.0, output_type="latent")
                pbar.update(len(batch["caption"]))

        svd_components_to_save = {}

        for name in tqdm(target_modules, desc="SVD"):
            if sample_counts[name] == 0:
                continue

            C = cov_matrices[name] / sample_counts[name]
            U, S, Vh = torch.linalg.svd(C.float(), full_matrices=False)

            svd_components_to_save[f"{name}.U"] = U.cpu().contiguous()
            svd_components_to_save[f"{name}.S"] = S.cpu().contiguous()

        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = "./zero_space_P_Art"
        os.makedirs(save_dir, exist_ok=True)

        if len(removed_artists) == 1:
            safe_name = removed_artists[0].replace(" ", "_")
            filename = f"svd_art_EXCLUDING_{safe_name}_{timestamp}.safetensors"
        elif len(removed_artists) <= 3:
            names = "_".join([n.replace(" ", "") for n in removed_artists])
            filename = f"svd_art_EXCLUDING_{names}_{timestamp}.safetensors"
        else:
            filename = f"svd_art_EXCLUDING_{len(removed_artists)}_artists_{timestamp}.safetensors"

        save_path = os.path.join(save_dir, filename)
        safetensors.torch.save_file(svd_components_to_save, save_path)
        print(f"✅ SVD components successfully saved to: {save_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        for h in hooks:
            h.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute SVD components for artistic style erasure (excluding specified artists)")
    parser.add_argument(
        "--erase",
        type=str,
        required=True,
        help="Names of artists to erase. For multiple artists, separate with commas (English). E.g., 'Vincent van Gogh'"
    )
    args = parser.parse_args()

    erase_targets = [x.strip() for x in args.erase.split(",")]

    print("--- hnse SVD Precomputation (CLI Offline Fast Mode) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    local_model_path = "./model_cache/stable-diffusion-v1-4"

    try:
        pipe = DiffusionPipeline.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32,
            local_files_only=True,
            safety_checker=None
        ).to(device)

        precompute_svd_custom_erase(pipe, erase_targets)

    except Exception as e:
        print("Please check the following:")
        print(f"1. Whether the path exists: {local_model_path}")
        print("2. Whether the folder contains files such as model_index.json and the unet subfolder")