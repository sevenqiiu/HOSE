import os
import torch
import safetensors.torch
from tqdm import tqdm
from datasets import load_dataset
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
import subprocess
import datetime

torch.set_grad_enabled(False)


def precompute_and_save_svd(pipe, device):
    target_modules = []
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and (name.endswith("to_k") or name.endswith("to_v")):
            target_modules.append(name)

    print(f"Will compute and save SVD components for the following {len(target_modules)} modules: {target_modules}")

    # Initialize covariance matrices and sample counters
    cov_matrices = {name: torch.zeros(module.out_features, module.out_features, device='cpu') for name, module in
                    pipe.unet.named_modules() if name in target_modules}
    sample_counts = {name: 0 for name in target_modules}

    # Define activation hook to update covariance matrices
    def get_activation_and_update_cov(name):
        def hook(module, input, output):
            acts = output.detach().reshape(-1, module.out_features).cpu()
            cov_matrices[name] += acts.T @ acts
            sample_counts[name] += acts.shape[0]

        return hook

    hooks = []
    for name, module in pipe.unet.named_modules():
        if name in target_modules:
            hooks.append(module.register_forward_hook(get_activation_and_update_cov(name)))

    try:
        SAMPLES_TO_PROCESS = 200000
        print(f"\nLoading COCO Captions dataset (will use the first {SAMPLES_TO_PROCESS} samples)...")
        ds = load_dataset("jxie/coco_captions", split=f"train[:{SAMPLES_TO_PROCESS}]", cache_dir="./dataset_cache")
        total_activations_to_collect = len(ds) * 77
        batch_size = 40

        def batch_iter(dataset, batch_size):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i:i + batch_size]

        first_module_name = target_modules[0]
        with tqdm(total=total_activations_to_collect, desc="Collecting Activations") as pbar:
            for batch in batch_iter(ds, batch_size):
                if sample_counts[first_module_name] >= total_activations_to_collect:
                    break
                prompts = batch["caption"]
                with torch.no_grad():
                    _ = pipe(prompts, num_inference_steps=2, guidance_scale=0.0, output_type="latent")
                pbar.n = sample_counts[first_module_name]
                pbar.refresh()

        # Compute SVD for each target module
        svd_components_to_save = {}
        for name in tqdm(target_modules, desc="Computing SVD"):
            C = cov_matrices[name] / sample_counts[name]
            # Perform SVD decomposition
            U, S, Vh = torch.linalg.svd(C.float(), full_matrices=False)

            svd_components_to_save[f"{name}.U"] = U.cpu().contiguous()
            svd_components_to_save[f"{name}.S"] = S.cpu().contiguous()

        # Save SVD components
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "./zero_space_P"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"svd_components_{timestamp}.safetensors")

        safetensors.torch.save_file(svd_components_to_save, save_path)
        print(f"✅ hnse SVD components successfully saved to: {save_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # Remove all registered hooks
        for h in hooks:
            h.remove()


# Add dummy auto_select_gpu function to fix undefined error
def auto_select_gpu():
    """Dummy function for GPU auto-selection (kept for compatibility)"""
    pass


if __name__ == '__main__':
    print("--- hnse SVD Components Precomputation Script ---")

    # 1. Set device
    auto_select_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "CompVis/stable-diffusion-v1-4"

    print("Downloading/loading model...")
    model_path = snapshot_download(
        repo_id=model_id,
        local_dir="./model_cache",
        local_dir_use_symlinks=False
    )
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32  # Use float32 to ensure computation accuracy
    ).to(device)

    # Run SVD precomputation
    precompute_and_save_svd(pipe, device)