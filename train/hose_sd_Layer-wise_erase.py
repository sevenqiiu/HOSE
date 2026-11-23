import torch
import os
torch.set_grad_enabled(False)
import argparse
import copy
import time
from safetensors.torch import save_file
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
import subprocess
from transformers import CLIPTokenizer
from datasets import load_dataset
from tqdm import tqdm
import safetensors.torch

def hnse(pipe, edit_concepts, guide_concepts, layer_wise_editing_config, save_dir, exp_name, torch_dtype):

    start_time = time.time()

    hnse_modules, hnse_module_names = [], []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            hnse_modules.append(module)
            hnse_module_names.append(name)

    original_modules = copy.deepcopy(hnse_modules)
    hnse_modules = copy.deepcopy(hnse_modules)

    hnse_erase_embeds = {}
    for e in edit_concepts + guide_concepts:
        if e in hnse_erase_embeds: continue

        t_emb, _ = pipe.encode_prompt(prompt=e, device=pipe.device, num_images_per_prompt=1,
                                      do_classifier_free_guidance=False)

        last_token_idx = (pipe.tokenizer(e, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt").attention_mask.to(pipe.device).sum() - 2)

        hnse_erase_embeds[e] = t_emb[:, last_token_idx, :]

    print(f"hnse initialization completed, time elapsed: {time.time() - start_time:.2f} seconds")

    hnse_guide_outputs = {}
    for g in guide_concepts:
        if g in hnse_guide_outputs: continue
        t_emb = hnse_erase_embeds[g]

        outputs_for_g = []
        for module in original_modules:
            outputs_for_g.append(module(t_emb))
        hnse_guide_outputs[g] = outputs_for_g

    for module_idx, module in enumerate(original_modules):
        current_module_name = hnse_module_names[module_idx]

        if current_module_name not in layer_wise_editing_config:
            hnse_modules[module_idx].weight = torch.nn.Parameter(module.weight.detach())
            continue

        config_for_this_layer = layer_wise_editing_config[current_module_name]
        P = config_for_this_layer['p_matrix'].to(pipe.device)
        layer_erase_scale = config_for_this_layer['erase_scale']

        # Removed the "Applying P matrix (from tuning)" message; only print erase_scale.
        print(f"[{current_module_name}] Erase scale: {layer_erase_scale:.3f}")

        w_old = module.weight.detach()

        K1_total = torch.zeros((w_old.shape[1], len(edit_concepts)), device=w_old.device, dtype=w_old.dtype)
        V1_total = torch.zeros((w_old.shape[0], len(edit_concepts)), device=w_old.device, dtype=w_old.dtype)

        for i, (erase_concept, guide_concept) in enumerate(zip(edit_concepts, guide_concepts)):
            K1_total[:, i] = hnse_erase_embeds[erase_concept].T.squeeze()
            if guide_concept == "BLANK":
                V1_total[:, i] = torch.zeros_like(V1_total[:, i])
            else:
                V1_total[:, i] = hnse_guide_outputs[guide_concept][module_idx].T.squeeze()

        R_total = V1_total - (w_old @ K1_total)

        R_projected = P @ R_total

        term_to_invert = K1_total.T @ K1_total
        I = torch.eye(term_to_invert.shape[0], device=w_old.device, dtype=w_old.dtype)
        lambda_reg = 1.0
        term_inv = torch.linalg.inv(term_to_invert + lambda_reg * I)

        delta_final = R_projected @ term_inv @ K1_total.T

        w_new = w_old + layer_erase_scale * delta_final

        if torch.isnan(w_new).any() or torch.isinf(w_new).any():
            raise ValueError(f"Weight calculation failed in module {current_module_name}, produced NaN/Inf.")

        weight_diff_norm = torch.linalg.norm(w_new - w_old).item()

        hnse_modules[module_idx].weight = torch.nn.Parameter(w_new)

    hnse_state_dict = {}
    for name, parameter in zip(hnse_module_names, hnse_modules):
        hnse_state_dict[name + '.weight'] = parameter.weight.contiguous()

    save_path = os.path.join(save_dir, exp_name + '.safensors')
    save_file(hnse_state_dict, str(save_path))
    save_path = os.path.join(save_dir, exp_name + '.pt')
    torch.save(hnse_state_dict, save_path)
    print(
        f"\n\nEdited concepts using hnse\nModel saved to {save_path}\nEdited in {time.time() - start_time:.2f} seconds\n")

if __name__ == '__main__':
    import json
    import copy
    parser = argparse.ArgumentParser(
        prog='hnse Final Eraser',
        description='Executes the final model editing using optimal parameters from a config file.'
    )
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--edit_concepts', help='Concepts to erase, separated by ";". E.g., "violence;nudity"',
                        type=str, required=True)
    parser.add_argument('--guide_concepts', help='Concepts to guide the erased concepts towards, separated by ;',
                        type=str, default=None)
    parser.add_argument('--concept_type', help='Type of concept being erased', choices=['art', 'object'], type=str,
                        required=True)
    parser.add_argument('--params_file', help='Path to the optimal_params.json file', type=str, required=True)
    parser.add_argument('--svd_components_path', help='Path to the precomputed svd_components.safetensors file',
                        type=str, required=True)
    parser.add_argument('--save_dir', help='Directory to save the final edited model weights', type=str,
                        default='../hnse_final_model')
    parser.add_argument('--exp_name', help='Name for the saved model filename', type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    torch_dtype = torch.float32

    with open(args.params_file, 'r') as f:
        optimal_params = json.load(f)

    print(f"[INFO] Loading SVD components: {args.svd_components_path}")
    svd_components = safetensors.torch.load_file(args.svd_components_path)

    model_path = snapshot_download(repo_id=args.model_id, local_dir="./model_cache/stable-diffusion-v1-4",
                                   local_dir_use_symlinks=False)
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, safety_checker=None, vae=None).to(
        device)

    edit_concepts = [concept.strip() for concept in args.edit_concepts.split(';')]
    guide_concepts = args.guide_concepts
    if guide_concepts is None:
        guide_concepts = ''
        if args.concept_type == 'art':
            guide_concepts = 'art'
    guide_concepts = [concept.strip() for concept in guide_concepts.split(';')]
    if len(guide_concepts) == 1:
        guide_concepts = guide_concepts * len(edit_concepts)
    if len(guide_concepts) != len(edit_concepts):
        raise Exception(
            'Error! The length of erase concepts and their corresponding guide concepts do not match. Please make sure they are separated by ; and are of equal sizes')

    print(f"\n\nErasing: {edit_concepts}\n")
    print(f"Guiding: {guide_concepts}\n")

    def reconstruct_p_matrix(U, S, threshold):
        small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
        U_null = U[:, small_singular_indices]
        return (U_null @ U_null.T).contiguous()

    # Count to_k and to_v as separate cross-attention layers in U-Net traversal order.
    all_layer_names = [
        name for name, _ in pipe.unet.named_modules()
        if 'attn2' in name and (name.endswith('to_k') or name.endswith('to_v'))
        and f"{name}.U" in svd_components and f"{name}.S" in svd_components
    ]
    if len(all_layer_names) != 32:
        raise ValueError(f"Expected 32 cross-attention projection layers, got {len(all_layer_names)}")
    layer_groups = {
        "shallow": all_layer_names[:9] + all_layer_names[28:32],
        "middle": all_layer_names[9:12],
        "deep": all_layer_names[12:28]
    }

    final_editing_config = {}
    for group_name, layers in layer_groups.items():
        if group_name not in optimal_params: continue

        group_params = optimal_params[group_name]
        threshold = group_params['threshold']
        erase_scale = group_params['erase_scale']
        print(f"  - Applying parameters for layer group [{group_name}]: t={threshold:.4f}, s={erase_scale:.2f}")

        for layer_name in layers:
            U = svd_components[f"{layer_name}.U"]
            S = svd_components[f"{layer_name}.S"]
            p_matrix = reconstruct_p_matrix(U, S, threshold)
            final_editing_config[layer_name] = {
                'p_matrix': p_matrix,
                'erase_scale': erase_scale
            }

    hnse(
        pipe=pipe,
        edit_concepts=edit_concepts,
        guide_concepts=guide_concepts,
        layer_wise_editing_config=final_editing_config,
        save_dir=args.save_dir,
        exp_name=args.exp_name,
        torch_dtype=torch_dtype
    )