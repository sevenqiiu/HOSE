# HOSE: Hierarchical Orthogonal Subspace Erasure

HOSE edits cross-attention weights in Stable Diffusion v1.4 to erase specified concepts.

## Scripts

- `precompute_svd_components.py`: Precompute SVD components using COCO captions.
- `precompute_svd_components._art.py`: Precompute SVD components for artistic-style erasure.
- `precompute_svd_components._object.py`: Precompute SVD components for object erasure.
- `create_object_dataset.py`: Create prompts for target and preserved objects.
- `hose_sd_Layer-wise_erase.py`: Apply concept erasure using precomputed SVD components and JSON parameters.
- `generate-images-sd.py`: Generate images with the original or edited model.

## Run Concept Erasure

First, generate the required SVD components using the appropriate precomputation script. Then run:

```bash
python hose_sd_Layer-wise_erase.py \
  --edit_concepts "Vincent van Gogh" \
  --concept_type art \
  --params_file ./optimal_params.json \
  --svd_components_path ./zero_space_P_Art/your_svd_file.safetensors \
  --save_dir ./edited_models \
  --exp_name hose_art
```

Replace the concept and SVD file path for your experiment. Use `--concept_type object` for object erasure.

## JSON Parameters

`optimal_params.json` supplies the `threshold` and `erase_scale` for the three layer groups (`shallow`, `middle`, and `deep`). The editing script reads these values to construct projection matrices and control erasure strength.
