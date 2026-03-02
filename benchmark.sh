export CUDA_VISIBLE_DEVICES=5
# nsys profile --stats true -w true -t cuda --force-overwrite true -o profile_rep/fm4_$(date +%Y%m%d_%H%M%S) python benchmark_flashmask.py --fm_version 4  --suffix ""

python benchmark_fa4_mask_mod.py

# sleep 60
# python draw.py
# python draw.py --baseline "flexattention"
# python plot_radar.py --methods 'fa4_mask_mod' 'flashmaskv4'
