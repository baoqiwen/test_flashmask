export CUDA_VISIBLE_DEVICES=5
export CUTE_DSL_LINEINFO=1

regex_filter="(.*flashmask.*|.*device_kernel.*)"
echo "[Warn] Note that regex is set for filtering kernel names, this could result in missing items. Be careful."
echo "[Warn] regex applied is '$regex_filter'"
ncu --set "full" --nvtx --nvtx-include "flashmask/" --kernel-name=regex:$regex_filter  -o profile_rep/fwd_32k_full -f --import-source yes python profile_flashmask.py

# run this if you want to profile on the benchmark case
# ncu --set "full" --nvtx --nvtx-include "flashmask/" -o profile_rep/fm4_$(date +%Y%m%d_%H%M%S) -f --import-source yes python benchmark_flashmask.py --fm_version 4  --suffix ""
# ncu --set "full" --nvtx --nvtx-include "fa4/" -o profile_rep/fa4_$(date +%Y%m%d_%H%M%S) -f --import-source yes python benchmark_fa4_mask_mod.py

# run this if you want a nsys rep
# nsys profile --stats true -w true -t cuda,nvtx --force-overwrite true -o profile_rep/fm4_$(date +%Y%m%d_%H%M%S) python benchmark_flashmask.py --fm_version 4  --suffix ""
# nsys profile --stats true -w true -t cuda,nvtx --force-overwrite true -o profile_rep/fa4_$(date +%Y%m%d_%H%M%S) python benchmark_fa4_mask_mod_profile.py
