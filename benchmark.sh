python benchmark_flashmask.py --fm_version 1 --suffix ""
sleep 60
python benchmark_flashmask.py --fm_version 3  --suffix ""
sleep 60
python benchmark_flashmask.py --fm_version 4  --suffix ""
sleep 60
python benchmark_flexattention.py
sleep 60
python benchmark_fa4_mask_mod.py
sleep 60
python draw.py
python draw.py --baseline "flexattention"
python plot_radar.py --methods 'fa4_mask_mod' 'flashmaskv4'
