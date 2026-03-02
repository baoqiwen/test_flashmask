export CUDA_VISIBLE_DEVICES=3
export FLAGS_alloc_fill_value=255
export FLAGS_use_system_allocator=1
export FLAGS_check_cuda_error=1
python -m pytest -v test_fwd_md5sum.py 2>&1 | tee test.log
# python -m pytest -v test_bwd_md5sum.py 2>&1 | tee test.log

# run this if you want to update gt
# python test_fwd_md5sum.py
# python test_bwd_md5sum.py
