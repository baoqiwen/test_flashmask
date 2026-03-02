export CUDA_VISIBLE_DEVICES=6
export FLAGS_alloc_fill_value=255
export FLAGS_use_system_allocator=1
export FLAGS_check_cuda_error=1
python -m pytest -v test_flashmask.py 2>&1 | tee test.log
