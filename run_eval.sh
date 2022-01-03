export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/evaluate_reconstruction.py --exp_dir experiments/ --experiment clevr --dataset clevr \
--checkpoint experiments/clevr/spade_64_clevr_model.pt --with_feats True --generative True --save_images True
