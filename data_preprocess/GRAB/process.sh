export GRAB_DATASET_PATH=/home/liuym/data/GRAB/grab
export SMPLX_MODEL_FOLDER=/home/liuym/models/smpl
export PATH_TO_SAVE_DATA=/remote-home/share/yumeng/data/

python for_unidex.py --grab-path $GRAB_DATASET_PATH \
                                  --model-path $SMPLX_MODEL_FOLDER \
                                  --out-path $PATH_TO_SAVE_DATA
