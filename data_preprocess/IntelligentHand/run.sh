salloc -N 1 -n 4 -p normal --gres=gpu:1
source /public/software/anaconda3/etc/profile.d/conda.sh
cd ~/projects/IntelligentHand/data_preprocess/IntelligentHand/
conda activate unidexgrasp
python main.py
