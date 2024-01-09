#!/bin/bash
#SBATCH --job-name=GPU
#SBATCH --partition=mayuexin
#SBATCH --cpus-per-task=8
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 # NVIDIATITANRTX NVIDIAA40 gpu:NVIDIAGeForceRTX2080Ti:1 NVIDIAGeForceRTX2080Ti TeslaM4024GB NVIDIAGeForceGTX1080 NVIDIAA40
#SBATCH --output=/public/home/v-liuym/sbatch-out/%j.out
#SBATCH --error=/public/home/v-liuym/sbatch-out/%j.err

bash ./run.sh
