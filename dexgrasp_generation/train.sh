python ./network/train_grab_baseline.py --config-name ipdf_config \
                          --exp-dir ./runs/ipdf_train

# python ./network/train_grab_baseline.py --config-name cm_net_config \
#                           --exp-dir ./runs/cm_net_train

python ./network/train_grab_baseline.py --config-name glow_joint_config \
                          --exp-dir ./runs/glow_joint

