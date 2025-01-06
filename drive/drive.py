import os
import time

model_names=[
    'bloom7',
    
    'llama2',
    'llama3',
    'llama3.1',
    ]
shots=[
    # 0,
    # 1,
    # 2,
    4,
    # 6,
    ]

seeds=[
    0,
    90,
    12,
    43,
    53
    ]
bs=4

"""
task1: word pair diambiguation
task2: semantic jugement
task3: semantic constraint
"""
task='task2'
cuda_visible=''
# cuda_visible="CUDA_VISIBLE_DEVICES=0,2 "
for l2 in [
    # 'de',
        #    'fr',
           'es'
           ]:
    for model_name in model_names:
        for k in shots:
            for seed in seeds:
                
                cmd_line=f"{cuda_visible}python task-gen.py --task {task} --l1 en --l2 {l2} --device cuda:3 --k {k} --model_name {model_name} --seed {seed} --bs {bs}"
                print(cmd_line)
                ret_status = os.system(cmd_line)
                if ret_status != 0:
                    print('DRIVER (non-zero exit status from execution)>>{ret_status}<<')
                    exit()
                if k==0:
                    break