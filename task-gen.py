import argparse
from src.data import Task1,Task2,Task3
from src.llm import get_model,get_prob_diff,llm_call
from torch.utils.data  import DataLoader
import pandas as pd
import os
import random
from tqdm import tqdm

# 'FALSE COGNATE','COGNATE','NON-COGNATE'

def main(args):
    
    random.seed(args.seed)
    
    model,tokenizer=get_model(args.model_name,args.device)
    tokenizer.padding_side='left'
    tokenizer.pad_token_id=tokenizer.eos_token_id
    
    if args.task=='task1':
        task1_dataset=Task1(
        l1=args.l1,
        l2=args.l2,
        k=args.k,
        tokenizer=tokenizer,
        apply_chat=False if args.model_name in ['bloom7'] else True,
        seed=args.seed
        
    )
    elif args.task=='task2':
        
        task1_dataset=Task2(
        l1=args.l1,
        l2=args.l2,
        k=args.k,
        tokenizer=tokenizer,
        apply_chat=False if args.model_name in ['bloom7'] else True,
        seed=args.seed
    )
    elif args.task=='task3':
        
        task1_dataset=Task3(
        l1=args.l1,
        l2=args.l2,
        tokenizer=tokenizer,
        setup=args.setup,
        apply_chat=False if args.model_name in ['bloom7'] else True,
    )
        
    else:
        raise KeyError("Wrong task feed")
    
    if args.bs!=1:
        print('Warning: batch size is more than 1. We will left pad the input, that might cause some deviations in analysis.')
    
    print("Example prompt\n")
    task1_dataset.print_test_example()
    
    dataset_loader=DataLoader(
        task1_dataset,
        batch_size=args.bs,
        shuffle=False
    )
    ids=task1_dataset.ids
    outputs=[]
    input_prompt=task1_dataset.prompt
    input_language=task1_dataset.input_languages
    input_type=task1_dataset.p_type
    input_labels=task1_dataset.label
    
    
    for batch in tqdm(dataset_loader):
        
        batch_tokenize=tokenizer(
            batch['prompt'],
            return_tensors='pt',
            padding=True
        )
        batch_tokenize.to(args.device)
        output=llm_call(batch_tokenize,model,tokenizer)
        # print(output)
        outputs.extend(output)
    
    path=f'results/{args.model_name}_{args.task}_l1={args.l1}_l2={args.l2}_{args.k}{args.setup}.csv'
    
    if os.path.exists(path):
        results=pd.read_csv(path)
    else:
        results={
            'id':ids,
            'prompt':input_prompt,
            'true_label':input_labels,
            'lang':input_language,
            'type':input_type
        }
        results=pd.DataFrame(results)
    results[f'out_{args.seed}']=outputs
    results.to_csv(path,index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--dataset_name',required=True, type=str)
    parser.add_argument('--task',required=True, type=str)
    parser.add_argument('--l1',required=True, type=str)
    parser.add_argument('--l2',required=True, type=str)
    parser.add_argument('--model_name',default='llama3.1', type=str)
    parser.add_argument('--device',required=True,type=str)
    parser.add_argument('--k', default=0,type=int)
    parser.add_argument('--setup', default='',type=str)
    parser.add_argument('--seed', default=0,type=int)
    parser.add_argument('--bs', default=1,type=int)
    args = parser.parse_args()  
    main(args)



