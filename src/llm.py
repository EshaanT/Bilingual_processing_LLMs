import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float

# We locally host models because of internet restrictions
#Provide paths to the model or there hugging face name in the following dictionary

LLMs_PATH={
    'llama3.1':'/home/models/Meta-Llama-3.1-8B-Instruct/',
    'llama3':'/home/models/Meta-Llama-3-8B-Instruct/',
    'bloom7':'/home/models/bloomz-7b1',
    'llama2':'/home/models/Llama-2-7b-chat-hf',
    'llama2-13b':'/home/models/Llama-2-13b-chat-hf',
    'mistral':'/home/models/Mistral-7B-Instruct-v0.2'
}
def llm_call(input,model,tokenizer):
    
    output=model.generate(**input,
                            pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens=128*4
                            )
    output_sent=tokenizer.batch_decode(output[:,input["input_ids"].shape[-1]:],skip_special_tokens=True)
    return output_sent

def get_model(model_name,device,get_logits=False):
    
    model_path=LLMs_PATH[model_name]    
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            output_hidden_states=get_logits,
            # device_map='auto'
            device_map=device
        )
    return model,tokenizer


# TASK1_LABELS={
#     'True':['True',' True','true',' true','TRUE',' TRUE'],
#     'False':['False',' False','false',' false','FALSE',' FALSE']
# }

def get_prob_diff(
    logits: Float[Tensor,'batch seq d_vocab'], 
    options_tokens: Float[Tensor,'batch n_options']
):
    final_token_prob: Float[Tensor,"batch d_vocab"]=logits[:,-1,:].softmax(dim=1)    
    answer_prob: Float[Tensor,"batch n_options"]=final_token_prob.gather(dim=-1,index=options_tokens)
    
    prob_diff=answer_prob[:,0]-answer_prob[:,1]
    
    return prob_diff.numpy(),answer_prob[:,0].numpy(),answer_prob[:,1].numpy()
    
    