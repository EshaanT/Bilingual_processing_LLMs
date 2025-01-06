import pandas as pd
from torch.utils.data import Dataset
import random

WORD_TYPE=[
    'FALSE COGNATE',
    'COGNATE',
    'NON COGNATE'
]

LABEL={
    'FALSE COGNATE':'False',
    'COGNATE':'True',
    'NON COGNATE':'True'
}

ISO_CODE={
    'en':'English',
    'es':'Spanish',
    'fr':'French',
    'de':'German'
}

class Task1(Dataset):
    
    def __init__(self,l1,l2,k,tokenizer,apply_chat=True,seed=0,add_mask=False,tyes_in_prompt=['FALSE COGNATE','COGNATE','NON COGNATE']) -> None:
        random.seed(seed)
        
        self.l1=l1
        self.l2=l2
        self.k=k
        self.add_mask=add_mask
        
        self.tokenizer=tokenizer
        self.apply_chat=apply_chat
        
        self.train_df=pd.read_csv(f'data/{l1}_{l2}_train.csv')
        self.test_df=pd.read_csv(f'data/{l1}_{l2}_test.csv')
        
        self.task1_temp='"{word_l1}" and "{word_l2}" -> {Label}'
        self.instruction=f'Given an {ISO_CODE[l1]} and {ISO_CODE[l2]} word label them True if they have the same meaning else label them as False.'
        self.demos=''
        if self.k>0:
            self.demos=self.get_demos(seed=seed)
            self.demos='Following are a few labeled examples to assist you with this task.\n'+self.demos
            
        self.ids,self.input_languages,self.prompt,self.label,self.p_type,self.incorrect_label=self.get_test_prompts()
    
    def print_test_example(self,index=0):
        print(self.prompt[index])
    
    def __len__(self):
        return len(self.prompt)
    
    def __getitem__(self,idx):
        
        prompt=self.prompt[idx]
        label=self.label[idx]
        p_type=self.p_type[idx]
        incorrect_label=self.incorrect_label[idx]
        # print(options)
        
        return {
            'prompt':prompt,
            'label':label,
            'type':p_type,
            'incorrect':incorrect_label
        }
        
    def get_demos(self,seed=0,tyes_in_prompt=['FALSE COGNATE','COGNATE']):
        
        df_demo=self.train_df.groupby('type').sample(n=self.k,random_state=seed)
        df_demo=df_demo[df_demo['type'].isin(tyes_in_prompt)]
        demos=[]
        
        for i,row in df_demo.iterrows():
            
            word_l1=row[self.l1]
            word_l2=row[self.l2]
            label=LABEL[row['type']]
            demo=self.task1_temp.format(
                word_l1=word_l1,
                word_l2=word_l2,
                Label=label
            )
            demos.append(demo)
            
        random.shuffle(demos)
        return '\n'.join(demos)
    
    def get_test_prompts(self):
        
        test_prompts_ids=[]
        test_prompts=[]
        test_prompts_label=[]
        test_prompts_type=[]
        test_prompt_incorrect_label=[]
        test_prompts_languages=[]
        
        for i,row in self.test_df.iterrows():
            
            l1=row[self.l1]
            l2=row[self.l2]
            label=LABEL[row['type']] 
            incorrect_label='True' if 'True'!=label else 'False'
            # options=[label,incorrect_label]
            
            input_example=self.task1_temp.format(
                word_l1=l1,
                word_l2=l2,
                Label=''
            )
            input_example=self.instruction+'\n'+(self.demos+'\nNow label the following\n' if self.k>0 else '')+input_example+('[MASK]' if self.add_mask else '')
            
            if self.apply_chat:
                input_example=self.tokenizer.apply_chat_template(
                                                                [{'role':'user',
                                                                'content':input_example}]
                                                                ,tokenize=False)
            test_prompts.append(input_example)
            test_prompts_label.append(label)
            test_prompts_ids.append(row['ID'])
            test_prompts_languages.append(self.l2)
            test_prompts_type.append(row['type'])
            test_prompt_incorrect_label.append(incorrect_label)
        
        return test_prompts_ids,test_prompts_languages,test_prompts,test_prompts_label,test_prompts_type,test_prompt_incorrect_label
    
        

class Task2(Dataset):
    
    """
    Task 2 is meaning selection. 
    Given a word and options. The model must select the correct option of meaning of the word.
    """
    
    def __init__(self,l1,l2,k,tokenizer,apply_chat=True,seed=0,add_mask=False,tyes_in_prompt=['FALSE COGNATE','COGNATE','NON COGNATE']) -> None:
        random.seed(seed)
        
        self.l1=l1
        self.l2=l2
        self.k=k
        self.add_mask=add_mask
        
        self.tokenizer=tokenizer
        self.apply_chat=apply_chat
        
        self.train_df=pd.read_csv(f'data/{l1}_{l2}_train.csv')
        self.test_df=pd.read_csv(f'data/{l1}_{l2}_test.csv')
        
        self.iso_lang=ISO_CODE
        
        self.task2_temp='What is the meaning of {word} in {lang}?\n1. {A}\n2. {B}\nLabel: {Label}'
        self.instruction='Given a word and two options, select the correct meaning of the word from the two provided options.'
        self.demos=''
        if self.k>0:
            self.demos=self.get_demos(seed=seed)
            self.demos='Following are a few labeled examples to assist you with this task.\n'+self.demos
            
        self.ids,self.input_languages,self.prompt,self.label,self.p_type,self.incorrect_label=self.get_test_prompts()
    
    def print_test_example(self,index=0):
        print(self.prompt[index])
    
    def __len__(self):
        return len(self.prompt)
    
    def __getitem__(self,idx):
        
        prompt=self.prompt[idx]
        label=self.label[idx]
        p_type=self.p_type[idx]
        incorrect_label=self.incorrect_label[idx]
        # print(options)
        
        return {
            'prompt':prompt,
            'label':label,
            'type':p_type,
            'incorrect':incorrect_label
        }
    def get_label(self,correct,incorrect):
        
        coin=random.randint(1,2)
        
        if coin==1:
            return correct,incorrect,'1.'

        return incorrect,correct,'2.'
        
    
    def get_demos(self,seed=0,tyes_in_prompt=['FALSE COGNATE','COGNATE']):
        df_demo=self.train_df.groupby('type').sample(n=self.k,random_state=seed)
        df_demo=df_demo[df_demo['type'].isin(tyes_in_prompt)]
        demos=[]
        
        for lang in [self.l1,self.l2]:
            iso_2_lang=self.iso_lang[lang]
            
            for i,row in df_demo.iterrows():
                
                word=row[lang]
                meaning=row[lang+'_meaning']
                incorrect=row[lang+'_neg_meaning']
                
                a,b,label=self.get_label(meaning,incorrect)
                demo=self.task2_temp.format(
                    word=word,
                    lang=iso_2_lang,
                    A=a,
                    B=b,
                    Label=label
                )
                
                demos.append(demo)
        demos=random.sample(demos,k=len(demos)//2)
        # random.shuffle(demos)
        return '\n'.join(demos)
    
    def get_test_prompts(self):
        
        
        test_prompts=[]
        test_prompts_label=[]
        test_prompts_type=[]
        test_prompt_incorrect_label=[]
        test_word_language=[]
        ids=[]
        
        for lang in [self.l1,self.l2]:
            iso_2_lang=self.iso_lang[lang]
            for i,row in self.test_df.iterrows():
                
                word=row[lang]
                meaning=row[lang+'_meaning']
                incorrect=row[lang+'_neg_meaning']
                
                a,b,label=self.get_label(meaning,incorrect)
                incorrect_label='1.' if '2.'==label else '2.' 
                input_example=self.task2_temp.format(
                    word=word,
                    lang=iso_2_lang,
                    A=a,
                    B=b,
                    Label=''
                )
                
                input_example=self.instruction+'\n'+(self.demos+'\n Now label the following\n' if self.k>0 else '')+input_example+('[MASK]' if self.add_mask else '')
                
                if self.apply_chat:
                    input_example=self.tokenizer.apply_chat_template(
                                                                [{'role':'user',
                                                                'content':input_example}]
                                                                ,tokenize=False)
                
                test_prompts.append(input_example)
                test_prompts_label.append(label.strip('.'))
                test_prompts_type.append(row['type'])
                test_prompt_incorrect_label.append(incorrect_label.strip('.'))
                test_word_language.append(lang)
                ids.append(row['ID'])
        
        return ids,test_word_language,test_prompts,test_prompts_label,test_prompts_type,test_prompt_incorrect_label
    
    
class Task3(Dataset):
    
    """
    Task 3 is meaning selection. 
    Sentence Understanding.
    """
    
    def __init__(self,l1,l2,tokenizer,setup,apply_chat=True) -> None:
        
        self.l1=l1
        self.l2=l2
        self.setup=setup
        
        if setup=='':
            raise ValueError("Incorrect arg.setup")
        
        self.tokenizer=tokenizer
        self.apply_chat=apply_chat
        
        self.df=pd.read_csv(f'data/task3_{l1}_{l2}.csv')
        
        self.iso_lang=ISO_CODE
        
        # self.task2_temp='What is the meaning of {word} in {lang}?\n1. {A}\n2. {B}\nLabel: {Label}'
        if self.setup=='cross':
            # self.instruction='Given the following code-mix sentence in {l1}-{l2}. Tell me the meaning of the word enclosed in double quotes. Also does the sentence make any meaning?'
            self.instruction='Given the following code-mix sentence in {l1}-{l2}. Answer the following questions about the word enclosed in double quotes\nQ1. What is the language of the enclosed word?\nQ2. What is the meaning of the enclosed word?\nQ3. Does the given sentence make sense in context with the meaning of the enclosed word?'
        else:
            self.instruction='Given the following sentence in {lang}. Tell me the meaning of the word enclosed in double quotes. Also does the sentence make any meaning?'
        self.ids,self.input_languages,self.prompt,self.label,self.p_type,self.incorrect_label=self.get_test_prompts()
    
    def print_test_example(self,index=0):
        print(self.prompt[index])
    
    def __len__(self):
        return len(self.prompt)
    
    def __getitem__(self,idx):
        
        prompt=self.prompt[idx]
        label=self.label[idx]
        p_type=self.p_type[idx]
        incorrect_label=self.incorrect_label[idx]
        # print(options)
        
        return {
            'prompt':prompt,
            'label':label,
            'type':p_type,
            'incorrect':incorrect_label
        }
    
    def get_test_prompts(self):
        
        
        test_prompts=[]
        test_prompts_label=[]
        test_prompts_type=[]
        test_prompt_incorrect_label=[]
        test_word_language=[]
        ids=[]
        
        for lang in [self.l1,self.l2]:
            target=lang
            if self.setup=='cross':
                target=self.l1 if target==self.l2 else self.l2
                instruction=self.instruction.format(l1=ISO_CODE[self.l1],l2=ISO_CODE[self.l2])
            else:
                instruction=self.instruction.format(lang=ISO_CODE[target])
            
            for i,row in self.df.iterrows():
                
                word=row[lang]
                hig_sem=row['high_'+target].replace('<MASK>','"'+word+'"')
                low_sem=row['low_'+target].replace('<MASK>','"'+word+'"')
                
                input_example_hs=instruction+'\n'+hig_sem
                input_example_ls=instruction+'\n'+low_sem
                
                if self.apply_chat:
                    input_example_hs=self.tokenizer.apply_chat_template(
                                                                [{'role':'user',
                                                                'content':input_example_hs}]
                                                                ,tokenize=False)
                    input_example_ls=self.tokenizer.apply_chat_template(
                                                                [{'role':'user',
                                                                'content':input_example_ls}]
                                                                ,tokenize=False)
                
                test_prompts.append(input_example_hs)
                test_prompts_label.append('hs')
                test_prompts_type.append(row['type'])
                test_prompt_incorrect_label.append('')
                test_word_language.append(lang)
                ids.append(row['ID'])
                
                test_prompts.append(input_example_ls)
                test_prompts_label.append('ls')
                test_prompts_type.append(row['type'])
                test_prompt_incorrect_label.append('')
                test_word_language.append(lang)
                ids.append(row['ID'])
        
        return ids,test_word_language,test_prompts,test_prompts_label,test_prompts_type,test_prompt_incorrect_label