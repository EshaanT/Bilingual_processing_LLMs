# Bilingual_processing_LLMs

![Experiment Results](figures/cognates_example.pdf)

## Dataset

1. Words and thier meaning
    The words along with their meanings can be found in the `data` folder in the file `en_es_dataset.csv`, `en_fr_dataset.csv`, and `en_de_dataset.csv`.

2. Semantic Constraint Sentences
    The semantic constraint sentences are located in the `data` folder in the file `task3_en_es.csv`, `task3_en_fr.csv`, and `task3_en_de.csv`

## Installation

To set up the project and install all required dependencies, follow these steps:

1. Clone the Repository

    ``` bash
    git clone https://github.com/EshaanT/Bilingual_processing_LLMs.git
    cd Bilingual_processing_LLMs
    ```
2. Set Up a Virtual Environment
    ``` bash
    python -m venv venv
    . venv/bin/activate   # On Linux
    ```
3. Install Dependencies
    ```
    pip install -r requirements.txt
    ```
## Experimental runs

Use the `task-gen.py` file to run the experiment. You can specify the task using the `--task` option. The available tasks are:
* Task 1: Word pair disambiguation
* Task 2: Semantic judgment
* Task 3: Semantic constraint

Set `--l1` as "en" and `--l2` as either "es", "fr", or "de". For task1 and 2 remeber to set `--k` as 4
 For further convinence we have added 'drives.py' in the `drive` folder. Select 



