<p align="center">
  <img src="./task2kb.png" width="400" />
  
</p>

<h3 align="center">
    <p>Task-oriented Knowledge Base for developing advanced conversational AI</p>
</h3>


---

<div>
  <img src="./overview.png" width="400" align="left"> 
  
  **Task2KB** is a task-oriented instructional knowledge base, which offers structured instructions and rich types of related information on tasks across **19** categories. Along with rich task-related information, Task2KB also enables accessing the available knowledge via various retrieval techniques (field-based and dense retrievers).  
  Additionally, to illustrate the value of Task2KB, we experimentally augment the knowledge into TOD models with two novel development pipelines: (1) fine-tune LLMs with knowledge from Task2KB for boosting TOD model performance, (2) direct context-augmentation with available knowledge. We observe significant and consistent in advancing the performance of recent TOD models.
</div>

---

## Categories of Tasks

Category (Quantity) | Category (Quantity) | Category (Quantity) 
:-------------------------:|:-------------------------:|:-------------------------:
Arts & Entertainment (13,320) | Car & Other Vehicles (4,925) | Computers & Electronics (24,447)
Education & Communication (24,530) | Family Life (5,383) | Finance & Business (15,305)
Food & Entertaining (9,966) | Health (25,471) | Hobbies & Crafts (22,383)
Holidays & Traditions (2,569)| Home & Garden (24,885) | Pets & Animals (15,087)
Philosophy & Religion (2,872) | Relationship (7,880) | Sports & Fitness (10,094)
Style (18,854)  | Travel (4,826) | Work (11,524)
Youth (7,112) ||

To access the Task2KB, we enable a quick access via json files that saved in Google drive [link](
https://drive.google.com/drive/folders/1heZ15q5N85EysNLFLlCuPGmLw-Innojk?usp=share_link). We further illustrate multiple applications of Task2KB.

## Task Attributes
Aside from the available knowledge of tasks that we collected from [WikiHow](https://www.wikihow.com/Main-Page), for each task, we also identify and share its related attributes that have the potential of being task slots for task completion. One example list of attributes is ['packaging', 'fridging method', 'shipping policy'] for the task of 'How to ship food'. The attribute identification strategy can be summarised into following steps:

1. Entity tagging with [TagMe](https://sobigdata.d4science.org/web/tagme/) that identifies entities can link to a specific wikipedia webpage.
2. Aggregate section titles or names, which are meaningful summarise or actions, as candidate attributes.
3. Compare the semantic similarity, via pre-trained [BERT](https://huggingface.co/docs/transformers/model_doc/bert) model, between task titles and the candidate attributes and use the top-5 similar attributes as the finalised attributes for a task.

The resulting attribute data can be accessed [here](https://drive.google.com/drive/folders/1blaPTObkFI1g72zj5Cigt2JIy4ILqAR2?usp=drive_link).

## Synthetic Dataset Generation and Fine-tune LLMs
Task2KB is capable of generating synthetic task-oriented conversational datasets with its step-wise instructions. We public available the implementation from fine-tuning a dialogue generator in T5 or Flan-T5, as well as the dialogue generation code. In addition, we show how we can further fine-tune a distilgpt2 model for response generation.

### data processing
The shared code in [data_processing](https://github.com/wangxieric/task2kb-resource/tree/main/data_preprocessing) folder includes the implementation that use three candidate conversational datasets, ORConvQA, QReCC and MultiWoZ, for the comparison of using different training data for developing dialog generators.

### dialog generator training
Next, in [question_generator_train](https://github.com/wangxieric/task2kb-resource/tree/main/question_generator_train), we also explore various strategies in generating the synthetic dialogues, such as using the Flan-T5 model that trained on ORConvQA dialogues and then further fine-tuned on the MultiWoZ dataset (i.e., [Flan_T5_OR_MultiWoZ.py](https://github.com/wangxieric/task2kb-resource/blob/main/question_generator_train/Flan_T5_OR_MultiWoZ.py)). 
We also publicly available the saved question_generation_model, which is trained_on_multiwoz:
https://drive.google.com/file/d/1H_dMut5HV72as8zkLV_w07Tz8HkM-OYA/view?usp=share_link

### synthetic dialog generation & model fine-tuning
After having the ability in dialogue generation or progressively generating questions with step description as answers, we move to the implementation of generating synthetic dialogues, which uses the last generated pair of question and answer as context and step-wisely generating full dialogues (see [dialogue_generation.py](https://github.com/wangxieric/task2kb-resource/blob/main/dialogue_generation.py)). 

Next, with the generated dialogues ([INST2DIAL-Auto](
https://drive.google.com/drive/folders/1ZVPeWrYHRMJ_6MBqGWYuM6eYKJTw3-bC?usp=share_link)), we can fine-tune a large language model (e.g., distilgpt2) for task-oriented response generation (in [distilgpt2-train-RespGen.py](https://github.com/wangxieric/task2kb-resource/blob/main/distilgpt2-train-RespGen.py)). The fine-tuned distilGPT2 model is also available here: [link](https://drive.google.com/file/d/1HaY_pWIR6AgxXXmKA-MqFG5kS0iK-zsa/view?usp=share_link). 

In particular, to show the effectiveness of using Task2KB, we experimentally compare with the use if [wikidialog](https://github.com/google-research/dialog-inpainting) that was generated using wikipedia passages. The corresponding checkpoint is also available [link](https://drive.google.com/file/d/1Ls3XRgYPjs4oH-SeZw-lMCg2OzNUSmL-/view?usp=share_link). Then, we use the fine-tuned distilgpt2in two recent advanced task-oriented conversational model (UBAR and JSA-TOD), and compare the performance differences:

Experimental Results of UBAR & variants         |  Experimental Results of JSATOD & variants
:-------------------------:|:-------------------------:
![](./result_ubar.png)  |  ![](./result_jsatod.png)


## Indexing Methods for Knowledge Access 
On the other hand, to enable a direct use of Task2KB, we also implment two document indexing methods: field-based indexing and dense indexing:

### Field-based Indexing
To be filled.

### Dense Indexing

The dense indexing and knowledge access are implemented with a joined effort of Facebook [Faiss](https://github.com/facebookresearch/faiss) and Dense Passgae Retrieval ([DPR](https://github.com/facebookresearch/DPR)). To balance the document length and information specificity, we structure each step of task instructions into the following format: 

    id [tab] introduction + step description [tab] Task title
    
Afterwards, we use the `[generate_dense_embeddings.py](https://github.com/facebookresearch/ParlAI/blob/main/parlai/agents/rag/scripts/generate_dense_embeddings.py)' script in [ParlAI](https://parl.ai/docs/index.html) to encode the information and running:

    python generate_dense_embeddings.py -mf zoo:hallucination/multiset_dpr/hf_bert_base.cp --dpr-model True --passages-file step_info_cl.tsv  
    --outfile task2kb_index/step_info --num-shards 50 --shard-id 0 -bs 32
    
The `step_info_cl.tsv' file can be obtained via the following [link](https://drive.google.com/file/d/1QUNZ20hnRb_rbSenS12d1cTDW_niVk27/view?usp=share_link). Next, we use the exact indexing type from ParlAI as well in addressing the indexing implementation as follows:

    python index_dense_embeddings.py --retriever-embedding-size 768  \
    --embeddings-dir task2kb_index/ --embeddings-name task2kb --indexer-type exact

Then, the resulting index of the dense embedding allow the quick access of task-oriented information for deploying knowledge-augmented task-oriented conversational models.


<!-- This repository is about a resource paper, 'Task-Oriented Dialog System with Structured Instructional Knowledge' (under review). -->



<!-- In this work, we publicly available two task-oriented conversational datasets joined with a knowledge graph, Task2KB.

The links to access each dataset as well as the knowledge graph are given as follows:

INST2DIAL-Manual: 
https://drive.google.com/drive/folders/1hROuwee8BqfPtXkvTo_jmdK7korlntnP?usp=share_link

Task2KB:
https://drive.google.com/drive/folders/1heZ15q5N85EysNLFLlCuPGmLw-Innojk?usp=share_link

The data are presented as per different categories with json format, which allow them to be easy access.

We also public available the code in developing the question generators with T5/Flan-T5 encoder-decoder models, as well as the code
for fine-tuning a response generator based upon the generated datasets. -->
