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

<div> <br /> </div>

**Categories of Tasks:**

Category (Quantity) | Category (Quantity) | Category (Quantity) 
:-------------------------:|:-------------------------:|:-------------------------:
Arts & Entertainment (13,320) | Car & Other Vehicles (4,925) | Computers & Electronics (24,447)
Education & Communication (24,530) | Family Life (5,383) | Finance & Business (15,305)
Food & Entertaining (9,966) | Health (25,471) | Hobbies & Crafts (22,383)
Holidays & Traditions (2,569)| Home & Garden (24,885) | Pets & Animals (15,087)
Philosophy & Religion (2,872) | Relationship (7,880) | Sports & Fitness (10,094)
Style (18,854)  | Travel (4,826) | Work (11,524)
Youth (7,112) ||


This repository is about a resource paper, 'Task-Oriented Dialog System with Structured Instructional Knowledge' (under review).

Experimental Results of UBAR & variants         |  Experimental Results of JSATOD & variants
:-------------------------:|:-------------------------:
![](./result_ubar.png)  |  ![](./result_jsatod.png)

In this work, we publicly available two task-oriented conversational datasets joined with a knowledge graph, Task2KB.

The links to access each dataset as well as the knowledge graph are given as follows:

saved_question_generation_model (trained_on_multiwoz):
https://drive.google.com/file/d/1H_dMut5HV72as8zkLV_w07Tz8HkM-OYA/view?usp=share_link

INST2DIAL-Auto:
https://drive.google.com/drive/folders/1ZVPeWrYHRMJ_6MBqGWYuM6eYKJTw3-bC?usp=share_link

INST2DIAL-Manual: 
https://drive.google.com/drive/folders/1hROuwee8BqfPtXkvTo_jmdK7korlntnP?usp=share_link

Task2KB:
https://drive.google.com/drive/folders/1heZ15q5N85EysNLFLlCuPGmLw-Innojk?usp=share_link

The data are presented as per different categories with json format, which allow them to be easy access.

We also public available the code in developing the question generators with T5/Flan-T5 encoder-decoder models, as well as the code
for fine-tuning a response generator based upon the generated datasets.
