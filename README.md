# GSoC' 19

**Table of Content**
- [Summary of Project](#summary-of-project)
  - [First Objective](#first-objective)
  - [Second Objective](#second-objective)
- [Tutorial](#tutorial)
- [Weekly Progress](#weekly-progress)

## Summary of Project
The project initially sets out to achieve two goals – update the annotation system for Red Hen’s NewsScape dataset to FrameNet 1.7 and expand FrameNet 1.7 through a knowledge-driven approach and a distributional semantics approach. 


### First Objective
I achieved the first objective by updating the annotation system for Red Hen's NewsScape dataset to FrameNet 1.7 using PyDaisy and Open-Sesame parsers. However, there are two critical changes.

First, I did not use `pyfn` library as suggested in my proposal because the library is not intended for annotating sentences outside of FrameNet. Besides, I could not resolve the bugs when deploying the library despite opening issues and working with the library's creator through GitHub. I changed to use PyDaisy and OpenSesame standalone library. PyDaisy is the alternative to SimpleFrameID for frame identification, and the library OpenSesame is also capable of identifying target words and frames.

Second, SEMAFOR could not be implemented because `pyfn` was not functional, and the pre-trained models (MaltParser trained on Penn Treebank and SEMAFOR trained on the FrameNet 1.7 datasets) in [repository](https://github.com/AlenUbuntu/semafor_Framenet_v1.7) were missing. I had sent multiple follow-up emails to the author and the person-of-contact listed in the repository, but I had not received any reply to date.

**Documentation of PyDaisy and Open-Sesame parsers** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20PyDaisy%20and%20OpenSesame.md

**Detailed reasons of not using `pyfn`** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/GSoC%20Phase%201_%20Report%20on%20pyfn.pdf


### Second Objective
#### Knowledge-driven approach - BabelNet
I did not successfully deploy the Singularity container for expanding FrameNet using BabelNet synsets. Due to the API request limits, it would take three months for completing the representing FrameNet frames with BabelNet synsets. This expected duration does not include finding the best representative BabelNet synsets for new lexical units. In other words, it would take more than three months to use BabelNet Web API to expand FrameNet.

My mentor Prof. Tiago and I have emailed Roberto Navigli to request for the entire BabelNet dataset, but to date, we have not heard back from him. However, despite this setback, I have created the Python Script `/home/zxy485/zxy485gallinahome/week5-8/babelnet.py` that maps a frame and POS to the set of BabelNet synsets that generalizes the LUs in the particular frame. The dictionary is stored as a pickled file `{frame_name}_subgraphs.p` in the folder `/home/zxy485/zxy485gallinahome/week5-8/babelnet_synsets`. 

To date, I have processed 52 frames and generated their BabelNet synsets representation for four different POS - nouns, verbs, and adjectives/adverbs. 

#### Distributional Semantics Approach - BERT Embeddings
Before generating frame embeddings, I read through 10 papers to understand the recent practices with creating frame embeddings and summarized their methodologies in a report. I decided to replace the initial proposed DSSM model with BERT model to generate embeddings for lexical units and frames. The reason is that BERT uses attention models to create sentence embeddings, which better capture the semantic concepts of a sentence.

The first application of the embeddings is identifying antonyms within FrameNet using a decision tree model trained with antonyms from WordNet synsets. There are two improvements made to increase the accuracy of the machine learning model. First, antonymous pairs in the training dataset share the same POS. Second, include the dependency level and relations (in integer representation) as the attribute in addition to the POS and cosine similarity of the antonym pair.

The second application of the embeddings is identifying new lexical units from UCLA NewsScape dataset that can be added into FrameNet 1.7. The figure below shows the final pipeline for expanding FrameNet 1.7 using the lexical units in NewsScape dataset. There are two additional changes to my proposed methods. First, I included an unsupervised clustering Affinity Propagation model to cluster the non-compliant new lexical units. Second, I assigned frame names to the clusters using alignments built by multilingual FrameNet projects, namely KoreanFN and BrasilFN. 

![Pipeline of Expanding FrameNet with NewsScape](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/images/Final%20Pipeline%20of%20Expanding%20FrameNet%20with%20NewsScape.png)


**Report of Frame Embeddings Generation** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Background%20Research%20-%20Frame%20Embeddings.pdf

**Documentation of Generating Embeddings for FrameNet Lexical Units** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Generating%20Embeddings%20for%20FrameNet%20Lexical%20Units.md

**Documentation of Generating Embeddings for WordNet Synsets** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Generating%20Embeddings%20for%20WordNet%20Synsets.md

**Documentation of Antonym Detection** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Antonym%20Detection.md

**Documentation of Expanding FrameNet with NewsScape and Embedding** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Expanding%20FrameNet%20with%20NewsScape%20and%20Embedding.md

**Documentation of Clustering Non-Compliant Lexical Units** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Clustering%20Non-Compliant%20Lexical%20Units.md

**Documentation of Frame Assignment using Multilingual FN** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Frame%20Assignment%20using%20Multilingual%20FN.md


## Tutorial
I put the tutorial for running the deployed Singularity containers for annotating NewsScape dataset, identifying antonyms and expanding FrameNet with NewsScape dataset in the page [TUTORIAL.md](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/TUTORIAL.md).


## Weekly Progress
The page [PROGRESS.md](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/PROGRESS.md) documents my weekly progress throughout GSoC' 19. 
