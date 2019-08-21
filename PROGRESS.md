# Progress Update

#### Week 1 (5/27/19 - 5/31/19)
1. **Familiarize with the NewsScape dataset + (Stretch) Implementing the pre-processing pipeline**: I have familiarized myself with the dataset. I have also completed the first part of the pre-processing pipeline: concatenating the text.
2. **Train Semafor, SIMPLEFRAMEID and OpenSesame with FrameNet 1.7 annotation sets**: I run into two problems with using the library `pyfn` as proposed in my proposal. First, some Python files for preprocessing and training the SEMAFOR and OpenSesame are written in Python 2 and some are in Python 3. I am looking into the Singularity Hub to find containers that contain both Python 2 and 3 but couldn’t find any for now. In fact, I can only find the container with Python 3. There are two solutions. One is to train locally (without using the Singularity container). The second is to modify a container with Python 3 to include Python 2. I am seeking out instructions to do so and I am getting in contact with HPC support team. Second issue is the lack of clear documentation about training SIMPLEFRAMEID. In fact, I run into a bug of lacking embedding file for training SIMPLEFRAMEID, and I open an issue on GitHub (https://github.com/akb89/pyfn/issues/13)

#### Week 2 (6/3/19 - 6/7/19):
1. **(Continuing from Week 1) Train Semafor, SIMPLEFRAMEID and OpenSesame with FrameNet 1.7 annotation sets**: There are still some [bugs](https://github.com/akb89/pyfn/issues?q=is%3Aissue+author%3Ayongzx+is%3Aclosed) with the library but I have completed Week 1 tasks with training of SIMPLEFRAMEID, SEMAFOR, and OpenSesame.
2. **Implement the pre-processing pipeline that tags POS and parse dependencies**: I have implemented the NLP4J and BPMS parser for POS-tagging and dependency parsing.
3. **Annotate NewsScape dataset with the trained Semafor and OpenSesame parsers**: I have some setbacks because I discover that the library is a closed system which only works on the gold-annotated training and testing files in FrameNet. Unlike what I formerly written in the proposal, it is not directly transferable to Red Hen NewsScape dataset. I am still working on this task by reading the codes and have been experimenting with different ways of replacing the input dataset with NewsScape dataset.

#### Week 3 (6/10/19 - 6/14/19):
1. **Write a [report](https://github.com/yongzx/GSoC-2019-FrameNet/blob/ef575ddb029e364c71be70a87ffc334675f92fb6/GSoC%20Phase%201_%20Report%20on%20pyfn.pdf) on my failure with integrating the `pyfn` library.** 
2. **Video called with Prof. Torrent to understand the other tools for annotaing the dataset.** I learned that I could attempt PyDaisy for the frame identification process since FrameNet-Brasil (FN-Br) use PyDaisy to annotate the frames in a snippet of words. 
3. **Tested out *PyDaisy* that is used for frame identification with FN-Br database and Google's `SLING` library for frame identification**: Without changing the database from FN-Br to Berkeley FN-1.7, I tested out PyDaisy's performance in annotating the frames in a sentence. I also tested out Google's `SLING` library and unfortunately, I encountered two problems. First the library is not only dedicated to FrameNet. Apparently, I would have to create the training documents from FrameNet 1.7 train datasets using the SLING Python API, and the SLING library only works on a Linux machine. This means that it is very difficult for me to test out the SLING library on my personal Mac computer since I need to install a virtual machine. Second, as I tested out the library on CWRU HPC, I ran into the error of the missing Intel license file necessary for training the SLING parser. Since PyDaisy eventually worked out, I didn't continue on pursuing SLING since the error remained after a few email exchange with the HPC support team.

#### Week 4 (6/17/19 - 6/21/19):
1. **Augmented PyDaisy to work with Berkeley FN1.7 database.**: Apply a model of n-gram window for PyDaisy to identify frames for a long sentence. Import Berkeley FN1.7 from `nltk` for PyDaisy to work with Berkeley FN1.7 library.
2. **Annotate the frames in NewsScape dataset and generate output in Red Hen Data Format.**: PyDaisy takes in the data file with closed captions and outputs the `.seg` (Red Hen data format) with a tag of `FRM_02`.
3. **Deployed PyDaisy in a Singularity Container.**: Successfully run the `sbatch` command with the PyDaisy Singularity container and necessary script files for frame identification on CWRU HPC clusters.

#### Week 5 (6/24/19 - 6/28/19) and Week 6 (7/1/19 - 7/5/19):
1. **OpenSesame**: I have completed and deployed the pipeline with OpenSesame in a Singularity Container. I have successfully test run the container on CWRU, and it can identify the frames and arguments of NewsScape dataset.
2. **Frame Embeddings**: I have read through 10 papers to understand the recent practices with creating frame embeddings and summarized their methodologies in a report. One of the most popular is BoW approach. I believe BERT and ELMo generation of frame embeddings will be better because these two models use attention models to create sentence embeddings, which better capture the semantic concepts of a sentence. </br>
There are some interesting observations:
    1. Only recent SEMEVAL papers on framenet verb and roles clustering tasks use BERT and ELMo to generate the context vectors that represent the sentence. Previous papers rely on Word2Vec and TF-IDF algorithms. Surprisingly, RNN, transformer, and attention models are not brought up in those older papers.
    2. It seems that there’s no consensus about how to best represent the semantic meaning yet. In fact, there’s no agreement on which types of embeddings are best for clustering. There’s no concrete discussion on the reasons why certain embeddings and clustering algorithms work.
    3. I didn't encounter any paper that touches upon expanding the number of frames in FrameNet.
    4. Only one paper that I come across uses data augmentation to increase the number of annotated sentences used to create embeddings. This is a big contrast to the computer vision field as data augmentation is commonly used to reduce the overfitting of the data. I believe this is important for Framenet as the available sentences for generating embeddings and training models are limited.
3. **SEMAFOR**: Since the `pyfn` library that makes argument-identification with SEMAFOR is not working, I have to read through the Java codes of SEMAFOR I ran into two problems. First, https://github.com/AlenUbuntu/semafor_Framenet_v1.7, which is the updated SEMAFOR suggested on RedHenLab is incomplete. One of the links that contains the pretrained SEMAFOR model is not working (https://utdallas.box.com/s/mgwbpje4stqyyzbl1eocg307pk0pqbc3). Second, I have been following the guide to retrain the model on my laptop for the past few days, and the training keeps running into the error. I realize that my data structure preparation step might not be working, which results in the serialization error.. Moreover, there’s no way for me to raise issues on the github page (because it is a fork version). My next step is to email the people in charge to directly ask for the models.
4. **BabelNet - Frame-to-Frame relations checking**: Successfully create a function for verify frame-to-frame relations of the suggested frame for unseen lexical units.
5. **BabelNet - Valence Pattern checking**: Successfully ran Valence API for verifying the valence patterns (frame elements) of a frame on my computer. The API retrieves the CoreSet of FEs for a given frame. Will work on containerizing it into Singularity because it requires a server and MongoDB database for retrieving the CoreSet of Frame Elements.

#### Week 7 (7/8/19 - 7/12/19):
1. Follow up on **SEMAFOR**. I have sent the email to the person-in-contact listed on the GitHub repo for requesting the pre-trained SEMAFOR model and the MaltParser. 
2. **Adapt Dependency trees to PyDaisy**: I have created the dependency tree using UDPipe API call. It is a modification of the n-gram by taking the root node and a fixed number of child nodes to parse in `parseInput`. It is deployed on Singularity container and I have test-run it on Red Hen Lab CWRU clusters.
3. **Extract unrecognized lemma in NewsScape sentences and turn it into lexical unit**: I found a library called `flair` whose POS-tagging works for all capitalized sentence.
4. **Generate the embeddings with different models**: I have successfully generated BERT embeddings for the lexical units using their exemplar sentences using the `flair` library. I am still generating their ELMO embeddings. 
5. **Create a function for matching the POS of unseen lexical units with the POS of the lexical units in the potential frame**
6. **Continue on adapting BabelNet and WordNet to this renewed approach.**: WordNet - Still proceed with what my proposal has suggested but instead of using DSSM model, use BERT and ELMO which factor in the context of the sentence. They are the SoTA word embeddings model right now (excluding the most recent xlNet). First, I will continue tackling the antonyms problem with the generated embeddings. At the same time,  I will focus on adding new lexical units to the current labels OR new labels. BabelNet - Unaffected.

#### Week 8 (7/15/19 - 7/19/19):
1. **Antonyms Detection**:
- I have generated BERT embeddings for all the synsets that come with examples in WordNet.
- I have trained the Decision Tree Classifier from sklearn with WordNet antonyms and non-antonyms pairs. There are 3322 antonyms pairs in total and 3322 self-generated non-antonym pairs. The antonyms and non-antonyms are split by a ratio of 0.33 into training and testing dataset. The accuracy in identifying the antonyms and non-antonyms in WordNet is 0.7423.
- I used the trained model to identify the potential antonyms in FrameNet. I pickled the list and uploaded it here. The model is very inaccurate in identifying the antonyms in FrameNet.

2. **Argument Labeling for Unseen Lexical Units**:
- As demonstrated a few days ago, I tried using Open-Sesame label the arguments of the sentence using the predicted frame of the unseen lexical units.
- I have read through @Tiago Torrent’s comment and I am thinking of a way out since the annotation comparison doesn’t work as intended.

3. **BabelNet**:
- The plan is to represent FrameNet frames with BabelNet synsets.
a) The lemma under FN frames (along with its POS) will be used to call BabelNet API to retrieve all the synsets.
b) Then, only retain synsets that generalize two or more lexical units under the particular frame.
c) Finally, call BabelNet API to retrieve all the lemmas and example sentences in the BabelNet synset to generate the BERT embeddings.
- I have successfully created all the necessary functions for the three steps. However, I encounter such error `Your key is not valid or the daily requests limit has been reached`. There’s daily requests limit of 1000 per day for the BabelNet API and I reached the limit.
- I have sent the request form for increasing the API calls limit on the official BabelNet website. The screenshot of the request form is attached below.

4. **SEMAFOR**: Sent a follow-up email to request for the pre-trained FN1.7 SEMAFOR model.

#### Week 9 (7/22/19 - 7/26/19):
1. **Antonym Detection**: I have created the containers to be tested on CWRU HPC. I have trained the models with dependency parsing using UDPipe and I am running it on FrameNet 1.7. I will put the result up and the documentation of running the containers once the model successfully parses LUs in FrameNet 1.7.

2. **BabelNet**: I have submitted the application and I haven’t received the dataset yet even though I received an email from the person-in-contact from BabelNet (I believe I have CC-ed Prof. Tiago in my reply to the email).

3. **Expanding FrameNet with Embeddings**: I have created the pipeline filter container for identifying lexical units from NewsScape that do not meet POS (using flair) and valence patterns of the LUs in the frame (using Valencer API). I will test the Singularity container on CWRU and upload the documentation here.

#### Week 10 (7/29/19 - 8/2/19):
1. **Documentation on Antonym Detection** (Successfully Deployed with Singularity on CWRU HPC): https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Antonym%20Detection.md

2. **Documentation on Expanding FrameNet with NewsScape and BERT** (Successfully Deployed with Singularity on CWRU HPC): https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Expanding%20FrameNet%20with%20NewsScape%20and%20Embedding.md

3. **Summary of Research Papers on Antonym Detection Models**: https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Background%20Research%20-%20Antonym%20Detection.pdf

4. **Multi-lingual Embeddings** - Compare all convenience of accessing the lexical units and their annotations (https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Summary%20-%20Multilingual%20FN.md) and working on generating the embeddings for Sweden-FN and Korean-FN.
Note:

5. I ran into deployment issues for most of the time during the week so I didn’t have enough time catering for the task of “Research papers on antonym detection models and implement multi-lingual embeddings for LUs.” The challenges and the bugs I met are documented in the documentation.

#### Week 11 (8/5/19 - 8/9/19):
1. **Clustering of Lexical Units that are Filtered out by POS and Core FEs**: I have applied Affinity Propagation clustering technique on the LU embeddings. Result photo can be seen [here](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/images/viz_clustered_LUs.png).

2. **Assign frames to clusters using multilingual FrameNet**: I have applied translation and KoreanFN to assign frames to the clusters.

#### Week 12 (8/12/16 - 8/16/19):
1. **Assign frames to clusters using multilingual FrameNet**: I have implemented BrazilFN to assign frames. However, I ran into the problem of Google Translation API Requests Limits, which means it is impossible to batch translate the lexical units and assigned frames with BrazilFN and KoreanFN.

2. I have **looked through 150 frames within FrameNet** and reading up on the FrameNet structure in “FrameNet II: Extended Theory and Practice”. I classify them into (1) frames where antonyms are grouped together, (2) antonymous frames due to the reversive pairs, (3) Relational opposites such as seller and buyer, (4), Erroneous Frames (Frames containing lexical units that are in relational opposites. Such erroneous frames are Guest_and_host where guest.n and host.n should be in separate frames according to “FrameNet II: Extended Theory and Practice”)

Therefore, it doesn’t seem worthwhile resolving inconsistency because it seems that FrameNet’s frames are supposed to contain antonyms such as dry and wet in the same frame, where the lexical units differ in their semantic types (positive or negative). There are, however, few frames which are inconsistent with the notion of relational opposites should be in separate frames.

#### Week 13 (8/19/16 - 8/23/19):
1. Deploy Singularity Containers for Clustering of Lexical Units and Multilingual frame assignment.
2. Ensure that previous singularity containers work properly (double-checking).
3. Finish up the blog posts about the third term week 9 - 12.
