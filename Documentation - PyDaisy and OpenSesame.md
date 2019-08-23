# PyDaisy and Open-Sesame
This documentation outlines the steps and the details of each step in annotating NewsScape data with PyDaisy and Open-Sesame. A short analysis and evaluation of the two libraries will be included at the end of each section.

## Prior to Annotation

#### Copying the NewsScape data library to be Annotated

Before annotating the NewsScape data library with the two annotating libraries, I used the Python script `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/get_newsscape.py` to copy the files. These duplicate files are annotated to avoid irreversible in-place changes to the original file during the development stage. The NewsScape data that is copied and will be processed are in `.seg` file format, which is the Red Hen Data Format. It is chosen because the file includes the metadata and the closed captions (which will be annotated).

The following code shows the command that copies the 2019/01/02 NewsScape data files into the folder `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/newsscape/`.

```bash
python3 get_newsscape.py --year 2019 --month 01 --day 02 --folder_path ./newsscape
```

## Annotation
A slurm script (`/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/frame_annot_tutorial.slurm`) is created to annotate the 2019/01/02 NewsScape data files, which is copied into the folder `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/newsscape/`.

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=3-00:00:00
#SBATCH --output=frame-annot-tutorial.stdout
#SBATCH --error=frame-annot-tutorial.err
#SBATCH --job-name="frame annotation tutorial"

module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
module load singularity
export SINGULARITY_BINDPATH="/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame:/mnt"

singularity exec production.sif python3 -u /mnt/annotate_dataset.py --path_to_folder /mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02 > ./frame-annot-tutorial-output.out
```

**Output**

`/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/frame-annot-tutorial-output.out` shows the current progress of annotating the NewsScape dataset. The following script is a sample output that shows that the annotation of `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/2019-01-02_0330_US_KNBC_Access.seg` is underway.

```bash
$ cat /home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/frame-annot-tutorial-output.out

# output
Annotating: /mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/2019-01-02_0330_US_KNBC_Access.seg
/mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/2019-01-02_0330_US_KNBC_Access.seg
/mnt/pyfn/experiments/xp_001/data/test.sentences
Initializing preprocessing...
Preprocessing setup:
  XP_DIR: /mnt/pyfn/scripts/../experiments/xp_001
  POS tagger: nlp4j
  Dependency parser: bmst
  Frame semantic parser: semafor
Initializing part-of-speech tagging...
POS tagging via NLP4J...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/test.sentences
Masking _ chars...
Done
Brownifying...
Done
Unmasking _ chars...
Done
Converting .sentences to .tsv format...
Done
POS tagging tsv file...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/test.sentences.tsv
Done
Converting .nlp4j to .conllx format...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/test.sentences.tsv.nlp4j
Done
Initializing part-of-speech tagging...
POS tagging via NLP4J...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/train.sentences
Masking _ chars...
Done
Brownifying...
Done
Unmasking _ chars...
Done
Converting .sentences to .tsv format...
Done
POS tagging tsv file...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/train.sentences.tsv
Done
Converting .nlp4j to .conllx format...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/train.sentences.tsv.nlp4j
Done
Initializing dependency parsing...
Dependency-parsing via BIST MST parser...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/test.sentences.conllx
Using external embedding: /mnt/pyfn/scripts/../resources/sskip.100.vectors
Initializing lstm mstparser:
Load external embedding. Vector dimensions 100
Finished predicting test. 2.9695649147 seconds.
Done
Initializing dependency parsing...
Dependency-parsing via BIST MST parser...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/train.sentences.conllx
Using external embedding: /mnt/pyfn/scripts/../resources/sskip.100.vectors
Initializing lstm mstparser:
Load external embedding. Vector dimensions 100
Finished predicting test. 0.000715970993042 seconds.
Done
Converting to .flattened format for the SEMAFOR parser...
Processing file: /mnt/pyfn/scripts/../experiments/xp_001/data/train.sentences.conllx
```

After the NewScape dataset is annotated, you will find the annotationg with `FRM_02` labels in the `.seg` NewsScape file. The following is the sample result. 

```
TOP|20190102033002|2019-01-02_0330_US_KNBC_Access
COL|Communication Studies Archive, UCLA
UID|b0c21298-0e3e-11e9-a614-089e01ba0326
SRC|UCLA Library
TTL|Access
PID|EP02839073.0332
CMT|Entertainment news
DUR|0:29:55.26
VID|640x352|1920x1080
LAN|ENG
LBT|2019-01-01 19:30 America/Los_Angeles
SEG_02|2019-01-04 01:45|Source_Program=RunTextStorySegmentation.jar|Source_Person=Rongda Zhu
SMT_01|2019-01-04 03:06|Source_Program=Pattern 2.6, Sentiment-02.py|Source_Person=Tom De Smedt, FFS|Codebook=polarity, subjectivity
SMT_02|2019-01-04 03:06|Source_Program=SentiWordNet 3.0, Sentiment-02.py|Source_Person=Andrea Esuli, FFS|Codebook=polarity, subjectivity
NER_03|2019-01-04 03:06|Source_Program=stanford-ner 3.4, NER-StanfordNLP-03.py|Source_Person=Jenny Rose Finkel, FFS|Codebook=Category=Entity
POS_02|2019-01-04 03:06|Source_Program=stanford-postagger 3.4, PartsOfSpeech-StanfordNLP-02.py|Source_Person=Kristina Toutanova, FFS|Codebook=Treebank II
FRM_01|2019-01-04 03:07|Source_Program=FrameNet 1.5, Semafor 3.0-alpha4, FrameNet-06.py|Source_Person=Charles Fillmore, Dipanjan Das, FFS|Codebook=Token|Position|Frame name|Semantic Role Labeling|Token|Position|Frame element
20190102033003.634|20190102033039.103|SEG_02|Type=Story|Score=1.0
20190102033003.634|20190102033008.539|CC1|THAT AMAZING, SPECTACULAR PERSON MADE ME FEEL LIKE I COULD DO THESE THINGS.
20190102033003.634|20190102033008.539|FRM_02|PERSON|4-5|People|SRL|PERSON|4-5|Descriptor|MADE|5-6|Origin
20190102033003.634|20190102033008.539|FRM_02|FEEL|7-8|Feeling|SRL|LIKE|8-9|Explanation
20190102033003.634|20190102033008.539|FRM_02|COULD|10-11|Possibility|SRL|COULD|10-11|Condition|DO THESE|11-12|Explanation
20190102033003.634|20190102033008.539|FRM_01|PERSON|4-5|People|SRL|PERSON|4-5|Person
20190102033003.634|20190102033008.539|FRM_01|FEEL|7-8|Sensation
20190102033003.634|20190102033008.539|FRM_01|COULD|10-11|Capability|SRL|DO|11-12|Event|SRL|I|9-10|Entity
20190102033003.634|20190102033008.539|FRM_01|DO|11-12|Intentionally_act|SRL|THESE THINGS|12-14|Act|SRL|I|9-10|Agent
20190102033003.634|20190102033008.539|POS_02|THAT/DT|AMAZING,/NN|SPECTACULAR/JJ|PERSON/NN|MADE/VBD|ME/PRP|FEEL/VB|LIKE/IN|I/PRP|COULD/MD|DO/VB|THESE/DT|THINGS./IN|
20190102033003.634|20190102033008.539|SMT_01|0.6|0.9|amazing|0.6|0.9|spectacular|0.6|0.9
20190102033003.634|20190102033008.539|SMT_02|AMAZING|0.0|0.0|SPECTACULAR|0.25|0.25|MADE|0.0|0.0|LIKE|-0.25|0.25|I|0.0|0.0
20190102033008.539|20190102033015.179|CC1|>> HAPPY NEW YEAR, YEARS, SO MANY POSSIBILITIES FOR OUR CELEBRITIES IN THE NEW YEAR.
...
```

Note: Currently, the frame identification of the OpenSesame library seems to outperform that of PyDaisy. The potential reasons could be that n-gram and dependency-tree models added to PyDaisy (see [My Modification and Analysis of PyDaisy](#my-modification-and-analysis-of-pydaisy)) to process long sentences from NewsScape jeopardize the accuracy of frame-identification. Therefore, the entire dataset is annotated with OpenSesame.

---
**Table of Content**
- [Implementation Details](implementation-details)
   - [Preprocessing the data](preprocessing-the-data)
   - [PyDaisy (Compatible with Berkeley FrameNet 1.7)](pydaisy-(compatible-with-berkeley-framenet-1.7))
   - [Open-Sesame](open-sesame)
   - [Future Direction](future-direction)
---
## Implementation Details
#### Preprocessing the data
Preprocessing is necessary for two reasons. First, we are only interested in annotating the closed captions in the `.seg` file so we have to extract it. Second, the closed captions are all in capital letters. Without processing, PyDaisy and OpenSesame cannot accurately annotate the sentence. 

In other words, there are two preprocessing step:

1. Extract the closed captions for annotation
2. Preprocess the closed captions text so they can be recognized and annotated by PyDaisy and OpenSesame.

I created the script `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/convert_newscape.py` that extracts the closed captions, which is run by the following command:

```bash
python3 convert_newscape.py --path_to_file <filename>
```

NLP4J and BMST are used to perform lemmatization, part-of-speech tagging and dependency parsing as their combination in the pre-processing pipeline yields the best result of frame and frame element identification (Kabbach, Ribeyre & Herbelot, 2018).

```bash
/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/pyfn/scripts/preprocess.sh -x 001 -t nlp4j -d bmst -p semafor
```

---

<div style="page-break-after: always;"></div>

### PyDaisy (Compatible with Berkeley FrameNet 1.7)

PyDaisy is the implementation of Disambiguation Algorithm for Inferring the Semantics of Y (Daisy) using Python for FrameNetBrasil. It can be accessed through the private GitHub repository - https://github.com/FrameNetBrasil/py_daisy.

This section is structured into two subsections. The first section explains how to annotate the NewsScape dataset using the deployed Singularity container on Red Hen's HPC clusters. The second section elaborates on what I have worked on and the analysis of PyDaisy.


#### Instructions: Running the Singularity Container for PyDaisy FrameNet Annotation

The folder `/home/zxy485/zxy485gallinahome/week1-4/final-pydaisy` contains every script needed for annotation with PyDaisy. In particular, the singularity container `production.sif` contains the Python libraries such as NLTK necessary for running the annotating script `frm_02.py`. 

The workflow is as shown by the following figure.

 ![image-20190703172954937](https://github.com/yongzx/GSoC-2019-FrameNet/blob/b50295ea4efeeec2b8d50e44a5fb653dc069e641/images/Documentation%201%20-%20PyDaisy%20Workflow.png)


The respective commands for annotating a single NewsScape file is as followed:

```bash
# extracting the closed captions from the NewsScape datafile <filename>
singularity exec production.sif python3 /mnt/convert_newscape.py --path_to_file /mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_0000_US_CNN_CNN_Special_Report.seg

# preprocessing the closed captions with NLP4J and BMST
singularity exec production.sif /mnt/pyfn/scripts/preprocess.sh -x 001 -t nlp4j -d bmst -p semafor

# Annotate with PyDaisy
singularity exec production.sif python3 /mnt/frm_02.py \
	--path_to_conllx /mnt/pyfn/experiments/xp_001/data \
	--path_to_file /mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_0000_US_CNN_CNN_Special_Report.seg \
	--path_to_log /mnt/errors/log-2019-01-01_0000_US_CNN_CNN_Special_Report.seg
```

I wrap the commands in a Python script `annotate_dataset.py`, which can loop through and annotate every file in the `newsscape` folder. The commands are shortened to one line:

```bash
singularity exec production.sif python3 /mnt/annotate_dataset.py --path_to_folder /mnt/newsscape
```

The sbatch script (`/home/zxy485/zxy485gallinahome/week1-4/final-pydaisy/frame_annot.slurm`) used to submit the annotation job for the NewsScape data files in `/home/zxy485/zxy485gallinahome/week1-4/final-pydaisy/newsscape` is as followed:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=3-00:00:00
#SBATCH --output=my.stdout
#SBATCH --error=my.err
#SBATCH --job-name="frame annotation"

module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
module load singularity
export SINGULARITY_BINDPATH="/home/zxy485/zxy485gallinahome/week1-4/final:/mnt"

singularity exec production.sif python3 -u /mnt/annotate_dataset.py --path_to_folder /mnt/newsscape > ./output.out
```

#### My Modification and Analysis of PyDaisy

The frame identification mechanism of PyDaisy is by first obtaining all the potential frames (including super-frames and sub-frames) represented by all frame-target words in the sentence, and subsequently finding the best cluster/combinations of frames. The ranking of clusters are based on the weighted relations between frames.

Instead of importing Berkeley FrameNet 1.7 that comes in XML format into a SQL database in order to use PyDaisy algorithms, I used NLTK Python library that comes with FN1.7 dataset to extract all the potential frames including the super-frames and sub-frames of the target word. 

The major downside of adapting PyDaisy to Berkeley FrameNet 1.7 (FN1.7) is that the domain of FN1.7 is wider than FrameNet Brazil. This means that during the step of finding the best cluster of frames, there are exponentially more number of clusters due to the factorial growth rate of the combination function. To avoid the error of `MemoryError`, I have to perform two mitigations.

The first is through n-gram or dependency-tree modeling of frame identification. For the n-gram, the algorithm now stores the rank values of the best cluster of frames for a window of fixed number of words. After iterating through the sentence, the algorithm picks the best frames for the set of respective frame-target words. For the dependency tree model, the algorithm takes in each word token (which is a node in the tree) and convert it into a n-gram window where the word token is in the middle and its neighboring words are its left and right children nodes. Since frame identification involves inspecting the semantic meaning represented by the entire sentence, I speculate that this reductionist approach could jeopardize the accuracy of the frame identification.

The second mitigation step is by ignoring the stopwords in the sentence. That is, even though there are many frames available for stopwords such as `a`, `the`, etc., I avoid assigning frames to them because the sheer large number of frames associated with these frame-target stopwords increases the size of the clusters, which increases the running time of the algorithm. 

It should also be noted that FrameNet annotation involves frame and semantic role labeling, and PyDaisy only labels the frames. Because of the long running time due to the large number of clusters of frames and the potential inaccuracy that comes with the n-gram model, I turned to look at OpenSesame whose open-source codes include both frame and semantic role labeling.

---

<div style="page-break-after: always;"></div>

### Open-Sesame

Open-sesame is a parser that detects FrameNet frames and their frame-elements (arguments) from sentences using Segmental RNNs. This section is divided into two subsections. The first part elaborates on how to annotate the NewsScape dataset with OpenSesame, and the second part discusses the development of the annotating script `generate_RHL_format.py` and the challenges associated with the deployment. 


#### Instructions: Running the Singularity Container for OpenSesame FrameNet Annotation

The folder `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame` contains every script needed for annotation with OpenSesame. In particular, the singularity container `production.sif` contains the Python libraries such as NLTK necessary for running the annotating script `generate_RHL_format.py`.

![image-20190704012415651](https://github.com/yongzx/GSoC-2019-FrameNet/blob/9c6f19dd77df5793dbc6c85fedabc95b626b4afa/images/Documentation%201%20-%20OpenSesame%20Workflow.png)

The respective commands for annotating a single NewsScape file is as followed:

```bash
# extracting the closed captions from the NewsScape datafile <filename>
singularity exec production.sif python3 /mnt/convert_newscape.py --path_to_file /mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_0000_US_CNN_CNN_Special_Report.seg

# preprocessing the closed captions with NLP4J and BMST
singularity exec production.sif /mnt/pyfn/scripts/preprocess.sh -x 001 -t nlp4j -d bmst -p semafor

# Annotate with OpenSesame
singularity exec production.sif python3 /mnt/generate_RHL_format.py \
	--path_to_lemmatized_file /mnt/pyfn/experiments/xp_001/data/test.sentences.conllx \
	--path_to_augmented_lemmatized_file /mnt/pyfn/experiments/xp_001/data/test.sentences.processed.conllx \
	--path_to_open_sesame_input_file /mnt/sentences.txt \
	--path_to_open_sesame_output_file /mnt/logs/fn1.7-pretrained-argid/predicted-args.conll \
	--path_to_seg_file /mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_0000_US_CNN_CNN_Special_Report.seg
```

I wrap the commands in a Python script `annotate_dataset.py`, which can loop through and annotate every file in the `newsscape` folder. The commands are shortened to one line:

```bash
singularity exec production.sif python3 /mnt/annotate_dataset.py --path_to_folder /mnt/newsscape
```

The sbatch script (`/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/frame_annot.slurm`) used to submit the annotation job for the NewsScape data files in `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/newsscape` is as followed:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=3-00:00:00
#SBATCH --output=my.stdout
#SBATCH --error=my.err
#SBATCH --job-name="frame annotation"

module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
module load singularity
export SINGULARITY_BINDPATH="/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame:/mnt"

singularity exec production.sif python3 -u /mnt/annotate_dataset.py --path_to_folder /mnt/newsscape > ./output.out
```



#### Development of `generate_RHL_format.py`

The `generate_RHL_format.py` is extended from the [three commands](https://github.com/swabhs/open-sesame#prediction-on-unannotated-data) that can generate predictions of frames and frame elements on unannotated sentences. 

All the unannotated sentences are written in the text file `sentences.txt` separated by line break. They will passed through `sesame/targetid.py` ,  `sesame/frameid.py` and  `sesame/argid.py`  in order to generate a CONLL file that contains the predicted frames and frame elements. 

Therefore, `generate_RHL_format.py` accomplished the three following tasks:

1. Copy the lemmatized sentences from `pyfn/experiments/xp_001/data/test.sentences.conllx ` to `sentences.txt`
2. Run the [three commands](https://github.com/swabhs/open-sesame#prediction-on-unannotated-data) to generate the predictions of frames and frame elements.
3. Convert the CONLL file that contains the predicted frames and frame elements into Red Hen Data Format.

Note that `generate_RHL_format.py` does not process file by file; instead, it processes sentence by sentence. In other words, it only copies one sentence into `sentences.txt`, and after predicting the frames and frame elements, the script adds the frames and frame elements of the sentence into the `.seg` file.


#### Challenges with Deploying Open-Sesame on HPC Cluster

The main challenge comes with my unfamiliarity with modifying the environment and file relations when running the Singularity container. More specifically, the OpenSesame annotation step, which is based on the [three commands](https://github.com/swabhs/open-sesame#prediction-on-unannotated-data) that generate predictions of frames and frame elements, runs library module as script. 

The three commands for generating predicted frames and arguments are as followed (copied from the OpenSESAME README.md). 

```bash
# before modification
$ python -m sesame.targetid --mode predict \
                            --model_name fn1.7-pretrained-targetid \
                            --raw_input sentences.txt
$ python -m sesame.frameid --mode predict \
                           --model_name fn1.7-pretrained-frameid \
                           --raw_input logs/fn1.7-pretrained-targetid/predicted-targets.conll
$ python -m sesame.argid --mode predict \
                         --model_name fn1.7-pretrained-argid \
                         --raw_input logs/fn1.7-pretrained-frameid/predicted-frames.conll
```

However, when the scripts ( `targetid.py`, `frameid.py`, `argid.py`) are mounted on the Singularity container, the exact commands would not work because of the path dependency. I had to change run the scripts just as scripts and not as packages. 

```bash
# after modification 
# scripts mounted on /mnt/sesame folder of Singularity container
$ python /mnt/sesame/targetid.py --mode predict \
                            --model_name fn1.7-pretrained-targetid \
                            --raw_input /mnt/sentences.txt
$ python /mnt/sesame/frameid.py --mode predict \
                           --model_name fn1.7-pretrained-frameid \
                           --raw_input /mnt/logs/logs/fn1.7-pretrained-targetid/predicted-targets.conll
$ python /mnt/sesame/argid.py --mode predict \
                         --model_name fn1.7-pretrained-argid \
                         --raw_input /mnt/logs/logs/fn1.7-pretrained-frameid/predicted-frames.conll
```

In addition, modifications are made to the following files to ensure that the path dependencies for reading and writing files remain valid after the files are mounted on the `/mnt/` folder of the Singularity container.

- `sesame/argid.py` :

  ```python
  # line 34
  model_dir = "/mnt/logs/{}/".format(options.model_name)
  ```

- `sesame/frameid.py`

  ```python
  # line 23
  model_dir = "/mnt/logs/{}/".format(options.model_name)
  ```

- `sesame/targetid.py`

  ```python
  # line 22
  model_dir = "/mnt/logs/{}/".format(options.model_name)
  ```

- `sesame/globalconfig.py`

  ```python
  # line 5
  config_json = open("/mnt/configurations/global_config.json", "r")
  ```

- `configurations/global_config.json`

  ```json
  {
      "version": 1.7,
      "data_directory": "/mnt/data/",
      "embeddings_file": "/mnt/data/glove.6B.100d.txt",
      "debug_mode": false
  }
  ```


It is important to note this changes down because OpenSESAME may be updated and these changes have to be customly introduced after pulling the repository from GitHub. 

---

<div style="page-break-after: always;"></div>

## Future Direction

After checking in with Prof. Tiago, who is my mentor, it seems that the argument identification of OpenSESAME is not precise (a lot of false positives). Therefore, in this week 6 (and possibly 7), I am looking into SEMAFOR and using its argument identification part to complement OpenSESAME's weakness. 
