# Tutorials 

**Target Audience**:
- Red Hen Mentors
- Newcomers who take over this project

**Important Notes for Red Hen Mentors**:
1. Do not delete files or folders because some tasks are sequential and may take more than a week for completion. Therefore, if the important files which are needed for the next step are deleted, you will have to wait for the task to be completed before moving to the next step.
The best way to overcome this issue is to `cat [filename].out` to verify the progress. 
2. For GSoC evaluation of my deployment, you may have to `scancel` the slurm task in the queue if necessary to release the compute nodes resources, especially when you `squeue` and find that the slurm task has not started running.

---

### NewsScape Annotation

**Steps**:
1. All resources are in `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/` so `cd` into the folder.
```
cd /home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/
```

2. Copy the 2019/01/02 NewsScape data files into the folder `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/newsscape/`.
```
module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
python3 get_newsscape.py --year 2019 --month 01 --day 02 --folder_path ./newsscape
```

3. Submit the slurm script to annotate 2019/01/02 NewsScape data files. This process takes more than a week to complete.
```
sbatch frame_annot_tutorial.slurm
```

**Verification for Success**

- Check the progress output. You should at least see `Annotating: {newsscape filename}` which means the annotation is underway. You will also see `Processed: {newsscape filename}` after the file is processed after 3 - 4 days.
```
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
...
Processed: /mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/2019-01-02_2200_US_FOX-News_The_Five.seg
Annotating: /mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/2019-01-02_2300_US_MSNBC_The_Beat_With_Ari_Melber.seg
/mnt/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/2019-01-02_2300_US_MSNBC_The_Beat_With_Ari_Melber.seg
/mnt/pyfn/experiments/xp_001/data/test.sentences
Initializing preprocessing...
Preprocessing setup:
  XP_DIR: /mnt/pyfn/scripts/../experiments/xp_001
  POS tagger: nlp4j
  Dependency parser: bmst
  Frame semantic parser: semafor
Initializing part-of-speech tagging...
POS tagging via NLP4J...
...
```

- See the FRM_02 labels in the processed .seg NewsScape file. The following is the sample result of `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/newsscape/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/2019-01-02_2200_US_FOX-News_The_Five.seg`.
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

---
