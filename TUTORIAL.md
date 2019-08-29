# Tutorials 

**Target Audience**:
- Red Hen Mentors
- Newcomers who take over this project

**Important Notes for Red Hen Mentors**:
1. Do not delete files or folders because some tasks are sequential and may take more than a week for completion. Therefore, if the essential files needed for the next step are deleted, you will have to wait for the completion of tasks before moving to the next step. It might take more than a week for the full verification of the pipeline. 
The best way to overcome this issue is to `cat [filename].out` to verify the progress. 
2. For GSoC evaluation of my deployment, you may have to `scancel` the slurm task in the queue if necessary to release the compute nodes resources, especially when you `squeue` and find that the slurm task has not started running.

---

### NewsScape Annotation

**Steps**:
1. All resources are in `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/` so `cd` into the folder.
```
$ cd /home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/
```

2. Copy the 2019/01/02 NewsScape data files into the folder `/home/zxy485/zxy485gallinahome/week1-4/final-open-sesame/newsscape/`.
```
$ module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
$ python3 get_newsscape.py --year 2019 --month 01 --day 02 --folder_path ./newsscape
```

3. Submit the slurm script to annotate 2019/01/02 NewsScape data files. This process takes more than a week to complete.
```
$ sbatch frame_annot_tutorial.slurm
```

**Verification for Deployment Success**

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
## Antonym Detection

**Steps**:
1. All the necessary files reside in the folder `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection`.
```
$ cd /home/zxy485/zxy485gallinahome/week9-12/antonym-detection
```

2. Submit the slurm script to find the antonymous lexical units in the first 250 frames. (There are five slurm scripts `task{1/2/3/4/5}.slurm` to generate all the antonymous lexical units within the same frame. I only use `task1.slurm` as the example of the tutorial.)
```
$ sbatch task1.slurm
```

**Verification for Deployment Success**

- Check the progress output. `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/output1.out` shows the progress of identifying antonymous lexical units within the first 250 frames.
```
$ cat /home/zxy485/zxy485gallinahome/week9-12/antonym-detection/output1.out
[Identifying antonyms in FrameNet] starts.
Abandonment
('abandon.v', 'leave.v', 14839, 14841)
('abandon.v', 'forget.v', 14839, 15317)
('leave.v', 'forget.v', 14841, 15317)
Abounding_with
('crowded.a', 'surfaced.a', 4771, 7001)
('crowded.a', 'overcrowded.a', 4771, 7004)
('crowded.a', 'full.a', 4771, 7010)
('crowded.a', 'paved.a', 4771, 7018)
...
[..., ('unscrew.v', 'tie.v', 4557, 4757), ('seal.v', 'fasten.v', 4625, 4677), ('seal.v', 'tie.v', 4625, 4757), ('fasten.v', 'unfasten.v', 4677, 4711), ('fasten.v', 'tie.v', 4677, 4757), ('unfasten.v', 'tie.v', 4711, 4757)]
[Identifying antonyms in FrameNet] takes 2324.13 minutes.
```

- Once the progress is completed, check the output pickled file `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_1.p`, which contains a list of pairs of antonymous lexical units within the same frame. They are in the format of `(lexical unit 1, lexical unit 2, id of lexical unit 1, id of lexical unit 2)`. 
```
$ module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
$ python3
>>> import pickle
>>> pickle.load(open('/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_1.p', 'rb'))

..., ('button.v', 'fasten.v', 4544, 4677), ('button.v', 'unfasten.v', 4544, 4711), ('button.v', 'tie.v', 4544, 4757), ('open.v', 'buckle.v', 4545, 4547), ('open.v', 'cap.v', 4545, 4550), ...
```

---

## Expanding FrameNet with NewsScape and Embedding

**Steps**:
1. All the necessary files reside in the folder `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs`.
```
$ cd /home/zxy485/zxy485gallinahome/week9-12/unseen_LUs
```

2. Copy the NewsScape .seg file or the folder containing all the .seg files into any folder within /home/zxy485/zxy485gallinahome/week9-12/unseen_LUs. These files are used to expand FrameNet. In the following commands, I demonstrated copying the file `/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_2300_US_WEWS_News_5_at_6pm.seg` into the folder `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data` and all the .seg files from `/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/` into the folder `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102`.
```
$ cp /mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_2300_US_WEWS_News_5_at_6pm.seg /home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data

$ cp /mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-02/*.seg /home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102
```

3. Submit the slurm script (`/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/task-newsscape.slurm`) to identify new lexical units from NewsScape dataset and create BERT embeddings for them.
```
$ sbatch task-newsscape.slurm
```

4. Submit the slurm script (`/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/task-cluster-LUs-AP.slurm`) to generate the t-SNE visualization and the clustering of the lexical units which are filtered out by the POS and CoreFEs filters.
```
$ sbatch task-cluster-LUs-AP.slurm
```

5. Submit the slurm script (`/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/task-multilingual-frame-assignment.slurm`) to assign frames to the clusters of non-compliant lexical units.
```
$ sbatch task-multilingual-frame-assignment.slurm
```

**IMPORTANT NOTE**: Supposedly step 5 depends on output files from step 4, and step 4 depends on the output files from step 3, but the entire linear process would take more than a week for completion. Therefore, step 3, 4, and 5 can be run in parallel using the output files which have been generated in the past.

**Outputs**

*Step 3*
- Check the progress output
```
$ cat /home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102/output.out

[Unseen LUs_Closest Match] starts.
Sentence: MURDER SCENE.
2019-08-25 23:39:14,948 loading file /home/zxy485/.flair/models/en-pos-ontonotes-v0.2.pt
Sentence: PLUS, WARNINGS GOING UNHEARD AND LIVES LOST.
2019-08-25 23:39:17,010 loading file /home/zxy485/.flair/models/en-pos-ontonotes-v0.2.pt
Sentence: A COUPLE KILLED.
2019-08-25 23:39:18,151 loading file /home/zxy485/.flair/models/en-pos-ontonotes-v0.2.pt
Sentence: ANOTHER FAMILY IN THE HOSPITAL AT THE HANDS OF ACCUSED DRUNK DRIVERS.
2019-08-25 23:39:18,992 loading file /home/zxy485/.flair/models/en-pos-ontonotes-v0.2.pt
Sentence: PASSING THE TORCH FROM 2018 TO 2019.
...

[Match Valence (Core FEs)] starts.
>>>>>>>>>>>>>>>> Lexical Unit: torch.n
>>>>>>>>>>>>>>>> Lexical Unit: ohio.n
>>>>>>>>>>>>>>>> Lexical Unit: neighborhood.n
>>>>>>>>>>>>>>>> Lexical Unit: clock.n
>>>>>>>>>>>>>>>> Lexical Unit: listing.n
...
```

After the process is completed, there would be three pickled files.
- `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/matched_coreFEs_lus-2019-01-01_2300_US_WEWS_News_5_at_6pm.seg.p` stores the new lexical units which are extracted from `/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_2300_US_WEWS_News_5_at_6pm.seg` and which do not exist in Berkeley FrameNet 1.7.
```
$ module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
$ python3
>>> import pickle
>>> pickle.load(open('/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/matched_coreFEs_lus-2019-01-01_2300_US_WEWS_News_5_at_6pm.seg.p', 'rb'))

{..., 'season.n': 16012, 'kit.n': 4221, 'insurance.n': 16121, 'penn.n': 16121, 'trebek.n': 4305, 'quarterback.n': 16012, 'washington.n': 10585, ...}
```

- `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/unmatched_pos_lus-2019-01-01_2300_US_WEWS_News_5_at_6pm.seg.p` stores new lexical units which are extracted from `/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_2300_US_WEWS_News_5_at_6pm.seg` and do not pass the POS filter.
```
$ module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
$ python3
>>> import pickle
>>> pickle.load(open('/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/unmatched_pos_lus-2019-01-01_2300_US_WEWS_News_5_at_6pm.seg.p', 'rb'))

{'crying.n': tensor([-0.3601,  0.9405,  0.4090,  ...,  0.7458,  0.1140, -0.6848]), 'nothing.n': tensor([ 0.0093, -0.0860, -0.0291,  ..., -0.1037, -0.0613, -0.6192]), 'dark.n': tensor([ 0.2622,  0.5236,  0.6394,  ...,  0.3478,  1.0145, -0.3825]), 'idling.n': tensor([-0.0031, -0.2561,  0.5420,  ..., -1.0558, -0.0892, -0.3096]), 'carter.n': tensor([-0.2083, -0.1723, -0.4687,  ..., -0.1857,  0.1192, -1.0462]), 'indefendant.n': tensor([ 0.1130, -0.0869,  0.1047,  ..., -0.5166, -0.1262, -0.1070]), 'context.n': tensor([-0.5003,  0.2394, -0.1679,  ...,  0.4758, -0.0024,  0.6117]), 'seed.n': tensor([ 0.4037, -0.0945, -0.2414,  ...,  0.0398, -0.2197, -0.2701]), 'qualifying.n': tensor([ 0.2838,  0.2489,  0.4820,  ..., -0.4504, -0.2197, -0.4507]), 'warm-up.n': tensor([-0.7397, -0.7589,  0.2845,  ...,  0.0975,  0.2958, -0.0367])}
```

- `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/unmatched_coreFEs_lus-2019-01-01_2300_US_WEWS_News_5_at_6pm.seg.p` stores new lexical units which are extracted from `/mnt/rds/redhen/gallina/tv/2019/2019-01/2019-01-01/2019-01-01_2300_US_WEWS_News_5_at_6pm.seg` and do not pass the Core Frame Elements filter.
```
$ module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
$ python3
>>> import pickle
>>> pickle.load(open('/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/unmatched_coreFEs_lus-2019-01-01_2300_US_WEWS_News_5_at_6pm.seg.p', 'rb'))

{ ..., 'lifetime.n': tensor([ 1.6898, -0.5895,  1.2588,  ...,  0.0404,  0.3949, -0.8979]), 'guarantee.n': tensor([ 0.3999, -0.6460,  0.2950,  ..., -0.3751,  0.2995, -0.0348]), 'coaching.n': tensor([-0.1619,  0.1011, -0.0326,  ..., -1.0178, -0.2577,  1.1803]), ...}
```


*Step 4*
- Check the progress output
```
$ cat /home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102/clutser_LUs_AP_output.out

[Load LUs and their tensors] starts.
[Load LUs and their tensors] takes 0.01 minutes.
[Visualize LUs' vectors] starts.
[Visualize LUs' vectors] takes 5.27 minutes.
[Cluster with Affinity Propagation] starts.
[Cluster with Affinity Propagation] takes 0.07 minutes.
[Visualize Clustered LUs' vectors] starts.
[Visualize Clustered LUs' vectors] takes 5.26 minutes.
```

- After the process has completed, `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102/lu_cluster_affinity_propagation.pkl` stores a tuple of (unmatched_LUs_to_tensors, X, LUs, cluster_centers_indices, labels).
```
$ module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
$ python3
>>> import pickle
>>> unmatched_LUs_to_tensors, X, LUs, cluster_centers_indices, labels = pickle.load(open("/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102/lu_cluster_affinity_propagation.pkl", 'rb'))

>>> unmatched_LUs_to_tensors
'sprinkler.n': tensor([-0.0639, -0.7937,  0.5725,  ..., -0.0290,  0.6760,  0.3853]), 'curb.n': tensor([ 0.6414, -0.0556, -0.1243,  ...,  0.0016, -0.5924, -0.8106]), 'lingering.n': tensor([ 0.9284, -0.2239,  0.4346,  ...,  0.8315, -0.3735,  0.2848]), 'motor.n': tensor([ 0.4993, -0.1982,  0.7537,  ...,  0.1615, -0.2226, -0.6524])

>>> X
[[ 0.8884867  -0.10377462  0.9043543  ... -0.62185156  0.26520827
  -0.01701835]
 [ 0.5532113  -0.7138817  -0.3777894  ... -0.2673508  -0.35105324
   0.5249945 ]
   ...
 [ -0.48098472  0.02973192 -0.16719978 ...  0.19730632 -0.05893578
   -0.25339624]]
   
>>> LUs
['trump.n' 'romney.n' 'rhino.n' ... 'lingering.n' 'motor.n' 'kenneth.n']

>>> cluster_centers_indices
[  11   13   25   59   60   67   69   71   82   85   92   95  102  124
  142  150  159  172  177  189  192  196  198  207  210  223  225  233
  249  269  272  274  294  303  307  309  338  339  343  347  364  368
  393  398  425  432  465  470  475  476  499  503  528  537  544  556
  598  607  617  638  645  717  722  723  744  774  778  816  919  975
 1016 1051 1079 1104 1139 1169 1223 1257 1291 1296 1319 1403 1417 1467
 1487 1529 1549 1608 1756 1809 1879]
 
>>> labels
[54 54 77 ... 32  3 56]
```

- `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/viz_LU_vectors.png` is the t-SNE Visualization of lexical units.

![viz_LU_vectors](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/images/tutorial_viz_LU_vectors.png)

- `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/viz_clustered_LUs.png` is the t-SNE Visualization of lexical units which are clustered.

![viz_LU_vectors](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/images/tutorial_viz_clustered_LUs.png)


*Step 5*
- Check the progress output
```
$ cat /home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102/multilingual_frame_assignment_output.out

### Korean FrameNet ###
	# contact: hahmyg@kaist, hahmyg@gmail.com #

/mnt/koreanframenet/resource/1.1/KFN_lus.json
[Load LUs and their tensors] starts.
[Load LUs and their tensors] takes 0.01 minutes.
[Load clusters_to_LU_tensor_tuples] starts.
[Load clusters_to_LU_tensor_tuples] takes 0.00 minutes.
[Assign Frames (KoreanFN)] starts.
Process Cluster Index: 54
Process Cluster Index: 77
Process Cluster Index: 23
...
API request limit reached. sleep for 24hours. Sleep Starts..
...
[Assign Frames (BrasilFN)] starts.
Process Cluster Index: 53
president trump will have an opportunity to frame this debate beforehand when he invites members of his cabinet at 12:00 . : O presidente Trump terá a oportunidade de estruturar esse debate de antemão quando ele convidar membros de seu gabinete às 12:00.
Process Cluster Index: 79
a toddler rushed to the hospital after falling into a rhino enclosure at a florida zoo . : uma criança correu para o hospital depois de cair em um recinto de rinoceronte em um zoológico da Flórida.
...
API request limit reached. sleep for 24hours. Sleep Starts...
...
```

- `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102/clusters_to_multilingual_potential_frames_counters.p` stores a dictionary that maps cluster indexes to a collections.Counter dictionary of potential frames retrieved from multilingual framenets. The tuple contains the frame name, the name of the multilingual FrameNet, and the frame's id in its FrameNet database.
**Note**: This stored output currently is incomplete because I only resolved the issue with API requests limit on week 13 (last week of GSoC), so there's not enough time for full output generation.
```
$ module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
$ python3
>>> import pickle
>>> pickle.load(open('/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/data/0102/clusters_to_multilingual_potential_frames_counters.p', 'rb'))

{53: Counter(), 79: Counter(), 22: Counter({('People_by_age', 'kofn', 5159): 1, ('frm_people_by_age', 'brfn', 3677): 1})}
```
