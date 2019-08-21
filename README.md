# GSoC 2019: Annotating NewsScape with FrameNet 1.7 and Expanding FrameNet with BabelNet and DSSM

### Introduction
This project sets out to achieve two goals. The first objective is to update the annotation system for Red Hen’s NewsScape dataset to FrameNet 1.7 using Open-Sesame and Semafor parsers. The second objective is to expand the lexical units, frames and frame-to-frame relations in FrameNet 1.7 through a knowledge-driven approach and a distributional semantics approach. The knowledge-driven approach uses BabelNet to induce the frames of unrecognized lexical units in the tagged NewsScape dataset. The latter distributional semantics approach uses Deep Structured Semantic Models (DSSM) to create word embeddings of lexical units (LUs) to resolve the inconsistency in FrameNet hierarchy, tag LUs with their missing frames, and locate new frames using SemCor corpus. If time permits, DSSM is used to expand the frame-to-frame relations with Entity and Event frames using ACE 2005 Entities and Events dataset.

### Goals
*Week 1 - 4*
- [X] Annotate the NewsScape dataset with FrameNet 1.7 using OpenSesame parser
- ~~Annotate the NewsScape dataset with Semafor compatible with FrameNet 1.7~~
  - This has not worked out well because the link to the pretrained model on Semafor page was removed and I have attempted the retraining process but to no avail. By today (7/14/19), I have reached out to the person-of-contact for Semafor-FN 1.7 and I have CC-ed Prof. Torrent along the email.
- [X] Deploy the OpenSesame pipeline on CWRU HPC
- [X] (New) Deploy the PyDaisy (with n-gram model and dependency tree model) pipeline on CWRU HPC.

*Week 5 - 8*
- [X] Create the representation of FrameNet 1.7 with BabelNet's synsets
- [X] Induce frames for unrecognized lexical units, LUs, from NewsScape dataset using BERT embeddings and cosine similarity
- [X] Create filters that remove the incorrectly induced frames based on POS and core FEs.
- [X] Deploy the pipeline that induces and filters frame for unrecognized LUs on CWRU HPC
- [X] (Optional) Suggest new frames that better capture the senses of the unrecognized lexical units using multilingual framenets such as KoreanFN and BrazilFN

*Week 9 - 12*
- [X] Create word semantic embeddings of LUs and distributional profile (frame clusters) of
FrameNet using ELMO and BERT.
  - The decision to change from DSSM to ELMO and BERT comes after reading the related papers on frame embeddings, where I learned that ELMO and BERT are primarily used for SemVal-2019 tasks of inducing frames. The summary of each paper is in this [report](https://github.com/yongzx/GSoC-2019-FrameNet/blob/35793a73fda4ad456beab9bf467d8156fcf46e81/Background%20Research%20-%20Frame%20Embeddings.pdf).
- [X] Create word semantic embeddings and semantic clusters of words in WordNet
- [X] Deploy the pipeline that uses BERT embeddings and WordNet sysnsets to identify inconsistent antonyms in FrameNet structure on CWRU HPC.
- [X] Create a pipeline that clusters and visualizes new lexical units that do not pass through the POS and coreFEs filters.
- [X] Deploy a pipeline that assign frames from multilingual framenets such as KoreanFN and BrazilFN to the clustered lexical units.

### Libraries / Tools / APIs
- `pyfn`
- `flair`
- `translate`
- PyDaisy
- OpenSesame
- https://github.com/AlenUbuntu/semafor
- BabelNet HTTP API
- NLTK, GenSim, SpaCy (and other NLP libraries)
- ~~Tensorflow for Deep Semantic Structured Model (https://liaha.github.io/models/2016/06/21/dssm-on-tensorflow.html)~~ (Replaced by BERT Model)
- Potentially other machine learning libraries such as scikit-learn

### Dataset
- ACE 2005 Entities dataset – Linguistic Data Consortium. (2005). English Entities V5.6.6 Annotation
[Data file]. Retrieved from
https://www.ldc.upenn.edu/collaborations/past-projects/ace/annotation-tasks-and-specifications
- ACE 2005 Events dataset – Linguistic Data Consortium. (2005). English Events V5.4.3 Annotation [Data
file]. Retrieved from
https://www.ldc.upenn.edu/collaborations/past-projects/ace/annotation-tasks-and-specifications
- BabelNet 4.0 – Roberto Navigli. (2014). BabelNet 4.0 [Data file]. Retrieved from http://babelnet.org
- English Wikipedia (March 2019 dump) – Wikimedia. (2019). Enwiki Dump Progress on 20190320 [Data
File]. Retrieved from https://dumps.wikimedia.org/enwiki/20190301/
- FrameNet 1.7 – Baker, C. F., Fillmore, C. J., & Lowe, J. B. (1998). FrameNet 1.7 [Data File]. Retrieved
from https://framenet.icsi.berkeley.edu/fndrupal/
- NewsScape (Closed captioning extracted from ATSC mpeg2 transport stream)– UCLA Library. (2013).
UCLA NewsScape Archive [Data File]. Retrieved from https://tvnews.sscnet.ucla.edu/public/
- SemCor 3.0 text annotated with WordNet – Princeton University. (2008) SemCor Corpus [Data File].
Retrieved from http://web.eecs.umich.edu/~mihalcea/downloads.html#semcor
- WordNet 3.0 – Princeton University. (2010). WordNet [Data File]. Retrieved from
https://wordnet.princeton.edu/download/current-version
