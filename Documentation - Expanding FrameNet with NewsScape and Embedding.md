# Documentation - Expanding FrameNet with NewsScape and Embedding

All the necessary files reside in the folder `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs`. 

The `sbatch` script used to identify new lexical units from NewsScape dataset and create BERT embeddings for them is as followed:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=7-00:00:00
#SBATCH --output=my.stdout
#SBATCH --error=my.err
#SBATCH --job-name="expanding-fn"

module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
module load singularity
export SINGULARITY_BINDPATH="/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs:/mnt"

# singularity exec production.sif python3 -u /mnt/create_embeddings.py --folder='/mnt/data' --seg_file='2019-01-01_2300_US_WEWS_News_5_at_6pm.seg' > './data/output.out'

singularity exec production.sif python3 -u /mnt/create_embeddings.py --folder='/mnt/data/0102'
```

The argument `--folder` specifies the folder of the NewsScape seg file to be processed and where all the output files will be stored. Remember that it must use SINGULARITY_BINDPATH.

The argument `--seg_file` specifies the file name of the NewsScape seg file to be processed. This is optional – if the argument is not specified, the script will identify all new lexical units and generate BERT embeddings for all the NewsScape seg files in the folder specified in `--folder` argument.

The final outputs are:

- Filtered new lexical units:  `{folder}/matched_coreFEs_lus-{seg_file}.p`
- New lexical units that do not pass the POS filter: `{folder}/unmatched_pos_lus-{seg_file}.p`
- New lexical units that do not pass the Core FEs filter: `{folder}/unmatched_coreFEs_lus-{seg_file}.p`

---

## How It Works

### 1. Generate BERT embeddings for Lexical Units

Lexical units are the combination of lemmas and their part-of-speech tags. For example, “run.v”, “long.v”, “long.adj”, etc. They are words that evoke a semantic frame (i.e., a description of a type of event, relation, or entity and the participants in it.) from FrameNet 1.7.

The BERT embedding of a lexical unit is obtained by averaging the BERT embeddings of the lexical unit appearing in the annotated sentences in FrameNet. If there’s no actual text in FrameNet that features the lexical unit, which means that there’s no sentence examples of the lexical unit, its embedding will be a zero tensor.

### 2. Assigning New Lexical Units to Corresponding Frames

New lexical units that do not exist in FrameNet 1.7 are obtained from UCLA NewsScape dataset. The words from NewsScape dataset are lemmatized and tagged with their respective POS before checking whether they exist in FrameNet.

The embedding of the new lexical unit is compared with the embeddings of existing lexical units in FrameNet. The new lexical unit is assigned to the frame whose lexical unit is the closest to the new lexical unit with respect to cosine similarity.

### 3. Filtering Potentially New Lexical Units

These unseen lexical units which are assigned to a frame are filtered for validating whether the frame fits them. 

The first filter checks whether any of the lexical units within the frame shares the same POS as the new unseen lexical unit.

The second filter checks whether the exemplar sentence of the unseen lexical units contains the core frame elements of the assigned frame.

---

## Overall Implementation Details (`create_embeddings.py`)

### 1. Generate BERT embeddings for Lexical Units

Output: A pickled file `lus_fn1.7_bert.p` that saves the mapping of the IDs of lexical units to their BERT embeddings 

```python
def create_fn_LU_embeddings(embedding, save_file):
    """
:param embedding: flair.embeddings (e.g. BertEmbeddings())
:param save_file: file name to save the hash map of lexical units' IDs mapped to their respective embeddings
:return: None
"""
    LU_embedding = {}
    for i, lu in enumerate(list(fn.lus())):
        print(lu.name)
        num_embed = 0
        embed = torch.zeros((3072))
        if len(lu.exemplars) == 0:
            # zero tensor
            LU_embedding[lu.ID] = embed
        else:
            for sent in lu.exemplars:
                sentence = Sentence(sent.text)
                embedding.embed(sentence)
                for token in sentence:
                    if "Target" in sent.keys() and (token.start_pos, token.end_pos) in sent.Target:
                        embed.add_(token.embedding)
                        num_embed += 1
            LU_embedding[lu.ID] = embed/num_embed
    pickle.dump(LU_embedding, open(save_file, 'wb'))
```

### 2. Assigning New Lexical Units to Corresponding Frames

Output: A pickled file `closest_lus-{seg_file}.p` that stores the mapping of unseen_LU to the ID of closest lexical unis in FN1.7

- Function `get_unseen_LUs`: returns a hash table `unseen_LUs` that maps new lexical units to a list of sentences where the lexical units occur. The sentences are necessary for generating BERT embeddings for the new lexical units. 
- Function `create_unseen_LU_embedding`: stores the mapping the hash table that maps new lexical units to their embeddings in a pickled file
- Function `get_closest_LUs`: stores the mapping of unseen_LU to the ID of closest lexical unis in FN1.7 in a pickled file. The closest lexical units are obtained by choosing the lexical units whose embedding has the highest cosine similarity with the unseen lexical unit.

```python
def check_existing_LU(lu_name):
    return fn.lus(lu_name)

def get_unseen_LUs(filename):
    texts = list()
    p = re.compile(".*\|.*\|CC1\|(.*)")
    with open(filename, 'r') as f:
        for sent in f:
            res = p.search(sent)
            if res:
                texts.append(res.group(1))

    lus = collections.defaultdict(list)
    unseen_LUs = collections.defaultdict(list)
    for i, text in enumerate(texts):
        for j, token_word_pos in enumerate(pos_tag(text)):
            word, pos = token_word_pos
            # nouns
            if pos == "<NN>":
                lus[word.lower() + ".n"].append((text.lower(), j))


    for lu in lus.keys():
        if not check_existing_LU(lu):
            unseen_LUs[lu] = lus[lu]

    # {'plus.n': [('>> plus, warnings going unheard and lives lost.', 1)],
    # 'torch.n': [('>> passing the torch from 2018 to 2019.', 3)],
    # 'tabun.n': [('>> the tabun fire going off in the brooklyn neighborhood.', 2)]})
    return unseen_LUs
  
def create_unseen_LU_embedding(unseen_LUs_filename, embedding, save_to_filename):
    """
    :param unseen_LUs: a map of unseen_lu -> a list of sentences containing the lemma and its position
    # {'plus.n': [('>> plus, warnings going unheard and lives lost.', 1)],
    #  'torch.n': [('>> passing the torch from 2018 to 2019.', 3)],
    #  'tabun.n': [('>> the tabun fire going off in the brooklyn neighborhood.', 2)]})
    :param embedding: embedding, e.g. BertEmbedding()
    :param save_to_filename: save the map of unseen_LU -> embedding
    :return: None
    """
    unseen_LUs = pickle.load(open(unseen_LUs_filename, 'rb'))
    unseen_LU_embeddings = {}
    for lu in unseen_LUs.keys():
        num_embed = 0
        embed = torch.zeros((3072))
        for sent, idx in unseen_LUs[lu]:
            sentence = Sentence(sent, use_tokenizer=True)
            embedding.embed(sentence)
            embed.add_(sentence[idx].embedding)
            num_embed += 1
        print(f"Create embedding for {lu}")
        unseen_LU_embeddings[lu] = embed / num_embed
    pickle.dump(unseen_LU_embeddings, open(save_to_filename, 'wb'))
    
def get_closest_LUs(unseen_LU_embeddings_file, seen_LU_embeddings, res_file):
    """
    :param unseen_LU_embeddings_file: the file that stores the map of unseen_LU -> embedding
    :param seen_LU_embeddings: the file that stores the map of lexical units in FN1.7 -> embedding
    :param res_file: the file that stores the map of unseen_LU -> id of closest lexical unis in FN1.7

    :return: the map of unseen_LU -> id of closest lexical unis in FN1.7
    e.g. {'torch.n': 17625, 'ohio.n': 16339, 'hurt.n': 2311, 'tabun.n': 2676, 'neighborhood.n': 14551}
    """
    unseen_LU_embeddings = pickle.load(open(unseen_LU_embeddings_file, 'rb'))
    seen_LU_embeddings = pickle.load(open(seen_LU_embeddings, 'rb'))
    closest_LUs = {}
    cos = torch.nn.CosineSimilarity(dim=0)
    for unseen_lu, unseen_lu_embed in unseen_LU_embeddings.items():
        max_sim_score = float('-inf')
        res_LU_id = None
        for seen_lu_id, seen_lu_embed in seen_LU_embeddings.items():
            sim_score = cos(seen_lu_embed, unseen_lu_embed)
            if sim_score > max_sim_score:
                max_sim_score = sim_score
                res_LU_id = seen_lu_id
        closest_LUs[unseen_lu] = res_LU_id
    pickle.dump(closest_LUs, open(res_file, 'wb'))
    return closest_LUs
```

### 3A. Filtering Potentially New Lexical Units - POS Filter

Output: Two pickled files (`matched_pos_lus-{seg_file}.p` and `unmatched_pos_lus-{seg_file}.p`) where one file stores the new lexical units whose POS matches that of one of the lexical units in the assigned frame, and the other stores the lexical units whose POS does not.

- The first file (POS matches) stores the mapping of the new lexical units to the ID of closest lexical unis in FN1.7. This file will be parsed into the second filter.
- The second file (POS does not match) stores the mapping of the new lexical units to its embedding

```python
def check_POS_match(closest_lus_filename,
                    unseen_LU_embeddings_file,
                    matched_lus_filename,
                    unmatched_lus_embeddings_filename):
    """
    :param closest_lus_filename: the file that stores the map of unseen_LU -> id of closest lexical unis in FN1.7
    :param unseen_LU_embeddings_file: the file that stores the map of unseen_LU -> embedding
    :param matched_lus_filename: the file that stores the map of unseen_LU (match POS) -> embedding
    :param unmatched_lus_embeddings_filename: the file that stores the map of unseen_LU (doesn't match POS) -> embedding
    :return: None
    """
    closest_LUs = pickle.load(open(closest_lus_filename, 'rb'))
    unseen_LU_embeddings = pickle.load(open(unseen_LU_embeddings_file, 'rb'))
    unmatch_LU_to_embeddings = {}
    match_LU_to_ID = {}
    for lu, matched_id in closest_LUs.items():
        existing_POS = set()
        for lu_in_frame in fn.lu_basic(matched_id).frame.lexUnit:
            # lu_in_frame is a string, e.g. "burn.v"
            existing_POS.add(lu_in_frame.split('.')[-1])

        if lu.split('.')[-1] not in existing_POS:
            unmatch_LU_to_embeddings[lu] = unseen_LU_embeddings[lu]
        else:
            match_LU_to_ID[lu] = matched_id

    pickle.dump(unmatch_LU_to_embeddings, open(unmatched_lus_embeddings_filename, 'wb'))
    pickle.dump(match_LU_to_ID, open(matched_lus_filename, 'wb'))
```

### 3B. Filtering Potentially New Lexical Units - Core FEs Filter

This filter receives the pickled file where the new lexical units match the POS of the lexical units in the assigned frame from the POS filter. 

Output: Two pickled files (`matched_coreFEs_lus-{seg_file}.p` and `unmatched_coreFEs_lus-{seg_file}.p`) where one file stores the lexical units whose exemplar sentences contain core FEs of the assigned frame, and the other stores the lexical units whose exemplar sentences do not contain.

```python
def check_core_FE(unseen_LUs_sentence_filename,
                  open_sesame_folder_pathname,
                  closest_lus_filename,
                  unseen_LU_embeddings_file,
                  matched_lus_filename,
                  unmatched_lus_embeddings_filename):
    open_sesame_folder = pathlib.Path(open_sesame_folder_pathname)
    closest_lus = pickle.load(open(closest_lus_filename, 'rb'))
    unseen_LU_embeddings = pickle.load(open(unseen_LU_embeddings_file, 'rb'))
    unseen_LU_sentences = pickle.load(open(unseen_LUs_sentence_filename, 'rb'))
    match_LU_to_ID = {}
    unmatch_LU_to_embeddings = {}
    for lu, closest_lu_id in closest_lus.items():
        print(">>>>>>>>>>>>>>>> Lexical Unit:", lu)
        if not fn.lu_basic(closest_lus[lu]).frame.FEcoreSets:
            match_LU_to_ID[lu] = closest_lus[lu]
        else:
            core_FEs = [set(fe.name for fe in L) for L in fn.lu_basic(closest_lus[lu]).frame.FEcoreSets]
            details = unseen_LU_sentences[lu]
            # empty the input sentence file for OpenSesame
            with open(open_sesame_folder / 'sentences.txt', 'w') as input_f:
                pass

            # write the exemplar sentences into the input sentence file for OpenSesame
            with open(open_sesame_folder / 'sentences.txt', 'a') as input_f:
                for sent, idx in details:
                    input_f.write(sent)
                    input_f.write('\n')

            # run the command (OpenSesame's README)
            subprocess.call(f"python {open_sesame_folder_pathname}/sesame/targetid.py --mode predict"
                            f" --model_name fn1.7-pretrained-targetid --raw_input {open_sesame_folder_pathname}/sentences.txt",
                            shell=True)

            # instead of running the OpenSesame's README's command to generate frames
            # insert the predicted frames
            with open(open_sesame_folder / 'logs' / 'fn1.7-pretrained-frameid' / 'predicted-frames.conll', 'w') as frames_f:
                sent_idx = 0
                with open(open_sesame_folder / 'logs' / 'fn1.7-pretrained-targetid' / 'predicted-targets.conll', 'r') as f:
                    for line in f:
                        if line == "\n":
                            sent_idx += 1
                        else:
                            line = line.split("\t")
                            try:
                                if int(line[0]) - 1 == details[sent_idx][1] and line[5] == "NN":
                                    line[-2] = fn.lu_basic(closest_lus[lu]).frame.name
                                    line[-3] = lu
                                else:
                                    line[-3] = "_"
                            except:
                                print("Error Sentence:", line, details, sent_idx, int)
                            line = '\t'.join(line)
                        frames_f.write(line)
                        if sent_idx >= len(details):
                            break

            # OpenSesame - Annotate FEs
            subprocess.call(f"python {open_sesame_folder_pathname}/sesame/argid.py --mode predict"
                            f" --model_name fn1.7-pretrained-argid"
                            f" --raw_input {open_sesame_folder_pathname}/logs/fn1.7-pretrained-frameid/predicted-frames.conll",
                            shell=True)

            # compare with the core FEs in the predicted frame
            fes = set()
            with open(open_sesame_folder / 'logs' / 'fn1.7-pretrained-argid' / 'predicted-args.conll', 'r') as args_f:
                for line in args_f:
                    if line == "\n":
                        if fes and fes in core_FEs:
                           match_LU_to_ID[lu] = closest_lus[lu]
                        else:
                            unmatch_LU_to_embeddings[lu] = unseen_LU_embeddings[lu]
                        continue
                    else:
                        line = line.split("\t")
                        if line[-1] !="O\n":
                            fe = " ".join(line[-1].strip().split('-')[1:])
                            fes.add(fe)

    pickle.dump(unmatch_LU_to_embeddings, open(unmatched_lus_embeddings_filename, 'wb'))
    pickle.dump(match_LU_to_ID, open(matched_lus_filename, 'wb'))
```



---

## Documentation of Creating Singularity Containers

I create the Singularity container `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/production.sif` for this task of identifying new lexical units and creating BERT embeddings for them using Vagrant Box in MacOS.

**Dependencies Installation in Singularity Container**

```bash
pip3 --no-cache-dir install nltk
pip3 --no-cache-dir install torch
pip3 --no-cache-dir install flair

apt-get update
sudo apt upgrade
sudo apt install python2.7 python-pip
pip --no-cache-dir install dynet
pip --no-cache-dir install nltk
```

If the **Out of Memory (OOM)** error is encountered during the pip installation, allocate 2GB to the Vagrant Box virtual environment by including the following script into Vagrantfile.

```bash
Vagrant.configure("2") do |config|
	# ...
	
  config.vm.provider "virtualbox" do |v|
  	v.memory = 2048
  end
  
  # ...
end
```

---

## Documentation of Bugs / Challenges Encountered in Assigning New Lexical Units to Corresponding Frames

**POS-tagging for the unseen lexical units** - All the characters in NewsScape closed captions text are capitalized. This creates challenge for POS-tagging. I attempted UDPipe, NLTK and NLP4J; all of them gave subpar results such as classifying all capitalized nouns as NNP. `flair` library worked best in POS-tagging the sentence despite all characters are capitalized.

---

## Documentation of Bugs / Challenges Encountered in Checking Core Frame Elements

**Empty `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/open-sesame-local/logs/fn1.7-pretrained-targetid/predicted-target.conll`** - If this file is empty, we cannot insert predicted frames for the unseen lexical units and check for whether the core FE exists in the sentence. 

One of the reasons for the error is due to the removal of period at the end of the sentence. For example, running Open Sesame for this sentence `passing the torch from 2018 to 2019` would result in the error but not `passing the torch from 2018 to 2019.`.  

**Unable to identify the line for lexical unit in `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/open-sesame-local/logs/fn1.7-pretrained-targetid/predicted-target.conll` to insert the predicted frames** - Since the unseen lexical units are do not exist in FrameNet, OpenSesame cannot generate the frames for the new lexical units in their exemplar sentences. However, OpenSesame is used to annotate the frame elements in the sentence. Therefore, we need to manually insert the predicted frames for the new lexical units to produce the file `fn1.7-pretrained-frameid/predicted-frames.conll` by using the file `fn1.7-pretrained-targetid/predicted-target.conll`.

Here's what `fn1.7-pretrained-targetid/predicted-target.conll` looks like:

```
1	first	_	first	_	RB	0	_	_	_	_	_	first.adv	_	O
2	we	_	we	_	PRP	0	_	_	_	_	_	_	_	O
3	start	_	start	_	VBP	0	_	_	_	_	_	_	_	O
4	with	_	with	_	IN	0	_	_	_	_	_	_	_	O
5	the	_	the	_	DT	0	_	_	_	_	_	_	_	O
6	new	_	new	_	JJ	0	_	_	_	_	_	_	_	O
7	year	_	year	_	NN	0	_	_	_	_	_	_	_	O
8	's	_	's	_	POS	0	_	_	_	_	_	_	_	O
9	eve	_	UNK	_	VBP	0	_	_	_	_	_	_	_	O
10	party	_	party	_	NN	0	_	_	_	_	_	_	_	O
11	that	_	that	_	WDT	0	_	_	_	_	_	_	_	O
12	ended	_	end	_	VBD	0	_	_	_	_	_	_	_	O
13	with	_	with	_	IN	0	_	_	_	_	_	_	_	O
14	three	_	three	_	CD	0	_	_	_	_	_	_	_	O
15	dead	_	dead	_	JJ	0	_	_	_	_	_	_	_	O
16	,	_	,	_	,	0	_	_	_	_	_	_	_	O
17	two	_	two	_	CD	0	_	_	_	_	_	_	_	O
18	hurt	_	hurt	_	NN	0	_	_	_	_	_	_	_	O
19	.	_	.	_	.	0	_	_	_	_	_	_	_	O
```

This CoNLL file corresponds to the lexical unit `hurt.n` whose exemplar sentence is `first we start with the new year's eve party that ended with three dead, two hurt.` . Notice that the position of `hurt` in the exemplar sentence is at index 15, whereas that in the CoNLL file is at index 17 (using 0-index). In other words, we need to tokenize the exemplar sentence such that we can identify the position of the new lexical unit in the CoNLL file. The tokenizer that works for this task is the `Sentence` class of the  `flair`  library, which is used in the `get_unseen_LUs ` function. That is, `lus[word.lower() + ".n"].append( (Sentence(text, use_tokenizer=True).to_tokenized_string().lower(), j))` where `j` is the index of the new lexical unit in its exemplar sentence.

**Exception: ('two different frames in a single parse, illegal', 331, 0)** - This error signifies that there are more than one frame-inducing lexical units annotated in a single sentence. In other words, the `fn1.7-pretrained-frameid/predicted-frames.conll` looks like this where `first.adv` and `hurt.n` are frame-inducing lexical units:

```bash
1	first	_	first	_	RB	0	_	_	_	_	_	first.adv	_	O
2	we	_	we	_	PRP	0	_	_	_	_	_	_	_	O
3	start	_	start	_	VBP	0	_	_	_	_	_	_	_	O
4	with	_	with	_	IN	0	_	_	_	_	_	_	_	O
5	the	_	the	_	DT	0	_	_	_	_	_	_	_	O
6	new	_	new	_	JJ	0	_	_	_	_	_	_	_	O
7	year	_	year	_	NN	0	_	_	_	_	_	_	_	O
8	's	_	's	_	POS	0	_	_	_	_	_	_	_	O
9	eve	_	UNK	_	VBP	0	_	_	_	_	_	_	_	O
10	party	_	party	_	NN	0	_	_	_	_	_	_	_	O
11	that	_	that	_	WDT	0	_	_	_	_	_	_	_	O
12	ended	_	end	_	VBD	0	_	_	_	_	_	_	_	O
13	with	_	with	_	IN	0	_	_	_	_	_	_	_	O
14	three	_	three	_	CD	0	_	_	_	_	_	_	_	O
15	dead	_	dead	_	JJ	0	_	_	_	_	_	_	_	O
16	,	_	,	_	,	0	_	_	_	_	_	_	_	O
17	two	_	two	_	CD	0	_	_	_	_	_	_	_	O
18	hurt	_	hurt	_	NN	0	_	_	_	_	_	hurt.n	Cause_harm	O
19	.	_	.	_	.	0	_	_	_	_	_	_	_	O
```

Note: `Cause_harm` is the predicted frame manually inserted into `fn1.7-pretrained-frameid/predicted-frames.conll`, and since we do not run OpenSesame's command for generating frames, `first.adv` does not have a corresponding frame in `fn1.7-pretrained-frameid/predicted-frames.conll`.

**IndexError: list index out of range** - This error happens when the word in the sentence transformed into `UNK` when the sentence parses through `open-sesame-local/sesame/targetid.p`. An example would be the `logs/fn1.7-pretrained-targetid/predicted-targets.conll` CoNLL file for the exemplar sentence `there's no tape , no evidence markers ,  and no flashing ligh lights .` of the new lexical unit (which is a type) `ligh.n`

```bash
1	there	_	there	_	EX	0	_	_	_	_	_	there.adv	_	O
2	's	_	's	_	VBZ	0	_	_	_	_	_	_	_	O
3	no	_	no	_	DT	0	_	_	_	_	_	_	_	O
4	tape	_	tape	_	NN	0	_	_	_	_	_	_	_	O
5	,	_	,	_	,	0	_	_	_	_	_	_	_	O
6	no	_	no	_	DT	0	_	_	_	_	_	_	_	O
7	evidence	_	evidence	_	NN	0	_	_	_	_	_	_	_	O
8	markers	_	UNK	_	NNS	0	_	_	_	_	_	_	_	O
9	,	_	,	_	,	0	_	_	_	_	_	_	_	O
10	and	_	and	_	CC	0	_	_	_	_	_	_	_	O
11	no	_	no	_	DT	0	_	_	_	_	_	_	_	O
12	flashing	_	UNK	_	NN	0	_	_	_	_	_	_	_	O
13	UNK	_	UNK	_	JJ	0	_	_	_	_	_	_	_	O
14	lights	_	light	_	NNS	0	_	_	_	_	_	_	_	O
15	.	_	.	_	.	0	_	_	_	_	_	_	_	O
```

This error happens for lexical unit which is a typo, so this valence checking filter also filters out the new lexical units which are typo. 
