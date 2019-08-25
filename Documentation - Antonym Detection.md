# Antonym Detection

This documentation describes the work on identifying antonymous lexical units within the same frame. This section is inspired by Hasegawa et al. (2011)'s paper that suggests a frame containing antonymous LUs should be split into two subordinate frames with a new antonymic frame-to-frame relation between each subordinate frame and the parent frame.

**Table of Content**
- [Tutorial](#tutorial)
- [How It Works](#how-it-works)
- [Implementation Details](#implementation-details)
  - [Bugs And Challenges](#bugs-and-challenges)
  - [Documentation of Singularity Containers](#documentation-of-singularity-containers)
- [Future Direction](#future-direction)

---
## Tutorial
All the necessary files reside in the folder `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection`.

There are five slurm scripts (`/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/task{1/2/3/4/5}.slurm`) used to identify antonyms within FrameNet 1.7. 
- `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/task1.slurm` generates `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_1.p` which contains antonymous lexical units in the first 250 frames.
- `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/task2.slurm` generates `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_2.p` which contains antonymous lexical units in the 251st to 500th frames.
- `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/task3.slurm` generates `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_3.p` which contains antonymous lexical units in the 501st to 750th frames.
- `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/task4.slurm` generates `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_4.p` which contains antonymous lexical units in the 751st to 1000th frames.
- `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/task5.slurm` generates `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_5.p` which contains antonymous lexical units in the rest of the frames (from 1001st frames onward).

The slurm script `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/task1.slurm` is as followed:
```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=7-00:00:00
#SBATCH --output=my1.stdout
#SBATCH --error=my1.err
#SBATCH --job-name="antonym-detection"

module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
module load singularity
export SINGULARITY_BINDPATH="/home/zxy485/zxy485gallinahome/week9-12/antonym-detection:/mnt"

singularity exec production.sif python3 -u /mnt/deployed_antonym_1.py > ./output1.out
```

**Output**

`/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/output1.out` shows the progress of identifying antonymous lexical units within the first 250 frames. 

`/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_1.p` is a pickled file of a list of pairs of antonymous lexical units within the same frame. They are in the format of `(lexical unit 1, lexical unit 2, id of lexical unit 1, id of lexical unit 2)`. The following is the sample result.

```
..., ('button.v', 'fasten.v', 4544, 4677), ('button.v', 'unfasten.v', 4544, 4711), ('button.v', 'tie.v', 4544, 4757), ('open.v', 'buckle.v', 4545, 4547), ('open.v', 'cap.v', 4545, 4550), ...
```

---

## How It Works

### 1. Generate BERT embeddings for Lexical Units

Lexical units are the combination of lemmas and their part-of-speech tags. For example, "run.v", "long.v", "long.adj", etc. They are words that evoke a semantic frame (i.e., a description of a type of event, relation, or entity and the participants in it.) from FrameNet 1.7.

The BERT embedding of a lexical unit is obtained by averaging the BERT embeddings of the lexical unit appearing in the annotated sentences in FrameNet. If there are no sentence examples of the lexical unit, the embedding of the lexical unit will be a zero tensor.

### 2. Generate BERT embeddings for WordNet Synsets

WordNet is the lexical database i.e., dictionary for the English language, specially designed for natural language processing. Nouns, verbs, adjectives, and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked through conceptual-semantic and lexical relations.

The BERT embedding of a synset is obtained by averaging the BERT embeddings of the synset's lemmas appearing in the annotated sentences in WordNet. The sentences will be preprocessed by tokenization and lemmatization. If the synset does not have any example sentence, it will not be included in the hash-map that maps the synset's name to its BERT embeddings. In other words, not all the synsets from WordNet have their respective embeddings (not even a zero tensor).

### 3. Generating Training and Testing Dataset from WordNet

Dataset of antonymous pairs of lemmas was generated to train a decision tree classifier. The training dataset was the cosine-similarity of antonymous pairs of synsets and non-antonymous pairs of synsets. Non-antonymous pairs of synsets were obtained by randomly pairing two synsets which are not antonymous to each other. The number of non-antonymous pairs of synsets was adjusted such that it matched the number of the antonymous pairs. 

### 4. Training the Decision Tree Classifier

Subsequently, I trained the decision tree classifier (which uses CART algorithm) from `sklearn` library with the dataset generated. The antonyms and non-antonyms were split by a ratio of 0.33 into training and testing dataset.

### 5. Testing the Classifier

### 5A - POS

Without factoring POS into account, there were 3322 antonyms pairs in total and 3322 self-generated non-antonym pairs. The accuracy of the classifier was 0.76. 

After factoring POS into account, there were 2320 antonyms pairs in total and 2320 self-generated non-antonym pairs. Each pair of the antonymous and non-antonymous pairs of synsets shared the same POS. The accuracy of the classifier was 0.83. 

**5B - Dependency-Parsing**

After inclduing the syntactic relations, which are the type of dependency relations and the level of the node in the parse tree, to the input, the accuracy increased to 0.88. 

### 6. Applying Classifier to FrameNet

For each frame in FrameNet, all of its lexical units with the same POS tag were grouped in combinations of pairs. To reduce the number of false positives, the input to the classifier was [x1, x2] where: 

- x1 is a list consists of (in the following order) the cosine similarity between the two lexical units, the average type of depedency relation of the first lexical unit (in integer), the average level of the first lexical unit node in the parse trees of the exemplar sentences, the average type of depedency relation of the second lexical unit (in integer), and the average level of the second lexical unit node in the parse trees of the exemplar sentences.

- x2 is a list consists of (in the following order) the cosine similarity between the two lexical units, the average type of depedency relation of the second lexical unit (in integer), the average level of the second lexical unit node in the parse trees of the exemplar sentences, the average type of depedency relation of the first lexical unit (in integer), and the average level of the first lexical unit node in the parse trees of the exemplar sentences.

The returned result of the function below is a list of tuples of antonymous pairs in the format of (L1, L2, id(L1), id(L2)) where L1 and L2 are antonymous lexical units and id() is a function that maps the lexical unit to its respective ID in FrameNet.

---

## Implementation Details
(`/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/deployed_antonym_{1/2/3/4/5}.py`)

### 1. Generate BERT embeddings for Lexical Units

Output: A pickled file (`lus_fn1.7_bert.p`) that saved the mapping of the IDs of lexical units to their BERT embeddings

The function is a script in another folder `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/create_embeddings.py`.

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

### 2. Generate BERT embeddings for WordNet Synsets

Output: A pickled file (`synsets_wn_bert.p`) that saved the mapping of the names of the WordNet synsets (e.g. `able.a.01`) to their BERT embeddings.

```python
def generate_embeddings_synsets(save_embedding_file, embedding=BertEmbeddings()):
    """
:param save_embedding_file: file name to save the hash map of synsets' names mapped to their respective embeddings
:param embedding: flair.embeddings (e.g. BertEmbeddings())
:return: None
"""
    syn_embeddings = {}
    lemmatizer = WordNetLemmatizer()
    for i, syn in enumerate(list(sorted(wn.all_synsets(), key=lambda syn: syn.name()))):
        if not syn.examples():
            continue

        syn_name = syn.name().split('.')[0]
        syn_pos = syn.name().split('.')[1]
        print(i, syn_name)
        embed = torch.zeros((3072))
        num_embed = 0
        for sent in syn.examples():
            sentence = Sentence(sent.lower(), use_tokenizer=True)
            embedding.embed(sentence)
            for token in sentence.tokens:
                try:
                    lemmatized_word = lemmatizer.lemmatize(token.text, syn_pos)
                    if lemmatized_word == syn_name:
                        embed.add_(token.embedding)
                        num_embed += 1
                except:
                    print("Error with token:", token.text, "when processing", syn.name())

        if any(embed):
            # non-zero tensor
            syn_embeddings[syn.name()] = embed / num_embed
    pickle.dump(syn_embeddings, open(save_embedding_file, 'wb'))
```

### 3. Generating Training and Testing Dataset from WordNet (with POS and Dependency Parsing)

Output: A pickled dataset named `generate_antonym_dataset_cosine_similarity_dep_with_non_antonym.p`. 

Overall, the function generates all the antonym pairs before generating the same number of non-antonymous pairs.

- Only pairs of synsets that share the same POS are considered in the dataset.
- The function `get_dep_relations` returns a mapping of each token in the tokenized sentence to its level and dependency relations in the dependency-parsed tree. This uses the API of the UDPipe. An example would be `{'take.VERB': [(0, 'root')], 'she.PRON': [(1, 'nsubj'), (2, 'nmod:poss')], 'care.NOUN': [(1, 'obj')], 'man.NOUN': [(1, 'obl')], 'good.ADJ': [(2, 'amod')], 'of.ADP': [(2, 'case')], 'faithful.ADJ': [(0, 'root')], 'woman.NOUN': [(1, 'obl')], 'be.AUX': [(1, 'cop')], 'he.PRON': [(1, 'nsubj'), (2, 'nmod:poss')], 'to.ADP': [(2, 'case')]}`
- The function `process_dep_relations` takes in the hash-map returned by `get_dep_relations` and average the level and dependency relations (in integer representation) of each token in the tokenized sentence. The example would be `{'take.VERB': (0.0, 34.0), 'she.PRON': (1.5, 25.5), 'man.NOUN': (1.0, 29.0), 'care.NOUN': (1.0, 28.0), 'of.ADP': (2.0, 6.0), 'good.ADJ': (2.0, 3.0), 'faithful.ADJ': (0.0, 34.0), 'be.AUX': (1.0, 12.0), 'he.PRON': (1.5, 25.5), 'woman.NOUN': (1.0, 29.0), 'to.ADP': (2.0, 6.0)}`
- `cos(X, Y)` generates the cosine similarity of two word embeddings `X` and `Y`.
- `X = [x1, x2, x3, ...]` where each `x` is a tuple of cosine similarity of the two synset embeddings, the average dependency level of the first synset, the average dependency relations of the first synset, the average dependency level of the second synset, and the average dependency relations of the second synset. 
- `Y = [y1, y2, y3]` where each `y` is `1` if it corresponds to an antonymous pair and `0` if it corresponds to a non-antonymous pair.

```python
def generate_antonym_dataset_cosine_similarity_dep(H):
    cos = torch.nn.CosineSimilarity(dim=0)
    X = []
    Y = []
    # antonym pairs
    for syn_name, embed in H.items():
        syn = wn.synset(syn_name)
        dep_rel_map_syn = collections.defaultdict(list)
        for sent in syn.examples():
            dep_rel_map_syn = get_dep_relations(sent, dep_rel_map_syn)

        for l in syn.lemmas():
            if l.antonyms() and l.antonyms()[0].synset().name() in H:
                antonym_name = l.antonyms()[0].synset().name()
                lemma_1, pos_1, _ = syn_name.split(".")
                lemma_2, pos_2, _ = antonym_name.split(".")

                # same POS word space
                if pos_1 != pos_2:
                    continue

                print("Antonym Pairs:", syn_name, antonym_name)
                dep_rel_map = deepcopy(dep_rel_map_syn)
                for sent in l.antonyms()[0].synset().examples():
                    dep_rel_map = get_dep_relations(sent, dep_rel_map)
                dep_rel_map = process_dep_relations(dep_rel_map)

                if f"{lemma_1}.{WORDNET_POS_TO_UPOS[pos_1]}" in dep_rel_map and \
                        f"{lemma_2}.{WORDNET_POS_TO_UPOS[pos_2]}" in dep_rel_map:
                    x = [cos(embed, H[antonym_name]),
                         *dep_rel_map[f"{lemma_1}.{WORDNET_POS_TO_UPOS[pos_1]}"],
                         *dep_rel_map[f"{lemma_2}.{WORDNET_POS_TO_UPOS[pos_2]}"]]
                    X.append(x)
                    Y.append(1)
                else:
                    print(dep_rel_map, lemma_1, pos_1, lemma_2, pos_2)
                    # raise AssertionError

    pickle.dump([X, Y], open("generate_antonym_dataset_cosine_similarity_dep.p", "wb"))

    X, Y = pickle.load(open("generate_antonym_dataset_cosine_similarity_dep.p", "rb"))

    total_antonym_pairs = len(X)
    r = random.Random(total_antonym_pairs)
    seen = set()
    # non-antonym pairs
    while len(X) < 2 * total_antonym_pairs:
        syn_name, embed = r.choice(list(H.items()))
        syn = wn.synset(syn_name)
        syn_name_2, embed_2 = r.choice(list(H.items()))

        lemma_1, pos_1, _ = syn_name.split(".")
        lemma_2, pos_2, _ = syn_name_2.split(".")

        if syn_name == syn_name_2 or pos_1 != pos_2 or (syn_name, syn_name_2) in seen:
            continue

        print("Non-Antonym Pairs:", syn_name, syn_name_2)
        dep_rel_map_syn = collections.defaultdict(list)
        for sent in syn.examples():
            dep_rel_map_syn = get_dep_relations(sent, dep_rel_map_syn)

        for l in syn.lemmas():
            if not l.antonyms() or syn_name_2 != l.antonyms()[0].synset().name():
                dep_rel_map = deepcopy(dep_rel_map_syn)
                for sent in wn.synset(syn_name_2).examples():
                    dep_rel_map = get_dep_relations(sent, dep_rel_map)
                dep_rel_map = process_dep_relations(dep_rel_map)

                if f"{lemma_1}.{WORDNET_POS_TO_UPOS[pos_1]}" in dep_rel_map and \
                                    f"{lemma_2}.{WORDNET_POS_TO_UPOS[pos_2]}" in dep_rel_map:
                    x = [cos(embed, embed_2),
                         *dep_rel_map[f"{lemma_1}.{WORDNET_POS_TO_UPOS[pos_1]}"],
                         *dep_rel_map[f"{lemma_2}.{WORDNET_POS_TO_UPOS[pos_2]}"]]
                    X.append(x)
                    Y.append(0)
                    seen.add((syn_name, syn_name_2))
                else:
                    print(dep_rel_map, lemma_1, pos_1, lemma_2, pos_2)
                    # raise AssertionError
    assert len(X) == len(Y)
    pickle.dump([X, Y], open("generate_antonym_dataset_cosine_similarity_dep_with_non_antonym.p", "wb"))
    return X, Y
```

##### 4 / 5. Training and Testing the Decision Tree Classifier

Output: The accuracy score of the classifier and the saved file of the trained classifier model

```python
@contextmanager
def timer(msg):
    print(f"[{msg}] starts.")
    start_time = time.time()
    yield
    duration = (time.time() - start_time)/60
    print(f"[{msg}] takes {duration:.2f} minutes.")

if __name__ == "__main__":
    H = pickle.load(open("./synsets_wn_bert.p", 'rb'))

    with timer("Generate Antonym Training Data"):
        X, Y = generate_antonym_dataset_cosine_similarity_dep(H)

    with timer("Training Decision Tree Classifier"):
        X, Y = pickle.load(open("generate_antonym_dataset_cosine_similarity_dep_with_non_antonym.p", 'rb'))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, random_state=47)
        print("Training Dataset Size:", len(X_train), "Testing Dataset Size:", len(X_test))
        model = DecisionTreeClassifier()

        model.fit(X_train, Y_train)
        joblib.dump(model, "antonym_decision_tree_classifier_cosine_sim_with_dep.joblib")

    with timer("Testing the Classifier"):
        Y_predict = model.predict(X_test)
        print(Y_predict)
        print(accuracy_score(Y_test, Y_predict))
```

### 6. Applying Classifier to FrameNet

Output: The pickled files (`/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/potential_antonyms_cosine_sim_with_dep_{1/2/3/4/5}.p`) that contains the list of tuples of antonymous pairs in the format of (L1, L2, id(L1), id(L2)) where L1 and L2 are antonymous lexical units and id() is a function that maps the lexical unit to its respective ID in FrameNet.

- Only pairs of lexical units which are in the same frame and have the same POS were classified.
- The function `get_dep_relations` returned a mapping of each token in the tokenized sentence to its level and dependency relations in the dependency-parsed tree. This used the API of the UDPipe. An example would be `{'abandon.VERB': [(0, 'root'), (2, 'conj'), (2, 'acl'), (0, 'root'), (0, 'root'), (1, 'parataxis'), (0, 'root'), (2, 'advcl'), (2, 'conj'), (1, 'conj'), (0, 'root'), (1, 'ccomp'), .. (0, 'root')], '..PUNCT': [(1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), .. ], 'plan.NOUN': [(1, 'obj'), (1, 'obj'), (2, 'obj'), (1, 'obj')],  ... }`
- The function `process_dep_relations` took in the hash-map returned by `get_dep_relations` and averaged the level and dependency relations (in integer representation) of each token in the tokenized sentence. The example would be `{'abandon.VERB': (0.9130434782608695, 17.869565217391305), '..PUNCT': (1.0, 32.0), 'plan.NOUN': (1.25, 28.0), ...}`
- `cos(X, Y)` generated the cosine similarity of two word embeddings `X` and `Y`.
- `X = [x1, x2]` where `x1` is a tuple of cosine similarity of the two LUs' embeddings, the average dependency level of the first LU, the average dependency relations of the first LU, the average dependency level of the second LU, and the average dependency relations of the second LU. `x2`  is a tuple of cosine similarity of the two LUs' embeddings, the average dependency level of the second LU, the average dependency relations of the second LU, the average dependency level of the first LU, and the average dependency relations of the first LU. The example is `[[tensor(0.6349), 0.9130434782608695, 17.869565217391305, 1.2, 20.0], [tensor(0.6349), 1.2, 20.0, 0.9130434782608695, 17.869565217391305]]`

```python
def detect_antonyms_framenet_cosine_similarity_dep(lus_embed_file, model):
    LU_embedding = pickle.load(open(lus_embed_file, 'rb'))
    antonyms = list()
    cos = torch.nn.CosineSimilarity(dim=0)
    for frame in list(fn.frames())[:250]:
        print(frame.name)
        seen = set()
        for lu_name1 in frame.lexUnit.keys():
            POS = lu_name1.split(".")[-1]
            seen.add(lu_name1)
            dep_rel_map_lu1 = collections.defaultdict(list)
            for sent in frame.lexUnit[lu_name1].exemplars:
                try:
                    dep_rel_map_lu1 = get_dep_relations(sent.text, dep_rel_map_lu1)
                except:
                    dep_rel_map_lu1 = collections.defaultdict(list)

            for lu_name2 in frame.lexUnit.keys():
                if lu_name2 in seen or lu_name2.split(".")[-1] != POS:
                    continue

                lu_id1 = frame.lexUnit[lu_name1].ID
                lu_id2 = frame.lexUnit[lu_name2].ID

                # if LU is not in LU_embedding or LU_embedding[lu_id] is a zero tensor, skip
                if lu_id1 not in LU_embedding or not any(LU_embedding[lu_id1]) \
                        or lu_id2 not in LU_embedding or not any(LU_embedding[lu_id2]):
                    continue

                cos_sim = cos(LU_embedding[lu_id1], LU_embedding[lu_id2])

                dep_rel_map = deepcopy(dep_rel_map_lu1)
                for sent in frame.lexUnit[lu_name2].exemplars:
                    try:
                        dep_rel_map = get_dep_relations(sent.text, dep_rel_map)
                    except:
                        dep_rel_map = deepcopy(dep_rel_map_lu1)
                dep_rel_map = process_dep_relations(dep_rel_map)

                if f"{lu_name1.split('.')[0]}.{FRAMENET_POS_TO_UPOS[frame.lexUnit[lu_name1].POS]}" in dep_rel_map and \
                    f"{lu_name2.split('.')[0]}.{FRAMENET_POS_TO_UPOS[frame.lexUnit[lu_name2].POS]}" in dep_rel_map:
                    x1 = [cos_sim,
                         *dep_rel_map[f"{lu_name1.split('.')[0]}.{FRAMENET_POS_TO_UPOS[frame.lexUnit[lu_name1].POS]}"],
                         *dep_rel_map[f"{lu_name2.split('.')[0]}.{FRAMENET_POS_TO_UPOS[frame.lexUnit[lu_name2].POS]}"]]

                    x2 = [cos_sim,
                          *dep_rel_map[f"{lu_name2.split('.')[0]}.{FRAMENET_POS_TO_UPOS[frame.lexUnit[lu_name2].POS]}"],
                          *dep_rel_map[f"{lu_name1.split('.')[0]}.{FRAMENET_POS_TO_UPOS[frame.lexUnit[lu_name1].POS]}"]]
                    X = [x1, x2]
                    if all(model.predict(X)):
                        antonyms.append((lu_name1, lu_name2, lu_id1, lu_id2))
                        print((lu_name1, lu_name2, lu_id1, lu_id2))


    pickle.dump(antonyms, open("/mnt/potential_antonyms_cosine_sim_with_dep_1.p", 'wb'))
    return antonyms
```

---
## Bugs And Challenges

**Long Processing Time** - To hasten the process of identifying all the antonyms, I used the parallel processing paradigm by creating five Python scripts (`deployed_antonym_1.py`, `deployed_antonym_2.py`, `deployed_antonym_3.py`, `deployed_antonym_4.py`, `deployed_antonym_5.py` in the folder `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection`) for processing all the frames within FrameNet where each script handled at most 250 frames. 

**Module Not Found Error** (`ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`) - This error was caused by my previous local installation of `nvidia/apex` for BERT processing using the `pytorch-pretrained-bert` library in HPC clusters. Even when `import apex` was not called in the script, if `apex` was installed and detected, the Python script would run into the `ModuleNotFoundError`. The first reason was that all Python scripts were not run on a GPU node with CUDA - instead, they were run with CPU nodes of HPC clusters. Second, the deployed Singularity container that was built on my personal MacOS environment did not have the `apex` library installed with CUDA extensions due to the lack of GPUs, and it's impossible to `sudo pip3 install` the `apex` library on the HPC clusters.

---
## Documentation of Singularity Containers

I created the Singularity container `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/production.sif` for this task of identifying new lexical units and creating BERT embeddings for them using Vagrant Box in MacOS.

**Dependencies Installation in Singularity Container**

```bash
pip3 --no-cache-dir install flair
pip3 --no-cache-dir install nltk
```

If the **Out of Memory (OOM)** error is encountered during the `pip3` installation, allocate 2GB to the Vagrant Box virtual environment by including the following script into Vagrantfile.

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
## Future Direction
According to "FrameNet II: Extended Theory and Practice", FrameNet is structured as such:
1. words that are antonyms of each other are grouped. For example, high and low are both in the Position on a scale frame.
2. relational opposites which take opposite points of view on a single event are placed into separate frames because they profile different (sets of) participants. For example, buyer.n and seller.n are in separate frames.
3. reversive pairs are placed in different frames since the members of the pairs denote different kinds of actions.

I manually looked through 150 randomly selected frames within FrameNet, and I classified them into
1. frames where antonyms are grouped together
2. antonymous frames due to the reversive pairs
3. relational opposites 
4. erroneous frames which violate the aforementioned FrameNet structure. 

```
(1) frames where antonyms are grouped together

 Temperature : ['temperature.n', 'hot.a', 'cool.a', 'freezing.a', 'cold.a', 'lukewarm.a', 'tepid.a', 'frigid.a', 'scalding.a', 'warm.a']
- Posture : ['bend.v', 'crouch.v', 'hunch.v', 'huddle.v', 'kneel.v', 'lean.v', 'lie.v', 'sit.v', 'slouch.v', 'sprawl.v', 'squat.v', 'stand.v', 'stoop.v', 'bent.a', 'crouched.a', 'huddled.a', 'hunched.a', 'sprawled.a', 'slouched.a', 'seated.a', 'posture.n', 'stance.n', 'position.n', 'cower.v', 'shrink.v']
- Ordinal_numbers : ['first.a', 'second.a', 'third.a', 'fourth.a', 'fifth.a', 'thirteenth.a', 'nineteenth.a', 'last.a', 'eighth.a', 'tenth.a', 'ninth.a', 'seventeenth.a', 'sixteenth.a', 'final.a']
- Change_of_leadership : ['coup.n', 'elect.v', 'insurrection.n', 'mutiny.n', 'mutiny.v', 'overthrow.v', 'rebellion.n', 'revolt.v', 'revolution.n', 'uprising.n', 'oust.v', 'depose.v', 'dethrone.v', 'overthrow.n', 'revolt.n', 'take over.v', 'topple.v', 'election.n', 'throne.v', 'enthrone.v', 'coronate.v', 'crown.v', 'vest.v', 'install.v', 'revolutionary.n', 'independence.n', 'rising.n', 'freedom.n', 'ouster.n']
- Gradable_proximity : ['near.a', 'close.a', 'proximity.n', 'far.a', 'distant.a']
- Attention : ['alert.a', 'attend.v', 'attention.n', 'attentive.a', 'closely.adv', 'close.a', 'keep an eye.v', 'ignore.v']
- Direction : ['east.adv', 'up.adv', 'forward.adv', 'left.adv', 'right.adv', 'north.adv', 'south.n', 'east.n', 'south.adv', 'down.adv', 'outward.adv', 'north.n', 'west.adv', 'true north.n', 'way.n']
- Judgment : ['admiration.n', 'admire.v', 'admiring.a', 'applaud.v', 'appreciate.v', 'appreciation.n', 'approbation.n', 'approving.a', 'blame.n', 'blame.v', 'contempt.n', 'contemptuous.a', 'critical.a', 'damnation.n', 'deplore.v', 'derisive.a', 'disapproval.n', 'disapprove.v', 'disapproving.a', 'disdain.n', 'disdain.v', 'disdainful.a', 'disrespect.n', 'esteem.n', 'esteem.v', 'fault.n', 'fault.v', 'mock.v', 'reproachful.a', 'scorn.n', 'scorn.v', 'scornful.a', 'stigma.n', 'stigmatize.v', 'stricture.n', 'uncritical.a', 'exalt.v', 'prize.v', 'boo.v', 'revere.v', 'reverence.n', 'mockery.n', 'exaltation.n', 'accolade.n', 'vilification.n', 'value.v', 'respect.n', 'respect.v', 'deify.v', 'reproach.n', 'reprehensible.a', 'appreciative.a', 'set store.v', 'accuse.v', 'approve.v']
- Extreme_value : ['high.n', 'low.n', 'acme.n', 'maximum.n', 'maximum.a', 'minimum.n', 'minimum.a']
- Having_or_lacking_access : ['access.n', 'blocked.a', 'accessible.a', 'inaccessible.a', 'access.v']
- Artificiality : ['artificial.a', 'bogus.a', 'fake.a', 'counterfeit.a', 'false.a', 'genuine.a', 'ersatz.a', 'phoney.n', 'pseudo.a', 'phoney.a', 'real.a', 'actual.a', 'true.a', 'disingenuous.a']
- Change_of_phase : ['freeze.v', 'liquefy.v', 'vaporize.v', 'evaporate.v', 'solidify.v', 'thaw.v', 'sublime.v', 'condense.v', 'melt.v', 'defrost.v', 'condensation.n', 'evaporation.n', 'solidification.n', 'sublimation.n', 'unfreeze.v']
- Inclusion : ['include.v', 'contain.v', 'have.v', 'integrated.a', 'exclude.v', 'excluding.prep', 'inclusive.a', 'including.prep', 'incorporate.v']
- Chemical-sense_description : ['tasty.a', 'piquant.a', 'yummy.a', 'flavourful.a', 'scrumptious.a', 'palatable.a', 'delectable.a', 'flavoursome.a', 'toothsome.a', 'ambrosial.a', 'sapid.a', 'salty.a', 'sour.a', 'bitter.a', 'spicy.a', 'sweet.a', 'hot.a', 'savory.a', 'delicious.a', 'pungent.a', 'tart.a', 'flavourless.a', 'bland.a', 'insipid.a', 'unpalatable.a', 'stench.n', 'odor.n', 'reek.n', 'reek.v', 'stink.n', 'stink.v', 'aroma.n', 'fragrance.n', 'scent.n', 'bouquet.n', 'smelly.a', 'aromatic.a', 'fragrant.a', 'tasteless.a', 'malodorous.a', 'smell.v']
- Capability : ['able.a', 'unable.a', 'can.v', 'capable.a', 'ability.n', 'potential.n', 'capability.n', 'capacity.n', 'potential.a', 'power.n', 'power [statistical].n', 'powerless.a', 'powerlessness.n', 'inability.n', 'incapable.a', 'incapacity.n', 'powerful.a']
- Awareness : ['aware.a', 'awareness.n', 'believe.v', 'comprehend.v', 'comprehension.n', 'conceive.v', 'conception.n', 'conscious.a', 'hunch.n', 'imagine.v', 'know.v', 'knowledge.n', 'knowledgeable.a', 'presume.v', 'presumption.n', 'reckon.v', 'supposition.n', 'suspect.v', 'suspicion.n', 'think.v', 'thought.n', 'understand.v', 'understanding.n', 'ignorance.n', 'consciousness.n', 'cognizant.a', 'unknown.a', 'idea.n']
- Hit_or_miss : ['hit.v', 'miss.v', 'wide.adv', 'hit.n', 'bullseye.n', 'hole in one.n']
- Sufficiency : ['enough.n', 'enough.adv', 'enough.a', 'suffice.v', 'sufficient.a', 'sufficiently.adv', 'insufficient.a', 'insufficiently.adv', 'adequate.a', 'adequately.adv', 'insufficiency.n', 'adequacy.n', 'inadequacy.n', 'inadequate.a', 'inadequately.adv', 'plenty.n', 'plenty.adv', 'plenty.a', 'ample.a', 'too.adv', 'so.adv', 'serve.v']
- Age : ['old.a', 'ancient.a', 'young.a', 'youngish.a', 'oldish.a', 'age.n', 'new.a', 'fresh.a', 'of.prep', 'maturity.n', 'mature.a', 'elderly.a']
- Reasoning : ['argue.v', 'prove.v', 'reason.v', 'demonstrate.v', 'show.v', 'disprove.v', 'argument.n', 'polemic.n', 'case.n', 'demonstration.n', 'reasoning.n']
- Being_in_effect : ['effective.a', 'effect.n', 'force.n', 'valid.a', 'void.a', 'null.a', 'binding.a']
- Taking_sides : ['oppose.v', 'for.prep', 'against.prep', 'pro.adv', 'support.v', 'side.v', 'side.n', 'opposition [act].n', 'in favor.prep', 'supportive.a', 'opponent.n', 'supporter.n', 'opposition [entity].n', 'endorse.v', 'back.v', 'backing.n', 'believe (in).v', 'part.n']
- Interrupt_process : ['uninterrupted.a', 'interrupt.v', 'interruption.n']
- Being_at_risk : ['secure.a', 'security.n', 'safe.a', 'insecure.a', 'unsafe.a', 'safety.n', 'risk.n', 'vulnerable.a', 'vulnerability.n', 'danger.n', 'threatened.a', 'susceptibility.n', 'susceptible.a']
- Preference : ['prefer.v', 'disprefer.v', 'favor.v']
- Medical_conditions : ['acromegaly.n', 'amnesia.n', 'arthritis.n', 'anorexia.n', 'asphyxia.n', 'asthma.n', 'bronchitis.n', 'cancer.n', 'candida.n', 'cataract.n', 'cholangitis.n', 'cholecystitis.n', 'cholera.n', 'cirrhosis.n', 'cold.n', 'colitis.n', 'conjunctivitis.n', 'cryptosporidiosis.n', 'diarrhea.n', 'depression.n', 'dermatitis.n', 'diabetes.n', 'diphtheria.n', 'diverticulosis.n', 'dysmenorrhoea.n', 'eczema.n', 'flu.n', 'influenza.n', 'hepatitis.n', 'hernia.n', 'hypertension.n', 'hypoglycaemia.n', 'illness.n', 'disease.n', 'sickness.n', 'infection.n', 'jaundice.n', 'leprosy.n', 'leukemia.n', 'malaria.n', 'measles.n', 'meningitis.n', 'menorrhagia.n', 'polio.n', 'psoriasis.n', 'pyelonephritis.n', 'rosacea.n', 'schizophrenia.n', 'sciatica.n', 'shock.n', 'stress.n', 'strongylosis.n', 'syphilis.n', 'tetanus.n', 'tuberculosis.n', 'tumor.n', 'ulcer.n', 'rubella.n', 'mumps.n', 'bacterial meningitis.n', 'German measles.n', 'ailment.n', 'wound.n', 'affliction.n', 'condition.n', 'syndrome.n', 'stenosis.n', 'sick.a', 'ill.a', 'unwell.a', 'health.n', 'plague.n', 'malnourishment.n', 'malnutrition.n', 'paraplegic.a', 'healthy.a', 'AIDS.n', "Alzheimer's.n", 'disorder.n', 'develop.v']
- Willingness : ['willing.a', 'unwilling.a', 'reluctant.a', 'willingness.n', 'loath.a', 'grudging.a', 'unwillingness.n', 'prepared.a', 'down.a']
- Aesthetics : ['beautiful.a', 'lovely.a', 'smart.a', 'ugly.a', 'tasty.a', 'elegant.a', 'fair.a', 'handsome.a', 'hideous.a']
- Being_relevant : ['relevant.a', 'irrelevant.a', 'pertinent.a', 'play (into).v']
- Usefulness : ['good.a', 'effective.a', 'excellent.a', 'outstanding.a', 'superb.a', 'wonderful.a', 'fantastic.a', 'fine.a', 'tremendous.a', 'terrific.a', 'marvellous.a', 'great.a', 'super.a', 'splendid.a', 'useful.a', 'utility.n', 'value.n', 'valuable.a', 'perfect.a', 'ideal.a', 'ineffective.a', 'work.v', 'strong.a']
- Openness : ['open.a', 'closed.a', 'dark.a']
- Luck : ['lucky.a', 'luck.n', 'happy.a', 'fortunate.a', 'luckily.adv', 'fortunately.adv', 'fortune.n', 'fortuitous.a', 'poor.a']

(2) Reversive Pairs (Actions are different and/or CoreFEs are different)

- Becoming_attached - Becoming_detached
- Cause_to_amalgamate - Cause_to_fragment
- Non-commutative_process - Commutative_process
- Being_detached - Being_attached
- Ceasing_to_be - Becoming_visible
- Sleep - Being_awake
- Process_completed_state - Process_uncompleted_state 
- Process_end - Process_start
- Activity_ongoing - Activity_pause
- Thriving - Death
- Erasing - Protecting
- Cause_to_move_in_place - Halt
- Subversion - Support
- Firing - Employing
- Process_continue - Process_pause
- Activity_start - Activity_finish
- Cause_to_make_noise - Silencing
- Avoiding - Confronting_problem
- Cause_to_be_dry - Cause_to_be_wet 
- Catching_fire - Fire_going_out
- Being_attached - Being_detached
- Beyond_compare - Being_up_to_it
- Being_dry - Being_wet
- Waking_up - Sleep

(3) Relational Opposites

- Commerce_buy - Commerce_sell
- Dominate_situation - Giving_in

(4) Erroneous Frames

- Guest_and_host : ['guest.n', 'host.n']
- Import_export_scenario : ['transship.v', 'exporter.n', 'importer.n', 'trans-shipping.n']
- Visitor_and_host : []
- Personal_relationship : ['husband.n', 'wife.n', 'widow.n', 'widower.n', 'widowed.a', 'spouse.n', 'couple.n', 'companion.n', 'divorcee.n', 'married.a', 'betrothed.n', 'bachelor.n', 'engaged.a', 'engagement.n', 'marriage.n', 'boyfriend.n', 'girlfriend.n', 'single.a', 'date.v', 'break-up.n', 'lover.n', 'partner.n', 'mistress.n', 'suitor.n', 'court.v', 'spinster.n', 'estranged.a', 'friend.n', 'betrothed.a', 'cohabit.v', 'cohabitation.n', 'seeing.v', 'widow.v', 'divorced.a', 'buddy.n', 'moll.n', 'paramour.n', 'inamorata.n', 'beau.n', 'sugar daddy.n', 'significant other.n', 'chum.n', 'pal.n', 'cobber.n', 'mate.n', 'crush.n', 'affair.n', 'adultery.n', 'amigo.n', 'sleep (together/with).v', 'spousal.a', 'marital.a', 'affianced.a', 'friendship.n', 'romance.n', 'befriend.v', 'familiar.a', 'fiancée.n', 'fiancé.n', 'relationship.n', 'chummy.a']
- Medical_interaction_scenario : ['patient.n', 'doctor.n', 'nurse.n', 'surgeon.n', 'medical care.n']
- Employing : [commission.v, employ.v, employee.n, employer.n, employment.n, personnel.n, staff.n, worker.n]
```

It doesn’t seem worthwhile resolving inconsistency because it appears that FrameNet’s frames are supposed to contain antonyms such as dry and wet in the same frame, where the lexical units differ in their semantic types (positive or negative). There are, however, few frames which are inconsistent with the notion of relational opposites should be in separate frames.
