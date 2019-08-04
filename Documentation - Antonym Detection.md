# Documentation - Antonym Detection

All the necessary files reside in the folder `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection`. 

The `sbatch` script used to identify antonyms within FrameNet 1.7 is as followed:

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

The list of pairs of antonymous lexical units within the same frame are stored in the five pickled files - `potential_antonyms_cosine_sim_with_dep_1.p`, `potential_antonyms_cosine_sim_with_dep_2.p`, `potential_antonyms_cosine_sim_with_dep_3.p`, `potential_antonyms_cosine_sim_with_dep_4.p`, `potential_antonyms_cosine_sim_with_dep_5.p`.

---

## Challenges During Deployment and Solutions

**Long Processing Time** - To hasten the process of identifying all the antonyms, I used the parallel processing paradigm by having five Python scripts (`deployed_antonym_1.py`, `deployed_antonym_2.py`, `deployed_antonym_3.py`, `deployed_antonym_4.py`, `deployed_antonym_5.py`) for processing all the frames within FrameNet where each script handles 250 frames. 

**Error** (`ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`) - This error is caused by my previous local installation of `nvidia/apex` for BERT processing using the `pytorch-pretrained-bert` library. Even when `import apex` is not called in the script, if `apex` is installed and detect, the Python script will run into the `ModuleNotFoundError` because first of all, the Python script is not run on a GPU node with CUDA. Second, the deployed Singularity container built on my personal MacOS environment doesn't have the `apex` library installed with CUDA extensions.

---

## How It Works

### 1. Generate BERT embeddings for Lexical Units

Lexical units are the combination of lemmas and their part-of-speech tags. For example, “run.v”, “long.v”, “long.adj”, etc. They are words that evoke a semantic frame (i.e., a description of a type of event, relation, or entity and the participants in it.) from FrameNet 1.7.

The BERT embedding of a lexical unit is obtained by averaging the BERT embeddings of the lexical unit appearing in the annotated sentences in FrameNet. If there’s no actual text in FrameNet that features the lexical unit, which means that there’s no sentence examples of the lexical unit, its embedding will be a zero tensor.

### 2. Generate BERT embeddings for WordNet Synsets

WordNet is the lexical database i.e. dictionary for the English language, specifically designed for natural language processing. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations.

The BERT embedding of a synset is obtained by averaging the BERT embeddings of the synset’s lemmas appearing in the annotated sentences in WordNet. The sentences will have to be preprocessed by tokenization and lemmatization. If the synset does not have any example sentence, it will not be included in the hash-map that maps the synset’s name to its BERT embeddings. In other words, not all the synsets from WordNet have their respective embeddings (not even a zero tensor).

### 3. Generating Training and Testing Dataset from WordNet

Dataset of antonymous pairs of lemmas are generated to train a decision tree classifier. The dataset consists of antonymous pairs of synsets and non-antonymous pairs of synsets, all of which are transformed into cosine-similarity of each other. Non-antonymous pairs of synsets are obtained by randomly pairing two synsets which are not antonymous to each other. The number of non-antonymous pairs of synsets is adjusted such that it matches the number of the antonymous pairs. 

### 4. Training the Decision Tree Classifier

Subsequently, I train the decision tree classifier (which uses CART algorithm) from `sklearn` library with the dataset generated. The antonyms and non-antonyms are split by a ratio of 0.33 into training and testing dataset.

### 5. Testing the Classifier

### 5A - POS

Without factoring POS into account, there are 3322 antonyms pairs in total and 3322 self-generated non-antonym pairs. The accuracy of the classifier is 0.76. 

After factoring POS into account, there are 2320 antonyms pairs in total and 2320 self-generated non-antonym pairs. In other words, each pair of the antonymous and non-antonymous pairs of synsets share the same POS. The accuracy of the classifier is 0.83. 

**5B - Dependency-Parsing**

After considering the syntactic relations, which are the type of dependency relations and the level of the node in the parse tree, the accuracy increases to 0.88. 

### 6. Applying Classifier to FrameNet

For each frame in FrameNet, all of its lexical units with the same POS tag are grouped in combinations of pairs. To reduce the number of false positives, the input to the classifier are [x1, x2] where: 

- x1 is a list consists of (in the following order) the cosine similarity between the two lexical units, the average type of depedency relation of the first lexical unit (in integer), the average level of the first lexical unit node in the parse trees of the exemplar sentences, the average type of depedency relation of the second lexical unit (in integer), and the average level of the second lexical unit node in the parse trees of the exemplar sentences.
- x2 is a list consists of (in the following order) the cosine similarity between the two lexical units, the average type of depedency relation of the second lexical unit (in integer), the average level of the second lexical unit node in the parse trees of the exemplar sentences, the average type of depedency relation of the first lexical unit (in integer), and the average level of the first lexical unit node in the parse trees of the exemplar sentences.

The returned result of the function below is a list of tuples of antonymous pairs in the format of (L1, L2, id(L1), id(L2)) where L1 and L2 are antonymous lexical units and id() is a function that maps the lexical unit to its respective ID in FrameNet.

---

## Overall Implementation Details (`Deployed_Antonym_{1/2/3/4/5}.py`)

### 1. Generate BERT embeddings for Lexical Units

Output: A pickled file that saves the mapping of the IDs of lexical units to their BERT embeddings

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

Output: A pickled file that saves the mapping of the names of the WordNet synsets (e.g. `able.a.01`) to their BERT embeddings.

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

Output: The pickled file that contains the list of tuples of antonymous pairs in the format of (L1, L2, id(L1), id(L2)) where L1 and L2 are antonymous lexical units and id() is a function that maps the lexical unit to its respective ID in FrameNet.

- Only pairs of lexical units which are in the same frame and have the same POS are classified.
- The function `get_dep_relations` returns a mapping of each token in the tokenized sentence to its level and dependency relations in the dependency-parsed tree. This uses the API of the UDPipe. An example would be `{'abandon.VERB': [(0, 'root'), (2, 'conj'), (2, 'acl'), (0, 'root'), (0, 'root'), (1, 'parataxis'), (0, 'root'), (2, 'advcl'), (2, 'conj'), (1, 'conj'), (0, 'root'), (1, 'ccomp'), .. (0, 'root')], '..PUNCT': [(1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), (1, 'punct'), .. ], 'plan.NOUN': [(1, 'obj'), (1, 'obj'), (2, 'obj'), (1, 'obj')],  ... }`
- The function `process_dep_relations` takes in the hash-map returned by `get_dep_relations` and average the level and dependency relations (in integer representation) of each token in the tokenized sentence. The example would be `{'abandon.VERB': (0.9130434782608695, 17.869565217391305), '..PUNCT': (1.0, 32.0), 'plan.NOUN': (1.25, 28.0), ...}`
- `cos(X, Y)` generates the cosine similarity of two word embeddings `X` and `Y`.
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
