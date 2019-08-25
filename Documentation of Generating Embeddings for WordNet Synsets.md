# Documentation of Generating Embeddings for WordNet Synsets

**Prerequisites**: I generated the BERT embeddings for lexical units using the natural language processing `flair` library. (See my [GitHub Documentation](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Generating%20Embeddings%20for%20FrameNet%20Lexical%20Units.md))

#### What is BERT embedding?
BERT model makes use of Transformer encoder to read text input and generate vector representations of each word in the sentence based on its surrounding words. The encoder reads the entire sequence of words at once, which allows the BERT model to learn the context of a word in the sentence. In other words, the vector representation (embedding) of a word depends on the meaning of the sentence. For example, we can see that for the same word "long", its BERT embeddings in two different sentences are different.

```python
from flair.data import Sentence
from flair.embeddings import BertEmbeddings

embedding = BertEmbeddings()
sent_1 = Sentence("The rope is long .")
sent_2 = Sentence("I long for love .")
embedding.embed(sent_1)
embedding.embed(sent_2)

for token in sent_1.tokens:
    if token.text == "long":
        print(token.embedding) #tensor([ 0.5141,  0.2149,  0.8124,  ..., -0.1647,  0.2265,  0.0461])

for token in sent_2.tokens:
    if token.text == "long":
        print(token.embedding) # tensor([ 0.4928,  0.8675,  0.7306,  ..., -0.5213,  0.0706, -0.0010])
```

#### Generating BERT Embeddings for WordNet Synsets
I generated the BERT embeddings for WordNet synsets, obtained from the `nltk` library, using the natural language processing `flair` library. 

WordNet is the lexical database i.e., dictionary for the English language, specially designed for natural language processing. Nouns, verbs, adjectives, and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked through conceptual-semantic and lexical relations.

The BERT embedding of a synset is obtained by averaging the BERT embeddings of the synset's lemmas appearing in the annotated sentences in WordNet. The sentences will be preprocessed by tokenization and lemmatization. If the synset does not have any example sentence, it will not be included in the hash-map that maps the synset's name to its BERT embeddings. In other words, not all the synsets from WordNet have their respective embeddings (not even a zero tensor). 

The function (`generate_embeddings_synsets`) below illustrates how the BERT embeddings of WordNet synsets are generated. It is in the file `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/deployed_antonym_{1/2/3/4/5}.py`.

```python
import torch
import pickle
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from flair.data import Sentence
from flair.embeddings import BertEmbeddings


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

**Output**
The pickled file `/home/zxy485/zxy485gallinahome/week9-12/antonym-detection/synsets_wn_bert.p` stores the dictionary of WordNet synsets mapped to their BERT embeddings.

```
...
'cranial.a.01': tensor([-0.5483,  0.1041,  0.2834,  ..., -0.5726,  0.4014, -0.3364]),
'crank.v.02': tensor([ 0.6688, -0.1836, -0.2802,  ...,  0.3593, -0.1630,  0.1890]),
'crannied.a.01': tensor([ 0.3682, -0.1359, -0.7155,  ..., -0.3763,  0.1030, -0.8874]),
'crape.v.01': tensor([ 1.2026,  0.6097,  0.6207,  ..., -0.4173,  0.4006, -1.8132]),
'crapshoot.n.01': tensor([ 0.4087,  0.4701,  0.5297,  ..., -0.2817, -0.2754, -0.5573]),
...
```
