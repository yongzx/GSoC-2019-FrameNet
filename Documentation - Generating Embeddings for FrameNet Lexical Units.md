# Generating Embeddings for FrameNet Lexical Units

I generated the BERT embeddings for FrameNet lexical units using the natural language processing `flair` library. 

#### What is BERT embeddings?
BERT model makes use of Transformer encoder to read text input and generate vector representations of each word in the sentence based on its surrounding words. The encoder reads the entire sequence of words at once, which allows the BERT model to learn the context of a word in the sentence. In other words, the vector representation (embedding) of a word depends on the meaning of the sentence. For example, we can see that for the same word "long", its BERT embeddings in two different sentences are different.

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

#### Generate BERT embeddings for Lexical Units
Lexical units are the combination of lemmas and their part-of-speech tags. For example, "run.v", "long.v", "long.adj", etc. They are words that evoke a semantic frame (i.e., a description of a type of event, relation, or entity and the participants in it.) from FrameNet 1.7. 

The BERT embedding of a lexical unit is obtained by averaging the BERT embeddings of the lexical unit appearing in the annotated sentences in FrameNet. If there are no sentence examples of the lexical unit, the embedding of the lexical unit will be a zero tensor.

The function `create_fn_LU_embeddings` below illustrates how the BERT embeddings of lexical units are generated. It is in the file `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/create_embeddings.py`.

```python
import torch
import pickle
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from nltk.corpus import framenet as fn
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

**Output**

The pickled file `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/lus_fn1.7_bert.p` stores the dictionary of lexical units' ids mapped to their BERT embeddings.
```
...
5139: tensor([-1.1780,  0.3961,  0.4322,  ..., -0.6934,  0.4036, -1.1020]),
5140: tensor([-0.3504,  0.2287, -0.1355,  ..., -0.4792,  0.0207, -1.0070]),
5142: tensor([-0.2085,  0.2586,  0.4965,  ..., -0.0963,  0.5369, -0.5002]),
5143: tensor([ 0.6830, -0.2432,  0.7989,  ..., -1.0033, -0.1554, -0.6765]),
5144: tensor([0., 0., 0.,  ..., 0., 0., 0.]),
...
```
