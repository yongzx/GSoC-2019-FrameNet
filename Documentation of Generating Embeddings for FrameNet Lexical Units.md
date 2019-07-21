# Documentation of Generating Embeddings for FrameNet Lexical Units

I generated the BERT embeddings for FrameNet lexical units using the natural language processing `flair` library. 

#### What is BERT embeddings?
BERT model makes use of Transformer encoder to read text input and generate vector representations of each word in the sentence based on its surrounding words. The encoder reads the entire sequence of words at once, which allows BERT model to learn the context of a word in the sentence. In other words, the vector representation (embedding) of a word depends on context of the sentence. 

For example, we can see that for the same word `"long"`, its BERT embeddings in two different sentences are totally different because the word `"long"` carries different meanings in the two sentences.
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

The BERT embedding of a lexical unit is obtained by averaging the BERT embeddings of the lexical unit appearing in the annotated sentences in FrameNet. If there's no actual text in FrameNet that features the lexical unit, which means that there's no sentence examples of the lexical unit, its embedding will be a zero tensor.

The function below illustrates how the BERT embeddings of lexical units are generated. 

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
