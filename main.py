import numpy as np
from pathlib import Path
import tiktoken

current_path = Path.cwd()
text_file = current_path / 'stories.txt'

# gpt 4o encoding
encoding = tiktoken.get_encoding("o200k_base")

def tokenize(text_file_input=text_file):
    with open(text_file_input, 'r') as file:
        text = file.read()
    return encoding.encode(text)

tokens = tokenize(text_file)
token_readable = [encoding.decode([t]) for t in tokens]

#print(len(tokens))                                   # how many tokens the whole file becomes
#print(token_readable[:200])   # decode each id back to its text, to see the pieces

used_ids = sorted(set(tokens))                              # the different ids that appear, smallest first
to_small = {big: small for small, big in enumerate(used_ids)} # map tiktoken ids to our smaller ids
token_ids = np.array([to_small[t] for t in tokens])                # the whole file in our ids
vocab_size = len(used_ids)                                  

#how many numbers describe token
embedding_size = 128
#how many tokens it can see at once
context_length = 64

random_generator = np.random.default_rng(0)

#it is vocab size bc it is all the tokens
token_embedding_table = random_generator.normal(0, 0.02, (vocab_size, embedding_size))
#it is context length bc it embeds whatever it can remeber in the 64 token thingy yk 
position_embedding_table = random_generator.normal(0, 0.02, (context_length, embedding_size))

#we have our tiny window of tokens, and we want to get their embeddings and add the position embeddings
window = token_ids[:context_length]
vectors = token_embedding_table[window] + position_embedding_table[:len(window)]
#print(vocab_size, vectors.shape)

layer_norm_scale = np.ones(embedding_size)
layer_norm_shift = np.zeros(embedding_size)

#we layer norm our vectors, so they have mean 0 and std 1, then scale and shift them
def layer_norm(vectors, scale, shift, tiny=1e-5):
    average = vectors.mean(axis=-1, keepdims=True)
    variance = vectors.var(axis=-1, keepdims=True)
    spread = np.sqrt(variance + tiny)
    normalized = (vectors - average) / spread
    return normalized * scale + shift

normalized_vectors = layer_norm(vectors, layer_norm_scale, layer_norm_shift)

#this legit just matrix multiplies the vectors by the weights and adds the bias, which is what a linear layer does
def linear(vectors, weights, bias):
    return vectors @ weights + bias

qkv_weights = random_generator.normal(0, 0.02, (embedding_size, 3 * embedding_size))
qkv_bias = np.zeros(3 * embedding_size)

queries_keys_values = linear(normalized_vectors, qkv_weights, qkv_bias)
queries, keys, values = np.split(queries_keys_values, 3, axis=-1)

#so there is basically 128 embeddings
#each head gets to look at 32 of those embeddigns in this case
#the heads learn different ways to look at each 32 group
#every token gets it own 32 numbers for each head


#splits up our input of the the different heads so the embeddings can be processed in parallel. Each head will have its own set of queries, keys, and values.
number_of_heads = 4
head_size = embedding_size // number_of_heads    # 128 / 4 = 32


#deadass just shifts around so qkv is the 4 heads, so each head has its own queries, keys, and values. It is a way to organize the data for multi-head attention.
#reshape: (64, 128) -> (64, 4, 32)   each token's 128 numbers become 4 rows of 32
#transpose: (64, 4, 32) -> (4, 64, 32)   group by head first, so each head is its own (64, 32) grid

#now we have the heads w/ the query keys and values
queries = queries.reshape(len(window), number_of_heads, head_size).transpose(1, 0, 2)
keys = keys.reshape(len(window), number_of_heads, head_size).transpose(1, 0, 2)
values = values.reshape(len(window), number_of_heads, head_size).transpose(1, 0, 2)



print(queries, keys, values)


#next step: score every token's query against every token's key, separately in each head
#a dot product (multiply the 32 matching numbers, add them up) is big when "what I'm looking for" matches "what you have"
#keys.transpose(0, 2, 1) flips each head's keys from (64, 32) to (32, 64) so the matrix multiply lines up
#(4, 64, 32) @ (4, 32, 64) -> (4, 64, 64): for each head, row i column j = how much token i cares about token j
#divide by sqrt(32) because adding up 32 products makes big numbers, and big numbers make softmax too extreme later
scores = queries @ keys.transpose(0, 2, 1) / np.sqrt(head_size)

#this shape represents the scores for each head, for each token in the window, against every other token in the window.
print(scores.shape)          # (4, 64, 64)
print(scores[0, :3, :3])     # head 0: how much 'Once', ' upon', ' a' care about each other (random for now)



