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



#hide the future: a token can only look at itself and the tokens BEFORE it, or it could just peek at the answer
#
#np.ones((64, 64), dtype=bool) makes a 64 by 64 grid filled with True (dtype=bool means True/False instead of 1.0)
#   with 4 tokens instead of 64:   [[T, T, T, T],
#                                   [T, T, T, T],
#                                   [T, T, T, T],
#                                   [T, T, T, T]]
#
#np.triu(grid, k=1) keeps only the part ABOVE the diagonal and turns everything else into False
#triu = "triangle upper". k=1 means start one spot right of the diagonal, so the diagonal itself becomes False too
#(a token is allowed to look at itself)
#   row = the token doing the looking, column = the token being looked at
#                                  [[F, T, T, T],    token 0 can't see tokens 1, 2, 3
#                                   [F, F, T, T],    token 1 can't see tokens 2, 3
#                                   [F, F, F, T],    token 2 can't see token 3
#                                   [F, F, F, F]]    token 3 can see everything before it
#so True = "this spot is in the future, hide it"
future = np.triu(np.ones((len(window), len(window)), dtype=bool), k=1)

#np.where(condition, a, b) goes through every spot: if condition is True there, use a. if False, keep b
#so every future spot becomes -1e9 (basically negative infinity) and every other spot keeps its real score
#   scores row 0: [0.02, 0.03, 0.01, ...]  ->  [0.02, -1e9, -1e9, ...]
#future is (64, 64) and scores is (4, 64, 64), so numpy uses the same future grid for all 4 heads
#next step softmax turns -1e9 into a weight of 0, so those tokens get completely ignored
scores = np.where(future, -1e9, scores)

print(future[:4, :4])        # the 4x4 example above
print(scores[0, :4, :4])     # head 0: the future spots are now -1e9

#next step: softmax. turns each row of scores into weights that are all positive and add up to 1
#so each token ends up with "how much should I listen to each token before me", like 70% this one, 20% that one...
#
#how it does it, with one row of 3 scores [2.0, 1.0, 0.1]:
#   1. e^x on every score         [7.39, 2.72, 1.11]   e^x is always positive, and bigger scores get WAY bigger
#   2. divide by the total 11.22  [0.659, 0.242, 0.099]  now they add up to 1
#a future spot is -1e9, and e^-1e9 is 0, so hidden tokens get exactly 0 weight
#
#subtracting the row's biggest score first gives the exact same answer (it cancels out when you divide)
#but stops e^x from blowing up to infinity if a score is big. axis=-1 = do it across each row
def softmax(scores):
    exps = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)

#(4, 64, 64): for each head, row i = how much token i listens to each token 0..63
weights = softmax(scores)

print(softmax(np.array([2.0, 1.0, 0.1])).round(3))   # the example above: [0.659 0.242 0.099]
print(weights[0, :4, :4].round(3))   # head 0: row 0 is [1, 0, 0, 0] because token 0 can only see itself
print(weights[0].sum(axis=-1)[:4])   # every row adds up to 1

#next step: use the weights to add up the values. this is where tokens actually pass info to each other
#each token's new vector = (weight on token 0 * value of token 0) + (weight on token 1 * value of token 1) + ...
#so tokens it listens to a lot count a lot, and hidden future tokens (weight 0) add nothing
#
#example for ' upon' (token 1) in head 0, weights [0.484, 0.516, 0, 0, ...]:
#   gathered = 0.484 * value of 'Once' + 0.516 * value of ' upon'
#
#one matrix multiply does all the multiplying AND adding, for every token in every head at once:
#(4, 64, 64) @ (4, 64, 32) -> (4, 64, 32): each head, each token, 32 numbers of info gathered from earlier tokens
#
#why add up the VALUES and not just the original embeddings (normalized_vectors)?
#because "how do I find this token" and "what should this token pass along" are two different jobs
#   query = what am I looking for       key = what makes me findable       value = what I hand over once found
#
#real example: "Tom lost his red ball. He was very sad."
#   when the model is at "He", one head wants to find who "He" is
#   - "He"'s QUERY says something like "looking for a person's name that came before"
#   - "Tom"'s KEY says "I'm a name, I'm a person". that's what makes the match happen (big score)
#   - "Tom"'s VALUE is what actually gets copied into "He": "male character, the one this story is about"
#     that's the stuff that helps guess what comes next ("was", "sad", later "his")
#   another head at "He" might be looking for adjectives, match "red", and its value passes "red"-ness instead
#
#if we added up the original embeddings instead:
#   - "He" would get ALL of Tom's raw 128 numbers, including stuff it doesn't need (its position, how it's spelled...)
#   - every head would pass the exact same thing. 4 heads would just be 4 different mixes of the same vectors,
#     instead of 4 different kinds of info. with values, each head has its own value weights (its own 32 columns
#     of qkv_weights), so each head pulls out a different 32-number summary of the same token
#   - each head only has 32 numbers to work with, so it can't carry all 128 anyway. the value weights learn
#     to squeeze in just what that head's job needs
#
#why this approach won: the early attention models (around 2014, for translation) did basically that, weighting
#the original hidden vectors directly. the 2017 "Attention Is All You Need" paper (the transformer, where GPT
#comes from) gave each head its own learned query, key AND value matrices. having separate heads was clearly
#better in their tests (one big head scored noticeably worse at translating), and separate heads only make sense
#if each one gets its own values. it costs almost nothing: the value part is just 128 more columns in qkv_weights
gathered = weights @ values

print(gathered.shape)    # (4, 64, 32)
#check token 1 in head 0 by hand: should match the matrix multiply exactly
by_hand = weights[0, 1, 0] * values[0, 0] + weights[0, 1, 1] * values[0, 1]
print(np.allclose(gathered[0, 1], by_hand))   # True

#next step: glue the 4 heads back together, then one more matrix. this finishes attention
#this is the opposite of when we cut into heads:
#transpose: (4, 64, 32) -> (64, 4, 32)   group by token again, so each token has its 4 head results
#reshape: (64, 4, 32) -> (64, 128)       put each token's 4 results side by side: head 0's 32, then head 1's 32, ...
gathered = gathered.transpose(1, 0, 2).reshape(len(window), embedding_size)

#then one more linear layer (128 -> 128). right now the 4 heads' results are just sitting next to each other,
#this lets every output number mix info from all 4 heads
output_weights = random_generator.normal(0, 0.02, (embedding_size, embedding_size))
output_bias = np.zeros(embedding_size)
attention_output = linear(gathered, output_weights, output_bias)

print(attention_output.shape)   # (64, 128): back to one 128-number vector per token, same shape we started with

#next step: add attention's result onto the original vectors (the "residual")
#we ADD instead of replacing, so each token keeps what it already knew and adds what it just learned from the others
#   ' He' before: "the word He, at spot 6"     after: "the word He, at spot 6, + it means Tom"
#it also makes training way easier: when training works backward, the + gives it a straight path back through
#every block, so the early blocks still get useful nudges even with lots of blocks stacked on top
vectors = vectors + attention_output

#next step: layer norm again before the mlp, same function, but its own scale and shift
layer_norm2_scale = np.ones(embedding_size)
layer_norm2_shift = np.zeros(embedding_size)
normalized_vectors = layer_norm(vectors, layer_norm2_scale, layer_norm2_shift)

#next step: the mlp (a mini neural network) part 1: widen each token from 128 numbers to 512 (4 times wider)
#attention was tokens talking to each other. the mlp is each token thinking on its own, no looking at other tokens
#the extra room lets it check for lots of patterns at once, each of the 512 numbers is like one "detector"
#   (e.g. one might end up lighting up for "this is a name", another for "a sentence is about to end")
#
#how the expanding actually works:
#expand_weights is a (128, 512) grid. think of it as 512 columns, and each column is 128 numbers long
#each column is one "detector", and each of the 512 new numbers comes from one column:
#   new number 0 = token's number 0 * column 0's number 0 + token's number 1 * column 0's number 1 + ... (all 128)
#   new number 1 = the same thing but with column 1
#   ... all the way to new number 511
#that multiply-and-add is a dot product, which is big when the token's 128 numbers "line up" with the column's,
#so each new number is basically "how much does this token match detector #n's pattern"
#
#tiny version with 2 numbers per token instead of 128, widened to 4 (4x, same as us):
#   token = [1.0, 2.0]
#   expand_weights = [[ 1,  0, -1,  2],     (2, 4): 4 columns = 4 detectors, each 2 numbers long
#                     [ 0,  1,  1, -1]]
#   detector 0: 1*1   + 2*0    =  1.0
#   detector 1: 1*0   + 2*1    =  2.0
#   detector 2: 1*(-1) + 2*1   =  1.0
#   detector 3: 1*2   + 2*(-1) =  0.0     -> wide = [1.0, 2.0, 1.0, 0.0]
#so 2 numbers became 4, and each of the 4 is a different mix of the same 2
#
#why 4x wider: nothing magic, the original transformer used 4x and everyone kept it. more detectors means more
#patterns it can check for at once. the weights start random, and training is what makes each column
#turn into a useful pattern
#
#then gelu keeps the detectors that are positive (they "fired") and squashes the negative ones to about 0,
#and the shrink matrix (512, 128) goes the other way: each of its 512 rows is 128 numbers, a "message" for one
#detector, and each fired detector adds its message onto the token, scaled by how hard it fired
#   e.g. a detector that fires on "a name came right before 'went'" might add "a place is coming next"
expand_weights = random_generator.normal(0, 0.02, (embedding_size, 4 * embedding_size))
expand_bias = np.zeros(4 * embedding_size)
wide = linear(normalized_vectors, expand_weights, expand_bias)    # (64, 512)

#next step: gelu, the bend. positive numbers pass through, negative ones get squashed to about 0
#   gelu([-2, 0, 2]) = [-0.045, 0, 1.955]
#so a detector that's positive "fires", and one that's negative stays quiet
#why it's needed: two matrix multiplies in a row with nothing between them are the same as one matrix multiply,
#so without the bend the mlp could only learn straight-line rules. the bend is what lets it learn "if this AND that"
#(0.7978845608 is sqrt(2/pi) and 0.044715 is a number from the gelu paper, it's a smooth version of "max(0, x)")
def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

bent = gelu(wide)                                                  # (64, 512)

#next step: the mlp part 2: shrink back down from 512 to 128, so it fits back onto the token's vector
#this mixes the detectors that fired into the 128 numbers the rest of the model uses
shrink_weights = random_generator.normal(0, 0.02, (4 * embedding_size, embedding_size))
shrink_bias = np.zeros(embedding_size)
mlp_output = linear(bent, shrink_weights, shrink_bias)             # (64, 128)

#next step: add the mlp's result on top too (the 2nd residual), same reason as before
vectors = vectors + mlp_output

#that's one whole block done:
#   layer norm -> attention -> add it on -> layer norm -> mlp -> add it on
#same shape in, same shape out, so the next block can take these vectors and do the same thing again
print(gelu(np.array([-2.0, 0.0, 2.0])).round(3))   # [-0.045  0.     1.955]
print(wide.shape, mlp_output.shape, vectors.shape)  # (64, 512) (64, 128) (64, 128)
