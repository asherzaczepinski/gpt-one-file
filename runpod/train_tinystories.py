import os
import sys
import time
import pickle
import math
import numpy as np      # numpy stays for the cpu jobs: tokenizing, picking random windows, rolling the dice when writing

#the gpu version: xp is cupy (numpy, but every array lives on the nvidia gpu and every math step runs there).
#if cupy isn't installed (like on a mac), xp is plain numpy, so this same file still runs anywhere
try:
    import cupy as xp
    import cupyx
    on_gpu = True
except ImportError:
    xp = np
    on_gpu = False
#gpus are many times faster at 32-bit numbers than 64-bit, so float32 on the gpu (float64 on the cpu)
dtype = xp.float32 if on_gpu else np.float64
#np.add.at on the cpu, cupyx.scatter_add on the gpu: same job (add into rows, a row used twice gets both pushes)
scatter_add = cupyx.scatter_add if on_gpu else np.add.at

#copy an array from the gpu back to the cpu (for saving, and for picking the next token)
def to_cpu(a):
    return a.get() if on_gpu else a
from pathlib import Path
import tiktoken

#main_functionized_cuda.py, scaled up to train on a RunPod GPU: lots more story text (TinyStories) and a bigger model.
#started by runpod/train_on_runpod.sh. the model code (forward, backward) is exactly main_functionized_cuda.py's
#main.py, but the model is one function (forward) so it's ready for backprop
#every learned number lives in one dict (params) and every step saves a "cache" for backprop to use later

current_path = Path(__file__).parent
#the text to train on: on the pod this is a few hundred MB of TinyStories (set by train_on_runpod.sh)
text_file = Path(os.environ.get("TEXT_FILE", current_path.parent / "stories.txt"))

# gpt 4o encoding
encoding = tiktoken.get_encoding("o200k_base")

#tokenize a big file in pieces: read ~4 MB at a time, tokenize them on several cpu threads at once, and turn each
#piece into a small numpy array right away (a python list of 100 million tokens would need ~4 GB of memory)
def tokenize(path):
    pieces, chunk = [], []
    with open(path, 'r', errors='ignore') as file:
        while True:
            block = file.read(4_000_000)
            if not block:
                break
            chunk.append(block)
            if len(chunk) == 16:
                pieces += [np.array(t, dtype=np.int32) for t in encoding.encode_ordinary_batch(chunk, num_threads=os.cpu_count())]
                chunk = []
    if chunk:
        pieces += [np.array(t, dtype=np.int32) for t in encoding.encode_ordinary_batch(chunk, num_threads=os.cpu_count())]
    return np.concatenate(pieces)

start_time = time.time()
tokens = tokenize(text_file)

#renumber to the ids actually used, same idea as before, but with numpy so it's fast for 100 million tokens:
#np.unique gives the different ids (sorted) AND every token's position in that list (= our small id)
used_ids, token_ids = np.unique(tokens, return_inverse=True)
used_ids = [int(t) for t in used_ids]
to_small = {big: small for small, big in enumerate(used_ids)}
token_ids = token_ids.astype(np.int32)
vocab_size = len(used_ids)
print(f"{len(token_ids):,} tokens, {vocab_size:,} different ones, tokenized in {time.time() - start_time:.0f} s", flush=True)

#how many numbers describe token
embedding_size = int(os.environ.get("EMBEDDING_SIZE", 256))
#how many tokens it can see at once
context_length = int(os.environ.get("CONTEXT_LENGTH", 128))

#splits up our input of the the different heads so the embeddings can be processed in parallel. Each head will have its own set of queries, keys, and values.
number_of_heads = int(os.environ.get("NUMBER_OF_HEADS", 8))
head_size = embedding_size // number_of_heads    # 128 / 4 = 32

#how many blocks stacked: block 1 feeds block 2 feeds ... block 6
number_of_blocks = int(os.environ.get("NUMBER_OF_BLOCKS", 6))

random_generator = np.random.default_rng(0)


#all the numbers the model learns go in this one dict, so backprop can go through and nudge every one of them
params = {}
#it is vocab size bc it is all the tokens
params["token_embedding_table"] = random_generator.normal(0, 0.02, (vocab_size, embedding_size))
#it is context length bc it embeds whatever it can remeber in the 64 token thingy yk
params["position_embedding_table"] = random_generator.normal(0, 0.02, (context_length, embedding_size))
#every block gets its own weights: block1.qkv_weights, block2.qkv_weights, ...
for i in range(number_of_blocks):
    b = f"block{i + 1}."
    params[b + "layer_norm_scale"] = np.ones(embedding_size)
    params[b + "layer_norm_shift"] = np.zeros(embedding_size)
    params[b + "qkv_weights"] = random_generator.normal(0, 0.02, (embedding_size, 3 * embedding_size))
    params[b + "qkv_bias"] = np.zeros(3 * embedding_size)
    params[b + "output_weights"] = random_generator.normal(0, 0.02, (embedding_size, embedding_size))
    params[b + "output_bias"] = np.zeros(embedding_size)
    params[b + "layer_norm2_scale"] = np.ones(embedding_size)
    params[b + "layer_norm2_shift"] = np.zeros(embedding_size)
    params[b + "expand_weights"] = random_generator.normal(0, 0.02, (embedding_size, 4 * embedding_size))
    params[b + "expand_bias"] = np.zeros(4 * embedding_size)
    params[b + "shrink_weights"] = random_generator.normal(0, 0.02, (4 * embedding_size, embedding_size))
    params[b + "shrink_bias"] = np.zeros(embedding_size)
#after the last block: one more layer norm, then the head turns 128 numbers into a score for every token
params["final_layer_norm_scale"] = np.ones(embedding_size)
params["final_layer_norm_shift"] = np.zeros(embedding_size)
params["head_weights"] = random_generator.normal(0, 0.02, (embedding_size, vocab_size))
params["head_bias"] = np.zeros(vocab_size)

#copy every learned number into the gpu's memory (as float32 there). from here on, params lives on the gpu
params = {name: xp.asarray(a, dtype=dtype) for name, a in params.items()}


#cache = stuff we save on the way forward so backprop can use it later
#backprop goes through every step in reverse, and each step needs to know what went into it

#we layer norm our vectors, so they have mean 0 and std 1, then scale and shift them
def layer_norm(vectors, scale, shift, tiny=1e-5):
    average = vectors.mean(axis=-1, keepdims=True)
    variance = vectors.var(axis=-1, keepdims=True)
    spread = xp.sqrt(variance + tiny)
    normalized = (vectors - average) / spread
    cache = (normalized, spread, scale)
    return normalized * scale + shift, cache

#this legit just matrix multiplies the vectors by the weights and adds the bias, which is what a linear layer does
def linear(vectors, weights, bias):
    cache = (vectors, weights)
    return vectors @ weights + bias, cache

#softmax. turns each row of scores into weights that are all positive and add up to 1
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
    exps = xp.exp(scores - scores.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)

#gelu, the bend. positive numbers pass through, negative ones get squashed to about 0
#   gelu([-2, 0, 2]) = [-0.045, 0, 1.955]
#so a detector that's positive "fires", and one that's negative stays quiet
#why it's needed: two matrix multiplies in a row with nothing between them are the same as one matrix multiply,
#so without the bend the mlp could only learn straight-line rules. the bend is what lets it learn "if this AND that"
#(0.7978845608 is sqrt(2/pi) and 0.044715 is a number from the gelu paper, it's a smooth version of "max(0, x)")
def gelu(x):
    return 0.5 * x * (1 + xp.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

#the whole model: a window of token ids in, a score for every possible next token out
def forward(windows):
    #windows is (B, 64): B windows at once (a gpu is only fast when it gets big chunks of work). every shape has a B in front now
    B, T = windows.shape
    #what needs a cache: any step that multiplies by something (linear, layer norm, the q@k and weights@values, gelu),
    #because backprop needs to know what got multiplied. what doesn't: plain adds (the residuals), reshapes and transposes, since backprop just passes the gradient straight through or shuffles it back
    #we have our tiny window of tokens, and we want to get their embeddings and add the position embeddings
    vectors = params["token_embedding_table"][windows] + params["position_embedding_table"][:T]
    #stores stuff for backprop
    block_caches = []
    for i in range(number_of_blocks):
        b = f"block{i + 1}."
        cache = {}    # everything this block's backprop will need
        normalized_vectors, cache["layer_norm"] = layer_norm(vectors, params[b + "layer_norm_scale"], params[b + "layer_norm_shift"])
        queries_keys_values, cache["qkv"] = linear(normalized_vectors, params[b + "qkv_weights"], params[b + "qkv_bias"])
        queries, keys, values = xp.split(queries_keys_values, 3, axis=-1)

        #so there is basically 128 embeddings
        #each head gets to look at 32 of those embeddigns in this case
        #the heads learn different ways to look at each 32 group
        #every token gets it own 32 numbers for each head

        #deadass just shifts around so qkv is the 4 heads, so each head has its own queries, keys, and values. It is a way to organize the data for multi-head attention.
        #reshape: (64, 128) -> (64, 4, 32)   each token's 128 numbers become 4 rows of 32
        #transpose: (64, 4, 32) -> (4, 64, 32)   group by head first, so each head is its own (64, 32) grid

        #now we have the heads w/ the query keys and values
        queries = queries.reshape(B, T, number_of_heads, head_size).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, T, number_of_heads, head_size).transpose(0, 2, 1, 3)
        values = values.reshape(B, T, number_of_heads, head_size).transpose(0, 2, 1, 3)

        #score every token's query against every token's key, separately in each head
        #a dot product (multiply the 32 matching numbers, add them up) is big when "what I'm looking for" matches "what you have"
        #keys.transpose(0, 2, 1) flips each head's keys from (64, 32) to (32, 64) so the matrix multiply lines up
        #(4, 64, 32) @ (4, 32, 64) -> (4, 64, 64): for each head, row i column j = how much token i cares about token j
        #divide by sqrt(32) because adding up 32 products makes big numbers, and big numbers make softmax too extreme later
        scores = queries @ keys.transpose(0, 1, 3, 2) / math.sqrt(head_size)

        #hide the future. a token can only look at itself and the tokens BEFORE it
        #otherwise in training it could just peek at the next word instead of learning to guess it
        #np.triu(..., k=1) makes a (64, 64) grid that's True above the diagonal, which is every "later token" spot
        #   token 0 can see [0]    token 1 can see [0, 1]    token 63 can see [0 ... 63]
        future = xp.triu(xp.ones((T, T), dtype=bool), k=1)

        #put -1e9 (basically negative infinity) in every future spot. softmax turns that into a weight of 0 next step
        #the same (64, 64) grid gets used for all 4 heads automatically
        scores = xp.where(future, xp.asarray(-1e9, dtype=scores.dtype), scores)

        #(4, 64, 64): for each head, row i = how much token i listens to each token 0..63
        weights = softmax(scores)

        #use the weights to add up the values. this is where tokens actually pass info to each other
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

        #glue the 4 heads back together, then one more matrix. this finishes attention
        #this is the opposite of when we cut into heads:
        #transpose: (4, 64, 32) -> (64, 4, 32)   group by token again, so each token has its 4 head results
        #reshape: (64, 4, 32) -> (64, 128)       put each token's 4 results side by side: head 0's 32, then head 1's 32, ...
        gathered = gathered.transpose(0, 2, 1, 3).reshape(B, T, embedding_size)

        #then one more linear layer (128 -> 128). right now the 4 heads' results are just sitting next to each other,
        #this lets every output number mix info from all 4 heads
        attention_output, cache["output"] = linear(gathered, params[b + "output_weights"], params[b + "output_bias"])

        #add attention's result onto the original vectors (the "residual")
        #we ADD instead of replacing, so each token keeps what it already knew and adds what it just learned from the others
        #   ' He' before: "the word He, at spot 6"     after: "the word He, at spot 6, + it means Tom"
        #it also makes training way easier: when training works backward, the + gives it a straight path back through
        #every block, so the early blocks still get useful nudges even with lots of blocks stacked on top
        vectors = vectors + attention_output

        #layer norm again before the mlp, same function, but its own scale and shift
        normalized_vectors, cache["layer_norm2"] = layer_norm(vectors, params[b + "layer_norm2_scale"], params[b + "layer_norm2_shift"])

        #the mlp (a mini neural network) part 1: widen each token from 128 numbers to 512 (4 times wider)
        #attention was tokens talking to each other. the mlp is each token thinking on its own, no looking at other tokens
        #the extra room lets it check for lots of patterns at once, each of the 512 numbers is like one "detector"
        #   (e.g. one might end up lighting up for "this is a name", another for "a sentence is about to end")
        #
        #how the widening works: it does NOT copy each number 4 times. if it just made 4 copies, the 512 numbers
        #would hold the exact same info as the 128 and the extra room would be useless
        #expand_weights is (128, 512): think of it as 512 columns, each column 128 numbers long
        #each of the 512 new numbers = all 128 of the token's numbers, each multiplied by that column's 128 weights,
        #then added up (a dot product). so every new number is its OWN mix of ALL 128 numbers
        #
        #tiny version: a token with 2 numbers [1, 2], widened 4x to 8 numbers. expand_weights is (2, 8):
        #   column:            0     1     2     3     4     5     6     7
        #   weight for "1":    1     0     2    -1     0     3     1    -2
        #   weight for "2":    0     1     1     1    -1     0     2     1
        #   new number 0 = 1*1 + 2*0   =  1        new number 4 = 1*0  + 2*(-1) = -2
        #   new number 1 = 1*0 + 2*1   =  2        new number 5 = 1*3  + 2*0    =  3
        #   new number 2 = 1*2 + 2*1   =  4        new number 6 = 1*1  + 2*2    =  5
        #   new number 3 = 1*(-1) + 2*1 = 1        new number 7 = 1*(-2) + 2*1  =  0
        #   -> wide = [1, 2, 4, 1, -2, 3, 5, 0]: 8 different mixes of the same 2 numbers, not 4 copies of them
        #training picks the columns, so each one turns into a useful "detector" pattern. a column's number comes out
        #big when the token's 128 numbers line up with that column, then gelu keeps the ones that came out positive
        #
        #expand_weights is a learned matrix just like every other one in params: it starts as small random numbers,
        #backprop works out its gradient (normalized_vectors.T @ gradient, same (128, 512) shape), and adam nudges it
        #every step. each block has its own (block1.expand_weights ... block6.expand_weights), and shrink_weights
        #(512 -> 128) gets tuned the exact same way
        wide, cache["expand"] = linear(normalized_vectors, params[b + "expand_weights"], params[b + "expand_bias"])    # (64, 512)

        bent = gelu(wide)                                                  # (64, 512)

        #the mlp part 2: shrink back down from 512 to 128, so it fits back onto the token's vector
        #this mixes the detectors that fired into the 128 numbers the rest of the model uses
        mlp_output, cache["shrink"] = linear(bent, params[b + "shrink_weights"], params[b + "shrink_bias"])             # (64, 128)

        #add the mlp's result on top too (the 2nd residual), same reason as before
        vectors = vectors + mlp_output

        #that's one whole block done:
        #   layer norm -> attention -> add it on -> layer norm -> mlp -> add it on
        #same shape in, same shape out, so the next block can take these vectors and do the same thing again

        #the attention and gelu parts don't go through linear/layer_norm, so save what backprop needs from them by hand
        cache["queries"], cache["keys"], cache["values"], cache["weights"], cache["wide"] = queries, keys, values, weights, wide
        block_caches.append(cache)

    #after all 6 blocks: one last layer norm, then the head turns each token's 128 numbers into a score for
    #every one of the 4841 tokens. the scores at spot t are the guess for what comes after token t
    vectors, final_cache = layer_norm(vectors, params["final_layer_norm_scale"], params["final_layer_norm_shift"])
    #picking!!!!
    #head_weights is (128, 4841): one column per token in the vocab, each column 128 numbers long
    #each token's score = its vector · that token's column (multiply the 128 matching numbers, add them up)
    #big when they line up, so each column is like a template for "what the vector looks like right before me"
    #   vector at 'Once' · column for ' upon' = 4.1     · column for ' a' = 1.3     · column for ' cat' = -2.0   ... all 4841
    #every spot does this at once: (64, 128) @ (128, 4841) -> (64, 4841)
    #logits = the raw scores, before softmax. one number per possible next token, and it can be anything (negative, 0, 12...)
    #they are NOT probabilities yet: bigger just means "more likely". softmax turns them into probabilities that add to 1
    #   logits for the next token after 'Once':   ' upon': 4.1   ' a': 1.3   ' cat': -2.0   ...
    #   after softmax:                           ' upon': 0.94  ' a': 0.06  ' cat': 0.00   (pretending those 3 are the only tokens)
    #the name comes from "log": softmax does e^score, so a logit is basically the log of an un-scaled probability
    logits, head_cache = linear(vectors, params["head_weights"], params["head_bias"])     # (64, 4841)

    #everything backprop needs, in the order it was made (backprop uses them in reverse: head first, window last):
    #   window        the token ids, so backprop knows which rows of the embedding tables to nudge
    #   block_caches  a list of 6 dicts, one per block, each with that block's layer norms, qkv, heads, weights, mlp stuff
    #   final_cache   just the one last layer norm after block 6 (its normalized numbers, spread and scale, to go back through it)
    #   head_cache    the head linear: the vectors that went in and head_weights, to work out how to nudge head_weights
    #
    #         FORWARD (down)                       saved for later                BACKWARD (up)
    #
    #   window: 64 token ids          ------->  window                       nudge the embedding table rows
    #        |                                                                             ^
    #   embeddings + positions                                                             |
    #        |                                                                             |
    #   block 1                       ------->  block_caches[0] = {...}      back through block 1
    #        |                                                                             ^
    #   block 2                       ------->  block_caches[1] = {...}      back through block 2
    #        |                                                                             ^
    #       ...                                       ...                         ...
    #        |                                                                             ^
    #   block 6                       ------->  block_caches[5] = {...}      back through block 6
    #        |                                                                             ^
    #   final layer norm              ------->  final_cache                  back through final layer norm
    #        |                                                                             ^
    #   head (128 -> 4841)            ------->  head_cache                   back through head
    #        |                                                                             ^
    #   logits (64, 4841)  ->  get_loss  ->  loss                            start here: logits gradient
    #
    #   inside each block_caches[i] dict:
    #      "layer_norm", "qkv", "output", "layer_norm2", "expand", "shrink"     what went into each linear / layer norm
    #      "queries", "keys", "values", "weights", "wide"                       the attention + gelu stuff, saved by hand
    return logits, (windows, block_caches, final_cache, head_cache)





#the simple version of what get_loss does:
#   1. the model gives a probability to EVERY token at every spot. it has no idea which one is right
#   2. WE know the right one, because we have the real text (that's targets)
#   3. so we just look up what probability the model gave the right one, and score it: -log(that probability)
#
#tiny example: pretend the whole vocab is only 4 tokens:   0 = 'Once'   1 = ' upon'   2 = ' a'   3 = ' time'
#   window  = ['Once', ' upon']        targets = [' upon', ' a']  ->  ids [1, 2]
#
#                        'Once'   ' upon'   ' a'   ' time'
#   spot 0 (after 'Once')   0.14     0.61     0.22     0.03     right answer ' upon' (id 1) -> it gave it 0.61
#   spot 1 (after ' upon')  0.10     0.17     0.46     0.28     right answer ' a'    (id 2) -> it gave it 0.46
#
#   loss at spot 0 = -log(0.61) = 0.50
#   loss at spot 1 = -log(0.46) = 0.79
#   loss = average = 0.64          (the example right under get_loss runs this for real)
#
#how wrong it was: for every spot, -log(probability it gave the right next token), averaged
#   sure and right (p = 0.9) -> 0.1        sure and wrong (p = 0.001) -> 6.9
#
#logits: (64, 4841), what forward gave back. row t = the raw scores for what comes after token t
#targets: (64,), the right answers. it's the same window shifted over by one (made at the bottom of the file):
#   window  = tokens 0..63    'Once'   ' upon'  ' a'    ' time' ...
#   targets = tokens 1..64    ' upon'  ' a'     ' time' ','     ...
#so targets[t] is the token that actually came next after window[t]. every spot is its own little question
#
#how it gets -log(probability) without making all the probabilities first:
#probability of the right one = e^(right score) / (e^score1 + e^score2 + ... all 4841)
#take -log of that and it splits into:   log(e^score1 + e^score2 + ...)  -  right score
#
#example, pretending there are only 3 tokens and the right answer is ' upon':
#   logits         ' upon': 4.1   ' a': 1.3   ' cat': -2.0
#   shifted          0.0          -2.8        -6.1       subtract the biggest (4.1) so e^x can't blow up, answer stays the same
#   e^shifted        1.0           0.061       0.002     add them up: 1.063, log of that: 0.061
#   loss = 0.061 - 0.0 (the right one's shifted score) = 0.061      (= -log(0.94), it was 94% sure, nice)
#   if the right answer had been ' a' instead: 0.061 - (-2.8) = 2.861, way worse


#super simple just negative logs the right answer probability to what the model predicted for the next token!
def get_loss(logits, targets):
    #(B, 64, 4841) -> (B*64, 4841): every spot in every window is its own question
    logits, targets = logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    #subtract each row's biggest score from that whole row
    shifted = logits - logits.max(axis=-1, keepdims=True)
    #for each row: e^ every score, add them up, log it. (64,)
    log_of_total = xp.log(xp.exp(shifted).sum(axis=-1))
    #for each row, grab the score of the right answer: row 0 column targets[0], row 1 column targets[1], ... (64,)
    right_answer_scores = shifted[xp.arange(len(targets)), targets]
    #-log(probability of the right answer) at every spot, then average the 64 of them into one number
    return float(xp.mean(log_of_total - right_answer_scores))

#the targets are the same window shifted by one, so every spot's answer is the next token
window = xp.asarray(token_ids[:context_length][None])          # a batch of 1 window, (1, 64), copied to the gpu
targets = xp.asarray(token_ids[1:context_length + 1][None])

logits, cache = forward(window)
loss = get_loss(logits, targets)

print(logits.shape)    # (1, 64, 4841): 1 window, a score for every possible next token at every spot
print(f"loss {loss:.3f}, random guessing would be {np.log(vocab_size):.3f}")   # about the same, nothing is trained yet

#next: backprop. go through forward in reverse using the caches, work out how to nudge every number in params
























































# ---------------- backward pass ----------------
#forward went: tokens -> embeddings -> 6 blocks -> final layer norm -> head -> logits -> loss
#backward goes the exact opposite way, starting from the loss
#at every step it answers: "if this number went up a tiny bit, would the loss go up or down, and by how much?"
#that answer is called the gradient. once every number in params has one, training nudges each number the
#opposite way (downhill), so the loss goes down
#
#the one rule everything below uses (the chain rule): every step takes the gradient of its OUTPUT and turns it
#into the gradient of its INPUT, then hands that back to the step before it. like passing blame back down a line
#
#how blame gets split at the 3 kinds of steps:
#   y = a + b      a and b both get the full gradient of y (that's why residuals are easy)
#   y = x @ w      x gets  (gradient of y) @ w.T      w gets  x.T @ (gradient of y)
#   y = x * w      x gets  (gradient of y) * w        w gets  (gradient of y) * x
#that's why we saved the caches: to split the blame on y = x @ w you need to know what x and w were





#back through linear (y = vectors @ weights + bias)
#   vectors get: gradient @ weights.T    (64, out) @ (out, in) -> (64, in)
#   weights get: vectors.T @ gradient    (in, 64) @ (64, out) -> (in, out), same shape as weights
#   bias gets:   gradient added up over the 64 tokens, since the same bias was added to every token
def linear_backward(gradient, cache):
    vectors, weights = cache
    #flatten the batch so every token in every window is one row: (B, 64, in) -> (B*64, in)
    flat_vectors = vectors.reshape(-1, vectors.shape[-1])
    flat_gradient = gradient.reshape(-1, gradient.shape[-1])
    vectors_gradient = gradient @ weights.T
    weights_gradient = flat_vectors.T @ flat_gradient
    bias_gradient = flat_gradient.sum(axis=0)
    return vectors_gradient, weights_gradient, bias_gradient

#back through layer norm
#scale and shift are easy: shift was added (so it gets the gradient), scale was multiplied by normalized
#the vectors part is trickier: layer norm subtracts the average and divides by the spread, so if you push every
#number up together, or stretch them all together, layer norm just undoes it. so those parts get removed
#from the gradient (the two .mean terms), then it's divided by the spread like forward did
def layer_norm_backward(gradient, cache):
    normalized, spread, scale = cache
    D = normalized.shape[-1]
    scale_gradient = (gradient * normalized).reshape(-1, D).sum(axis=0)     # added up over every token in every window
    shift_gradient = gradient.reshape(-1, D).sum(axis=0)
    g = gradient * scale
    vectors_gradient = (g - g.mean(axis=-1, keepdims=True) - normalized * (g * normalized).mean(axis=-1, keepdims=True)) / spread
    return vectors_gradient, scale_gradient, shift_gradient

#back through gelu: multiply by gelu's slope at each input (how steep the bend was right there)
#positive inputs: slope about 1, so the gradient passes through. very negative: slope about 0, so it stops
def gelu_backward(gradient, x):
    t = xp.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    slope = 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * x ** 2)
    return gradient * slope


def backward(logits, targets, cache):
    windows, block_caches, final_cache, head_cache = cache
    B, T = windows.shape     # the shapes in the comments below are for one window, there's a B in front of all of them
    grads = {}    # will end up with one gradient for every array in params, same shapes

    #step 1: gradient of the loss with respect to the logits (the very last thing forward made)
    #turns out to be super simple: the probabilities it guessed, minus 1 at the right answer
    #   say the right next token is ' upon' and it guessed  ' upon': 0.10   ' a': 0.30   ' cat': 0.60
    #   gradient =                                           0.10 - 1 = -0.90     0.30        0.60
    #   negative = "raise this score", positive = "lower this score", bigger = push harder
    #so it pushes the right answer's score up, and every wrong token down by however much probability it stole
    #then divide by 64 because the loss was an average over the 64 spots
    flat_targets = targets.reshape(-1)                                  # every spot in every window, in one list
    logits_gradient = softmax(logits.reshape(-1, logits.shape[-1]))     # (B*64, 4841): its guessed probabilities
    logits_gradient[xp.arange(len(flat_targets)), flat_targets] -= 1    # minus 1 at each spot's right answer
    logits_gradient = (logits_gradient / len(flat_targets)).reshape(logits.shape)

    #step 2: back through the head (picking). now we know how to nudge each token's column in head_weights,
    #and gradient becomes (64, 128): how each token's final vector should change
    gradient, grads["head_weights"], grads["head_bias"] = linear_backward(logits_gradient, head_cache)

    #step 3: back through the final layer norm
    gradient, grads["final_layer_norm_scale"], grads["final_layer_norm_shift"] = layer_norm_backward(gradient, final_cache)

    #step 4: back through the blocks, block 6 first, down to block 1
    #inside each block it's the forward steps backwards: mlp first, then attention
    for i in reversed(range(number_of_blocks)):
        b = f"block{i + 1}."
        cache = block_caches[i]
        queries, keys, values, weights = cache["queries"], cache["keys"], cache["values"], cache["weights"]

        # --- back through the mlp ---
        #forward was: vectors = vectors + mlp_output. the + means gradient goes straight to the vectors below
        #AND into the mlp. we work out the mlp's part here and add it on at the end
        mlp_gradient, grads[b + "shrink_weights"], grads[b + "shrink_bias"] = linear_backward(gradient, cache["shrink"])
        mlp_gradient = gelu_backward(mlp_gradient, cache["wide"])
        mlp_gradient, grads[b + "expand_weights"], grads[b + "expand_bias"] = linear_backward(mlp_gradient, cache["expand"])
        mlp_gradient, grads[b + "layer_norm2_scale"], grads[b + "layer_norm2_shift"] = layer_norm_backward(mlp_gradient, cache["layer_norm2"])
        gradient = gradient + mlp_gradient     # the two paths from the + join back up

        # --- back through attention ---
        #same deal: vectors = vectors + attention_output, so gradient also goes straight down
        attention_gradient, grads[b + "output_weights"], grads[b + "output_bias"] = linear_backward(gradient, cache["output"])

        #un-glue the heads: (64, 128) -> (64, 4, 32) -> (4, 64, 32), the same cut as forward
        attention_gradient = attention_gradient.reshape(B, T, number_of_heads, head_size).transpose(0, 2, 1, 3)

        #forward was gathered = weights @ values (y = x @ w), so:
        weights_gradient = attention_gradient @ values.transpose(0, 1, 3, 2)   # (4, 64, 64): how each weight should change
        values_gradient = weights.transpose(0, 1, 3, 2) @ attention_gradient   # (4, 64, 32): how each value should change

        #back through softmax: a weight going up means the others in its row go down (they have to add to 1),
        #so subtract off that row's share. then divide by sqrt(32) like forward did
        #hidden future spots had weight 0, so they get 0 gradient here, the mask needs no backward of its own
        scores_gradient = weights * (weights_gradient - (weights_gradient * weights).sum(axis=-1, keepdims=True)) / math.sqrt(head_size)

        #forward was scores = queries @ keys.T, so (y = x @ w again):
        queries_gradient = scores_gradient @ keys                          # (4, 64, 32)
        keys_gradient = scores_gradient.transpose(0, 1, 3, 2) @ queries       # (4, 64, 32)

        #glue each one's heads back to (64, 128) and put q, k, v side by side, the way they came out of qkv
        attention_gradient = xp.concatenate([
            queries_gradient.transpose(0, 2, 1, 3).reshape(B, T, embedding_size),
            keys_gradient.transpose(0, 2, 1, 3).reshape(B, T, embedding_size),
            values_gradient.transpose(0, 2, 1, 3).reshape(B, T, embedding_size),
        ], axis=-1)                                                        # (64, 384)

        attention_gradient, grads[b + "qkv_weights"], grads[b + "qkv_bias"] = linear_backward(attention_gradient, cache["qkv"])
        attention_gradient, grads[b + "layer_norm_scale"], grads[b + "layer_norm_shift"] = layer_norm_backward(attention_gradient, cache["layer_norm"])
        gradient = gradient + attention_gradient    # the two paths join back up again

    #step 5: the embedding tables, the very bottom
    #forward was vectors = token_embedding_table[window] + position_embedding_table[:64]
    #so each token's gradient goes to its row in the token table. np.add.at so a token that shows up twice in the
    #window gets BOTH pushes added up (a plain += would only keep one). rows not in the window get 0
    grads["token_embedding_table"] = xp.zeros_like(params["token_embedding_table"])
    scatter_add(grads["token_embedding_table"], windows, gradient)
    #position t gets the gradient from spot t
    grads["position_embedding_table"] = xp.zeros_like(params["position_embedding_table"])
    grads["position_embedding_table"][:T] = gradient.sum(axis=0)     # position t: added up over every window

    return grads


grads = backward(logits, targets, cache)
print(len(grads), "gradients, one for every array in params:", len(params))



#learning_rate = 0.2
#
#
##need window to change for it to learn
#for step in range(30):
#    logits, cache = forward(window)
#    loss = get_loss(logits, targets)
#    grads = backward(logits, targets, cache)
#    for name in params:
#        params[name] -= learning_rate * grads[name]
#    if step % 5 == 0:
#        print(f"step {step}  loss {loss:.3f}")


















































# ---------------- training, on a time budget ----------------
#the pod costs money every minute, so instead of a fixed number of steps this trains for TRAIN_MINUTES:
#it times the first 30 steps, works out how many steps fit, and sets up the cosine slide to end right at the deadline
train_minutes = float(os.environ.get("TRAIN_MINUTES", 2))
windows_per_step = int(os.environ.get("WINDOWS_PER_STEP", 32))
learning_rate = float(os.environ.get("LEARNING_RATE", 0.001))
warmup_steps = 200
out_folder = Path(os.environ.get("OUT_FOLDER", current_path / "out"))
out_folder.mkdir(exist_ok=True)

#last 1% hidden to test on (with this much text, 1% is still over a million tokens)
cut = int(len(token_ids) * 0.99)
train_ids, test_ids = token_ids[:cut], token_ids[cut:]

def random_window(ids, generator):
    start = generator.integers(0, len(ids) - context_length - 1)
    return ids[start : start + context_length], ids[start + 1 : start + context_length + 1]

def test_loss():
    test_generator = np.random.default_rng(123)
    pairs = [random_window(test_ids, test_generator) for _ in range(50)]
    test_windows = xp.asarray(np.stack([w for w, t in pairs]))
    test_targets = xp.asarray(np.stack([t for w, t in pairs]))
    return get_loss(forward(test_windows)[0], test_targets)

def save(path):
    with open(path, "wb") as file:
        pickle.dump({"params": {name: to_cpu(a) for name, a in params.items()}, "used_ids": used_ids,
                     "settings": {"embedding_size": embedding_size, "context_length": context_length,
                                  "number_of_heads": number_of_heads, "number_of_blocks": number_of_blocks}}, file)

def generate(prompt, new_tokens=120, temperature=0.8, top_k=40, seed=0):
    write_generator = np.random.default_rng(seed)
    ids = [to_small[t] for t in encoding.encode(prompt) if t in to_small]
    for _ in range(new_tokens):
        recent = xp.asarray(np.array([ids[-context_length:]]))
        scores = to_cpu(forward(recent)[0][0, -1]).astype(np.float64) / temperature
        scores = np.where(scores < np.sort(scores)[-top_k], -1e9, scores)
        probabilities = np.exp(scores - scores.max())
        probabilities = probabilities / probabilities.sum()
        ids.append(int(write_generator.choice(vocab_size, p=probabilities)))
    return encoding.decode([used_ids[i] for i in ids])

prompts = ["Once upon a time, there was a", "Tom and Sam went to the park.", "The little cat was sad because",
           "Lily found a shiny key in the garden."]

def one_step(step, total_steps, generator):
    batch = [random_window(train_ids, generator) for _ in range(windows_per_step)]
    windows = xp.asarray(np.stack([w for w, t in batch]))
    targets = xp.asarray(np.stack([t for w, t in batch]))
    logits, cache = forward(windows)
    loss = get_loss(logits, targets)
    grads = backward(logits, targets, cache)
    length = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if length > 1.0:
        for name in grads:
            grads[name] *= 1.0 / length
    if step < warmup_steps:
        step_size = learning_rate * (step + 1) / warmup_steps
    else:
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        step_size = learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))   # slides down to 10%
    for name in params:
        average_gradient[name] = 0.9 * average_gradient[name] + 0.1 * grads[name]
        average_squared[name] = 0.95 * average_squared[name] + 0.05 * grads[name] ** 2
        direction = average_gradient[name] / (1 - 0.9 ** (step + 1))
        size = average_squared[name] / (1 - 0.95 ** (step + 1))
        params[name] -= step_size * direction / (xp.sqrt(size) + 1e-8)
    return loss

print(f"running on {'the GPU (cupy)' if on_gpu else 'the CPU (numpy, no cupy found)'}")
print(f"model: {embedding_size} numbers per token, {number_of_blocks} blocks, {number_of_heads} heads, sees {context_length} tokens back, "
      f"{sum(a.size for a in params.values()):,} learned numbers. {windows_per_step} windows per step, training for {train_minutes} minutes")
print(f"test loss before training: {test_loss():.3f} (random guessing: {math.log(vocab_size):.3f})", flush=True)

average_gradient = {name: xp.zeros_like(a) for name, a in params.items()}
average_squared = {name: xp.zeros_like(a) for name, a in params.items()}
train_generator = np.random.default_rng(0)

#time the first 30 steps to see how many steps fit in the budget
t0 = time.time()
for step in range(30):
    one_step(step, 10**9, train_generator)
seconds_per_step = (time.time() - t0) / 30
total_steps = max(31, int(train_minutes * 60 / seconds_per_step))
tokens_per_second = windows_per_step * context_length / seconds_per_step
print(f"{seconds_per_step:.3f} s per step ({tokens_per_second:,.0f} tokens/sec) -> {total_steps:,} steps in {train_minutes} min "
      f"= {total_steps * windows_per_step * context_length:,} tokens = {total_steps * windows_per_step * context_length / len(train_ids):.2f} trips through the text", flush=True)

best = float("inf")
recent = []
check_every = max(100, total_steps // 25)
train_start = time.time()
for step in range(30, total_steps):
    recent.append(one_step(step, total_steps, train_generator))
    if (step + 1) % check_every == 0 or step + 1 == total_steps:
        current = test_loss()
        if current < best:
            best = current
            save(out_folder / "best_model.pkl")
        minutes = (time.time() - train_start) / 60
        print(f"step {step + 1:,}/{total_steps:,}  train {np.mean(recent):.3f}  test {current:.3f}  (best {best:.3f})  {minutes:.1f} min", flush=True)
        recent = []
    #hard stop a little past the budget no matter what (the steps can slow down)
    if time.time() - train_start > train_minutes * 60 * 1.1:
        print("past the time budget, stopping", flush=True)
        break

print(f"\ndone. best test loss {best:.3f}", flush=True)
#write the stories with the best model, not just the last one
with open(out_folder / "best_model.pkl", "rb") as file:
    params.update({name: xp.asarray(a, dtype=dtype) for name, a in pickle.load(file)["params"].items()})
with open(out_folder / "stories.txt", "w") as file:
    for n, prompt in enumerate(prompts):
        story = generate(prompt, seed=n)
        print("\n" + story, flush=True)
        file.write(story + "\n\n")
