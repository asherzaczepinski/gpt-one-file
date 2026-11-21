import numpy as np
from pathlib import Path
import tiktoken

#main.py, but the model is one function (forward) so it's ready for backprop
#every learned number lives in one dict (params) and every step saves a "cache" for backprop to use later

current_path = Path(__file__).parent
text_file = current_path / 'stories.txt'

# gpt 4o encoding
encoding = tiktoken.get_encoding("o200k_base")

def tokenize(text_file_input=text_file):
    with open(text_file_input, 'r') as file:
        text = file.read()
    return encoding.encode(text)

tokens = tokenize(text_file)

used_ids = sorted(set(tokens))                              # the different ids that appear, smallest first
to_small = {big: small for small, big in enumerate(used_ids)} # map tiktoken ids to our smaller ids
token_ids = np.array([to_small[t] for t in tokens])                # the whole file in our ids
vocab_size = len(used_ids)

#how many numbers describe token
embedding_size = 128
#how many tokens it can see at once
context_length = 64

#splits up our input of the the different heads so the embeddings can be processed in parallel. Each head will have its own set of queries, keys, and values.
number_of_heads = 4
head_size = embedding_size // number_of_heads    # 128 / 4 = 32

#how many blocks stacked: block 1 feeds block 2 feeds ... block 6
number_of_blocks = 6

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


#cache = stuff we save on the way forward so backprop can use it later
#backprop goes through every step in reverse, and each step needs to know what went into it

#we layer norm our vectors, so they have mean 0 and std 1, then scale and shift them
def layer_norm(vectors, scale, shift, tiny=1e-5):
    average = vectors.mean(axis=-1, keepdims=True)
    variance = vectors.var(axis=-1, keepdims=True)
    spread = np.sqrt(variance + tiny)
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
    exps = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)

#gelu, the bend. positive numbers pass through, negative ones get squashed to about 0
#   gelu([-2, 0, 2]) = [-0.045, 0, 1.955]
#so a detector that's positive "fires", and one that's negative stays quiet
#why it's needed: two matrix multiplies in a row with nothing between them are the same as one matrix multiply,
#so without the bend the mlp could only learn straight-line rules. the bend is what lets it learn "if this AND that"
#(0.7978845608 is sqrt(2/pi) and 0.044715 is a number from the gelu paper, it's a smooth version of "max(0, x)")
def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

#the whole model: a window of token ids in, a score for every possible next token out
def forward(window):
    #what needs a cache: any step that multiplies by something (linear, layer norm, the q@k and weights@values, gelu),
    #because backprop needs to know what got multiplied. what doesn't: plain adds (the residuals), reshapes and transposes, since backprop just passes the gradient straight through or shuffles it back
    #we have our tiny window of tokens, and we want to get their embeddings and add the position embeddings
    vectors = params["token_embedding_table"][window] + params["position_embedding_table"][:len(window)]
    #stores stuff for backprop
    block_caches = []
    for i in range(number_of_blocks):
        b = f"block{i + 1}."
        cache = {}    # everything this block's backprop will need
        normalized_vectors, cache["layer_norm"] = layer_norm(vectors, params[b + "layer_norm_scale"], params[b + "layer_norm_shift"])
        queries_keys_values, cache["qkv"] = linear(normalized_vectors, params[b + "qkv_weights"], params[b + "qkv_bias"])
        queries, keys, values = np.split(queries_keys_values, 3, axis=-1)

        #so there is basically 128 embeddings
        #each head gets to look at 32 of those embeddigns in this case
        #the heads learn different ways to look at each 32 group
        #every token gets it own 32 numbers for each head

        #deadass just shifts around so qkv is the 4 heads, so each head has its own queries, keys, and values. It is a way to organize the data for multi-head attention.
        #reshape: (64, 128) -> (64, 4, 32)   each token's 128 numbers become 4 rows of 32
        #transpose: (64, 4, 32) -> (4, 64, 32)   group by head first, so each head is its own (64, 32) grid

        #now we have the heads w/ the query keys and values
        queries = queries.reshape(len(window), number_of_heads, head_size).transpose(1, 0, 2)
        keys = keys.reshape(len(window), number_of_heads, head_size).transpose(1, 0, 2)
        values = values.reshape(len(window), number_of_heads, head_size).transpose(1, 0, 2)

        #score every token's query against every token's key, separately in each head
        #a dot product (multiply the 32 matching numbers, add them up) is big when "what I'm looking for" matches "what you have"
        #keys.transpose(0, 2, 1) flips each head's keys from (64, 32) to (32, 64) so the matrix multiply lines up
        #(4, 64, 32) @ (4, 32, 64) -> (4, 64, 64): for each head, row i column j = how much token i cares about token j
        #divide by sqrt(32) because adding up 32 products makes big numbers, and big numbers make softmax too extreme later
        scores = queries @ keys.transpose(0, 2, 1) / np.sqrt(head_size)

        #hide the future. a token can only look at itself and the tokens BEFORE it
        #otherwise in training it could just peek at the next word instead of learning to guess it
        #np.triu(..., k=1) makes a (64, 64) grid that's True above the diagonal, which is every "later token" spot
        #   token 0 can see [0]    token 1 can see [0, 1]    token 63 can see [0 ... 63]
        future = np.triu(np.ones((len(window), len(window)), dtype=bool), k=1)

        #put -1e9 (basically negative infinity) in every future spot. softmax turns that into a weight of 0 next step
        #the same (64, 64) grid gets used for all 4 heads automatically
        scores = np.where(future, -1e9, scores)

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
        gathered = gathered.transpose(1, 0, 2).reshape(len(window), embedding_size)

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
    return logits, (window, block_caches, final_cache, head_cache)





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
def get_loss(logits, targets):
    #subtract each row's biggest score from that whole row
    shifted = logits - logits.max(axis=-1, keepdims=True)
    #for each row: e^ every score, add them up, log it. (64,)
    log_of_total = np.log(np.exp(shifted).sum(axis=-1))
    #for each row, grab the score of the right answer: row 0 column targets[0], row 1 column targets[1], ... (64,)
    right_answer_scores = shifted[np.arange(len(targets)), targets]
    #-log(probability of the right answer) at every spot, then average the 64 of them into one number
    return float(np.mean(log_of_total - right_answer_scores))


#the targets are the same window shifted by one, so every spot's answer is the next token
window = token_ids[:context_length]
targets = token_ids[1:context_length + 1]

logits, cache = forward(window)
loss = get_loss(logits, targets)

print(logits.shape)    # (64, 4841): a score for every possible next token, at every spot
print(f"loss {loss:.3f}, random guessing would be {np.log(vocab_size):.3f}")   # about the same, nothing is trained yet

#next: backprop. go through forward in reverse using the caches, work out how to nudge every number in params






