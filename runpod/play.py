# a playground for the RunPod model: a tiny web server on your Mac that runs the model and a page to drive it
#   .venv/bin/python runpod/play.py          (opens http://127.0.0.1:8765 in your browser; Ctrl+C to stop)
# the page streams each new token as soon as the model picks it, plus the top choices it was picking between
import json
import time
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

print("loading the model...")
import write_story as model   # loads best_model.pkl and the exact forward() from train_tinystories.py

log = (model.here / "results" / "train.log").read_text() if (model.here / "results" / "train.log").exists() else ""
best_loss = next((line.split()[-1] for line in log.splitlines() if line.startswith("done. best test loss")), "?")
info = {
    "learned_numbers": int(sum(a.size for a in model.params.values())),
    "embedding_size": model.embedding_size, "context_length": model.context_length,
    "number_of_heads": model.number_of_heads, "number_of_blocks": model.number_of_blocks,
    "vocab_size": model.vocab_size, "best_test_loss": best_loss,
}

def piece(small_id):
    # one token's text. "<|endoftext|>" is the marker between stories
    return model.encoding.decode_single_token_bytes(model.used_ids[small_id]).decode("utf-8", errors="replace")

def tokenize(text):
    # the prompt as tokens. tokens that never appeared in TinyStories have no row in the model, so they get skipped
    out = []
    for big in model.encoding.encode_ordinary(text):   # plain text: the model learned <|endoftext|> as 7 ordinary tokens
        text_piece = model.encoding.decode_single_token_bytes(big).decode("utf-8", errors="replace")
        out.append({"piece": text_piece, "id": model.to_small.get(big), "known": big in model.to_small})
    return out

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass   # keep the terminal quiet

    def send(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            self.send((model.here / "play.html").read_bytes(), "text/html; charset=utf-8")   # re-read, so edits show on refresh
        elif url.path == "/info":
            self.send(json.dumps(info).encode(), "application/json")
        elif url.path == "/tokenize":
            self.send(json.dumps(tokenize(q.get("text", ""))).encode(), "application/json")
        elif url.path == "/generate":
            self.generate(q)
        else:
            self.send_error(404)

    def generate(self, q):
        new_tokens = max(1, min(int(q.get("tokens", 150)), 1000))
        temperature = max(0.05, min(float(q.get("temperature", 0.8)), 3.0))
        top_k = max(1, min(int(q.get("top_k", 40)), model.vocab_size))
        seed = int(q.get("seed", 0))
        prompt_tokens = tokenize(q.get("prompt", ""))
        ids = [t["id"] for t in prompt_tokens if t["known"]]

        # server-sent events: one "data: {...}" line per token, so the page can show each word as it arrives
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        def event(data):
            self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
            self.wfile.flush()

        try:
            event({"type": "prompt", "tokens": prompt_tokens})
            if not ids:
                ids = [t["id"] for t in tokenize("<|endoftext|>")]   # an empty prompt starts as if a story just ended
            write_generator = np.random.default_rng(seed)
            start = time.time()
            for _ in range(new_tokens):
                probabilities = model.next_token_probabilities(ids, temperature, top_k)
                choice = int(write_generator.choice(model.vocab_size, p=probabilities))
                ids.append(choice)
                top = np.argsort(probabilities)[::-1][:8]
                event({"type": "token", "piece": piece(choice), "p": float(probabilities[choice]),
                       "top": [[piece(int(i)), float(probabilities[i])] for i in top if probabilities[i] > 0]})
            event({"type": "done", "seconds": time.time() - start, "tokens": new_tokens})
        except (BrokenPipeError, ConnectionResetError):
            pass   # the page pressed Stop

port = 8765
while True:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        break
    except OSError:
        port += 1
address = f"http://127.0.0.1:{port}"
print(f"playground running at {address}  (Ctrl+C to stop)")
threading.Timer(0.5, lambda: webbrowser.open(address)).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nstopped")
