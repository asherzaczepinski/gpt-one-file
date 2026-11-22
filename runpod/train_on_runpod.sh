#!/usr/bin/env bash
# One command: rent a GPU on RunPod, train the bigger story model on TinyStories, copy the results back, delete the pod.
# Same create / ssh / delete-and-verify pattern as ASHERLLM GOLDEN/01 runpod_train.sh and Race Hawks' pod scripts,
# but through RunPod's GraphQL API (api.runpod.io/graphql): rest.runpod.io stopped resolving in Nov 2026.
#
#   caffeinate -i bash runpod/train_on_runpod.sh                 # about $1 total
#   BUDGET=0.50 caffeinate -i bash runpod/train_on_runpod.sh     # about 50 cents
#
# It never spends more than BUDGET: it works out how many minutes BUDGET buys at the pod's price, trains for that
# minus setup time, and the pod has a timer that deletes itself at the limit even if this Mac sleeps or loses wifi.
set -uo pipefail
cd "$(dirname "$0")"

BUDGET=${BUDGET:-1.00}          # dollars for the whole thing
TEXT_MB=${TEXT_MB:-400}         # how much TinyStories text to download (400 MB is about 100 million tokens)
SETUP_MINUTES=${SETUP_MINUTES:-15}   # boot + installs + download + tokenize + writing stories + copying back
CREDS=${RUNPOD_CREDS:-"$HOME/Desktop/Race Hawks Older/Old backend_v2/backend_v2/cloud-render-test/runpod-creds-asherzac2020.json"}
LOG=run_log.txt
OUT=results
GPUS=("NVIDIA RTX 2000 Ada Generation" "NVIDIA RTX A4000" "NVIDIA RTX A4500" "NVIDIA RTX A5000" "NVIDIA RTX 4000 Ada Generation")
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"     # the same image Race Hawks' GPU pods use (CUDA 12.8)

note(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# the api key: RUNPOD_API_KEY if it's set, otherwise the Race Hawks creds file
APIKEY=${RUNPOD_API_KEY:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["apiKey"])' "$CREDS" 2>/dev/null)}
[ -z "$APIKEY" ] && { echo "no API key: set RUNPOD_API_KEY or RUNPOD_CREDS (a json file with an apiKey)"; exit 1; }
[ -f "$HOME/.ssh/id_ed25519.pub" ] || { echo "no ssh key at ~/.ssh/id_ed25519.pub (make one: ssh-keygen -t ed25519)"; exit 1; }
PUBKEY="$(cat "$HOME/.ssh/id_ed25519.pub")"
SSHOPT=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o StrictHostKeyChecking=accept-new -i "$HOME/.ssh/id_ed25519")
SSH(){ ssh "${SSHOPT[@]}" -p "$PORT" "root@$HOST" "$@"; }
API(){ curl -s -H "Authorization: Bearer $APIKEY" -H "Content-Type: application/json" "$@"; }
# gql 'query text' -> sends it to RunPod's GraphQL API (python builds the JSON so quotes can't break it)
gql(){ python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$1" | API https://api.runpod.io/graphql -d @-; }
bal(){ gql '{ myself { clientBalance } }' | python3 -c 'import json,sys; print("%.2f" % json.load(sys.stdin)["data"]["myself"]["clientBalance"])'; }
listed(){ gql '{ myself { pods { id } } }' | grep -q "\"$1\""; }

# keep the mac awake while this script runs (it still works if it sleeps: the pod deletes itself at the limit)
caffeinate -i -w $$ & disown

PODID=""
# delete AND verify (a DELETE once returned without killing the pod, see GOLDENGOLDENGOLDEN/14)
cleanup(){ [ -z "$PODID" ] && return
  for i in 1 2 3 4 5 6; do
    gql "mutation { podTerminate(input: {podId: \"$PODID\"}) }" > /dev/null; sleep 5
    listed "$PODID" || { note "pod $PODID deleted (verified)"; PODID=""; return; }
  done; note "POD $PODID STILL LISTED after 6 deletes: delete it by hand at runpod.io -> Pods"; }
trap cleanup EXIT INT TERM

BAL0=$(bal)
note "== train_on_runpod: budget \$$BUDGET, $TEXT_MB MB of TinyStories. balance before: \$$BAL0"
python3 -c "import sys; sys.exit(0 if float('$BAL0') >= float('$BUDGET') + 0.25 else 1)" || { note "balance too low for this budget, stopping"; exit 1; }

# 1. rent a GPU: try the cheap ones in order until one is in stock
T0=$(date +%s)
for gpu in "${GPUS[@]}"; do
  resp=$(gql "mutation { podFindAndDeployOnDemand(input: { name: \"gpt-tinystories\", imageName: \"$IMAGE\",
    gpuTypeId: \"$gpu\", gpuCount: 1, cloudType: SECURE, containerDiskInGb: 30, volumeInGb: 0,
    ports: \"22/tcp\", startSsh: true, env: [{ key: \"PUBLIC_KEY\", value: \"$PUBKEY\" }] }) { id costPerHr } }")
  PODID=$(echo "$resp" | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["data"]["podFindAndDeployOnDemand"]["id"] or "")
except Exception: print("")')
  [ -n "$PODID" ] && { note "pod $PODID created on $gpu"; break; }
  note "no $gpu available ($(echo "$resp" | head -c 160))"
done
[ -z "$PODID" ] && { note "no GPU available right now, try again later"; exit 1; }

# 2. how many minutes the budget buys at this pod's price
PRICE=$(echo "$resp" | python3 -c 'import json,sys
try: print(float(json.load(sys.stdin)["data"]["podFindAndDeployOnDemand"]["costPerHr"] or 0))
except Exception: print(0)')
python3 -c "import sys; sys.exit(0 if float('$PRICE') > 0 else 1)" || { PRICE=0.60; note "couldn't read the price, assuming \$0.60/hr to be safe"; }
POD_MINUTES=$(python3 -c "print(int(float('$BUDGET') * 0.95 / float('$PRICE') * 60))")
TRAIN_MINUTES=$(( POD_MINUTES - SETUP_MINUTES ))
note "price \$$PRICE/hr -> pod limit $POD_MINUTES min, training $TRAIN_MINUTES min"
[ "$TRAIN_MINUTES" -lt 10 ] && { note "budget too small for this GPU price, stopping"; exit 1; }

# 3. wait for ssh
ep=""
for i in $(seq 1 60); do
  ep=$(gql "{ pod(input: {podId: \"$PODID\"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }" | python3 -c 'import json,sys
try:
    for p in json.load(sys.stdin)["data"]["pod"]["runtime"]["ports"]:
        if p["privatePort"] == 22 and p["isIpPublic"]: print(p["ip"] + ":" + str(p["publicPort"])); break
except Exception: pass')
  [ -n "$ep" ] && break; sleep 10
done
[ -z "$ep" ] && { note "no ssh endpoint"; exit 1; }
HOST=${ep%%:*}; PORT=${ep#*:}
for i in $(seq 1 30); do SSH true 2>/dev/null && break; sleep 5; done
note "ssh up after $(( $(date +%s) - T0 )) s"

# 4. the self-destruct timer: the pod deletes itself at POD_MINUTES no matter what happens to this Mac.
#    (the key goes to a file on YOUR pod only, and disappears when the pod is deleted)
SSH "umask 077; cat > /root/selfdestruct.sh" <<KILL
sleep $(( POD_MINUTES * 60 ))
curl -s -H "Authorization: Bearer $APIKEY" -H "Content-Type: application/json" https://api.runpod.io/graphql -d '{"query":"mutation { podTerminate(input: {podId: \"$PODID\"}) }"}'
KILL
SSH "nohup bash /root/selfdestruct.sh > /dev/null 2>&1 < /dev/null &"
note "self-destruct timer set: the pod deletes itself after $POD_MINUTES min"

# 5. send the training script, then run setup + training on the pod in the background (survives ssh drops)
SSH "mkdir -p /root/gpt" && scp "${SSHOPT[@]}" -P "$PORT" -q train_tinystories.py "root@$HOST:/root/gpt/" && note "training script copied"
SSH "cat > /root/gpt/remote.sh" <<REMOTE
set -uo pipefail
cd /root/gpt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# --break-system-packages: Ubuntu 24.04 blocks plain pip installs (PEP 668). fine on a throwaway pod
pip install -q --break-system-packages cupy-cuda12x tiktoken numpy 2>&1 | tail -2
python -c "import cupy; a = cupy.ones((1000, 1000)); print('cupy works:', float((a @ a).sum()))" || { echo "SETUP FAILED: cupy can't use the GPU"; echo REMOTE-DONE; exit 1; }
# the first $TEXT_MB MB of TinyStories (free, ~2 million short stories): head stops the download early
curl -sL https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt | head -c $(( TEXT_MB * 1000000 )) > tinystories.txt
echo "downloaded \$(du -h tinystories.txt | cut -f1) of stories"
TEXT_FILE=tinystories.txt TRAIN_MINUTES=$TRAIN_MINUTES OUT_FOLDER=/root/gpt/out python -u train_tinystories.py
echo REMOTE-DONE
REMOTE
# setsid + nohup = training keeps going on its own. this ssh used to hang until training ended (Nov 21 run),
# so it gets 15 seconds to start things and is then cut off (that can't hurt the training)
ssh "${SSHOPT[@]}" -n -p "$PORT" "root@$HOST" "cd /root/gpt && setsid nohup bash remote.sh > train.log 2>&1 < /dev/null &" &
sleep 15; kill $! 2>/dev/null
note "training started on the pod. progress every 2 minutes:"

# 6. watch it (the training prints a line every few % of the run)
DEADLINE=$(( T0 + POD_MINUTES * 60 - 120 ))
shown=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 120
  lines=$(SSH "cat /root/gpt/train.log" 2>/dev/null) || { note "(couldn't reach the pod this time, still trying)"; continue; }
  echo "$lines" | tail -n +$(( shown + 1 )) | grep -vE "^\[|^ *\[" | tee -a "$LOG"
  shown=$(echo "$lines" | wc -l)
  echo "$lines" | grep -q REMOTE-DONE && break
done

# 7. copy the results back, then the trap deletes the pod
mkdir -p "$OUT"
SSH "cd /root/gpt && tar czf - train.log \$(ls -d out 2>/dev/null)" | tar xzf - -C "$OUT" && note "results copied to runpod/$OUT/ (best_model.pkl, stories.txt, train.log)"
cleanup
sleep 20
note "balance after: \$$(bal)  (before: \$$BAL0)"
note "pods still running on the account: $(gql '{ myself { pods { id } } }' | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["data"]["myself"]["pods"]))')"
