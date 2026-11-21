#watchdog: keeps an eye on training so the mac doesn't get hurt overnight
#every 10 seconds it checks two things, and if either goes wrong it stops every training process
#and writes runs/STOP so no new training run gets started:
#   1. training using too much cpu: more than 1.5 cores in total for 30 seconds in a row
#      (training is set to use 1 core, so this would mean something is wrong)
#   2. the mac reporting any thermal or performance warning (pmset -g therm), meaning it's running hot
#every 5 minutes it writes a status line to runs/watchdog.log
#run it with:  nice -n 19 .venv/bin/python runs/watchdog.py
import os
import signal
import subprocess
import time
from pathlib import Path

runs_folder = Path(__file__).parent
log_file = runs_folder / "watchdog.log"
stop_file = runs_folder / "STOP"

max_cores = 1.5            # more than this many cores busy with training = too much
strikes_to_stop = 3        # this many checks in a row over the limit before stopping (3 x 10 sec = 30 sec)
check_seconds = 10
status_every_seconds = 300

def log(message):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + message
    with open(log_file, "a") as file:
        file.write(line + "\n")

#every running training process: [(pid, cpu percent)]. 100% = one full core
def training_processes():
    #only real python processes whose script argument is main_functionized2.py (not shells that just mention it)
    output = subprocess.run(["ps", "-axo", "pid=,%cpu=,args="], capture_output=True, text=True).stdout
    found = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        program, arguments = parts[2], parts[3:]
        is_python = "python" in os.path.basename(program).lower()
        runs_script = any(a == "main_functionized2.py" or a.endswith("/main_functionized2.py") for a in arguments)
        if is_python and runs_script:
            found.append((int(parts[0]), float(parts[1])))
    return found

#True if macos has recorded any thermal/performance warning. normally it says "No ... has been recorded"
def mac_is_hot():
    output = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("Note: No"):
            continue
        #anything else (a warning level, a CPU speed limit under 100) counts as hot
        if "CPU_Speed_Limit" in line and line.endswith("100"):
            continue
        return True, line
    return False, ""

def stop_training(reason):
    stop_file.write_text(reason + "\n")
    for pid, _ in training_processes():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid, _ in training_processes():         # anything still alive gets force-stopped
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    log("STOPPED TRAINING: " + reason)

log("watchdog started")
strikes = 0
last_status = 0
while True:
    processes = training_processes()
    cores = sum(cpu for _, cpu in processes) / 100
    hot, hot_line = mac_is_hot()

    if hot:
        stop_training("mac reported a thermal warning: " + hot_line)
    elif cores > max_cores:
        strikes += 1
        log(f"warning: training using {cores:.2f} cores (strike {strikes} of {strikes_to_stop})")
        if strikes >= strikes_to_stop:
            stop_training(f"training used more than {max_cores} cores for {strikes * check_seconds} seconds")
            strikes = 0
    else:
        strikes = 0

    if time.time() - last_status >= status_every_seconds:
        state = "STOP file present, no new runs" if stop_file.exists() else "ok"
        log(f"status: {len(processes)} training process(es), {cores:.2f} cores, thermal warning: {'yes' if hot else 'no'}, {state}")
        last_status = time.time()

    time.sleep(check_seconds)
