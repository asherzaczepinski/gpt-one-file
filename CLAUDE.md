Before doing anything in this project, read `for-claude/README.md`. It has the goal, who edits which file,
and where Asher is in the build.

Short version: **never edit `main.py`** (Asher writes it by hand). `main2.py` is the commented step-by-step build
(put the next step there). `main3.py` is the complete, commented program that trains.
`main4.py` = `main.py` + the next forward step. `main_functionized.py` = `main.py` as a forward function.
`main6.py` = that + the backward pass, one step at a time.
