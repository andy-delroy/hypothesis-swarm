# pod_evidence/

This directory holds artefacts captured during the AMD MI300X pod training session,
demonstrating real GPU compute usage for **Track 3** compliance ("must demonstrate
AMD compute usage"). Drop files here during the session; they are committed to the
repo as permanent evidence.

---

## Expected contents

| File / pattern | What to capture |
|---|---|
| `rocm-smi.png` (or `.txt`) | Output of `rocm-smi` run **before training starts**, showing the AMD GPU(s) being addressed. Screenshot or `rocm-smi > pod_evidence/rocm-smi.txt`. |
| `training_log.txt` | Full, unedited terminal output of `bash verl_train.sh`. Pipe with `bash verl_train.sh 2>&1 \| tee pod_evidence/training_log.txt`. |
| `checkpoint_status_mid.txt` | Output of `bash checkpoint_status.sh` captured at least once **mid-session** (e.g. after the first few checkpoints appear). |
| `checkpoint_status_final.txt` | Output of `bash checkpoint_status.sh` captured at the **end of the session**, confirming the final step saved cleanly before the pod terminated. |
| `notebook_mid.png` | At least one screenshot of the Jupyter notebook UI while training is actively running (loss curve, step counter, or similar visible). |

Capture `checkpoint_status_mid.txt` and `checkpoint_status_final.txt` with:

```bash
bash checkpoint_status.sh | tee pod_evidence/checkpoint_status_mid.txt
# ... later ...
bash checkpoint_status.sh | tee pod_evidence/checkpoint_status_final.txt
```

---

## Why this exists

Track 3 of the hackathon requires demonstrable AMD GPU usage — not just a training
script, but evidence the run actually happened on AMD hardware. The `rocm-smi` output
pins the hardware, the training log shows the full job, and the checkpoint status
files confirm real checkpoints were written at real step counts. The notebook
screenshot ties the terminal output to the Jupyter environment on the pod.

For additional context, see the project's Notion decision log (if accessible), or
treat this README as the standalone explanation.
