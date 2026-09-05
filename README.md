# RA-VLA: Retrieval-Augmented VLA for Test-Time Adaptation

This repository is built upon [NVIDIA GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) framework.

## ⚠️ Important Disclaimer

Please note that any identity-related information or metadata found within this codebase (e.g., author names, emails, or file paths) is inherited from external templates or previous codebase. These details are **not associated with the authors of this work**.

---

## 📂 Project Structure

The core implementation of RA-VLA can be found in the following locations:

* `gr00t/rag/*.py`: Core model architecture and retrieval logic.
* `scripts/rag/*.py`: Execution and utility scripts.

---

## 🚀 Execution Guide

Follow the steps below to train the components and evaluate the policy.

**1. Train Retriever:**

```bash
python scripts/rag/1_train_retriever.py
```

**2. Build Memory:**

```bash
python scripts/rag/2_build_memory.py
```

**3. Train RA-VLA:**

```bash
python scripts/rag/3_train_ravla.py
```

**4. Serve RA-VLA:**

```bash
python scripts/rag/4_serve_ravla.py
```

**5. Run LIBERO Evaluation:**
```bash
python eval_scripts/gr00t/run_libero_eval.py
```

---
