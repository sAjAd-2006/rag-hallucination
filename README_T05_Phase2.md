# T-05 Phase 2 — Colab Instructions

1. Open `T05_Phase1_and_Phase2_RAG.ipynb` in Google Colab.
2. Run all cells from the beginning; Phase 1 must run before Phase 2.
3. Phase 1 installs `datasets`, `sentence-transformers`, `faiss-cpu`, and
   `transformers`.
4. `N_EVAL=30` is the fast setting; use 100 for a fuller CPU run.
5. Phase 1 keeps the model on CPU; selecting a GPU alone does not move it.
6. Outputs are written to `./t05_phase2_outputs` and zipped after report generation.
7. Phase 2 does not replace or modify the Phase 1 baseline.
