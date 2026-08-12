# T-05 Final Submission

## Topic

Retrieval-Augmented Generation and the Study of Hallucination

## Recommended review order

1. `T05_Phase1_Report.pdf`
2. `T05_Phase1_Baseline.ipynb`
3. `T05_Phase1_Outputs.zip`
4. `T05_Final_Report.pdf`
5. `T05_Phase2_Evaluation.ipynb`
6. `T05_Phase2_Outputs.zip`
7. `phase2_manual_review_completed.csv`
8. `model_comparison.csv`
9. `T05_Bonus_Evaluation.ipynb`
10. `T05_Bonus_Outputs.zip`
11. `T05_Presentation.pptx`

The `report_sources/` folder contains the standalone HTML/CSS source for both PDF reports. The delivered PDFs are rendered directly from those HTML pages. Both reports use the supplied dark-navy, blue/green, card-and-table visual system, and the presentation is fully English.

## Main verified results

- 120 evaluation questions: 80 answerable and 40 unanswerable.
- 113 unique indexed passages.
- Generator: `google/flan-t5-base`.
- Overall Exact Match: 50.83%.
- Overall Token F1: 59.09%.
- Answerable retrieval hit@1: 88.75%.
- Answerable retrieval hit@3: 98.75%.
- Unanswerable abstention accuracy: 25.00%.
- The model returned substantive answers for 30 of 40 unanswerable questions.

The completed manual review covers 22 deliberately selected candidate failures from the base-model run. It is a failure-enriched qualitative sample and must not be used as an overall hallucination-rate estimate.

## Controlled generator comparison

The previous `flan-t5-small` run and the current `flan-t5-base` run use the same 120 questions, seed, corpus, embeddings, FAISS retrieval, baseline prompt, and greedy decoding. Changing only the generator produced:

- overall EM: 43.33% → 50.83% (+7.50 points);
- overall F1: 50.66% → 59.09% (+8.43 points);
- unanswerable abstention: 2.50% → 25.00% (+22.50 points);
- answerable EM: unchanged at 63.75%;
- retrieval hit@1 and hit@3: unchanged at 88.75% and 98.75%.

Nine new abstentions were added, all on unanswerable questions, and no previous abstention was lost. The main gain therefore comes from better answerability behavior rather than a change in retrieval.

## Lightweight bonus extensions

The separate bonus notebook adds three low-code extensions without changing the original baseline:

- a six-case representative demo, including answerable, unanswerable, high/low-similarity, and two custom examples;
- an equal-size descriptive slice of the 120 predictions into low, medium, and high Top-1 similarity groups;
- a controlled comparison of the original prompt with a stricter abstention prompt on the same 120 examples.

With `flan-t5-base`, the stricter prompt creates a small trade-off: overall EM changes from 50.83% to 51.67%, overall F1 from 59.09% to 60.59%, and unanswerable abstention from 25.00% to 30.00%, while answerable F1 decreases from 76.13% to 75.88%. It corrects four Exact Match cases and harms three. Similarity-group results are descriptive associations, not a calibrated confidence threshold, because the groups have different answerable/unanswerable composition.

## Reproduction

Run the Phase 1 notebook first. Its output ZIP is the input artifact for the separate Phase 2 notebook. After Phase 2, run the bonus notebook with both Phase 1 and Phase 2 output ZIPs available. The notebooks use SQuAD v2, all-MiniLM-L6-v2, exact FAISS search, and FLAN-T5-base. No training or fine-tuning is required.

To reproduce either report, open its HTML file from `report_sources/` and print to A4 PDF at 100% scale, with no margins and background graphics enabled.

## Academic integrity note

External AI assistance was used to help structure code, manual-review documentation, reports, and slides. The experiment outputs were produced by the submitted notebooks and reviewed against retrieved evidence. The submitting team remains responsible for understanding and defending every design choice and reported number.
