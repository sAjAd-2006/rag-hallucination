# T-05 Phase 2 Report — RAG and Hallucination

## Project summary
Phase 2 evaluates the unchanged Phase 1 RAG and No-RAG baselines on a
closed-corpus SQuAD 2.0 subset.

## Baseline setup reminder
- Dataset: SQuAD 2.0 (`train`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Search: FAISS flat L2 over 300 contexts
- Generator: `google/flan-t5-small`
- RAG evidence: top-1 context

## Dataset and evaluation subset
- Questions: 30
- Answerable: 23
- Unanswerable: 7
- The gold context is inside the Phase 1 index by construction, so retrieval
  results measure ranking inside this selected corpus and are not estimates
  for the full SQuAD dataset.

## Retrieval results
| Metric | Value |
|---|---:|
| Recall@1 | 0.9130 |
| Recall@3 | 1.0000 |
| Recall@5 | 1.0000 |
| MRR | 0.9565 |

## Generation results
| Mode | EM | Token F1 | ROUGE-L | Answerable EM | Unanswerable detection |
|---|---:|---:|---:|---:|---:|
| RAG | 0.3333 | 0.4411 | 0.4411 | 0.4348 | 0.0000 |
| No-RAG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

An all-zero No-RAG result is not changed or smoothed. The notebook displays
real No-RAG predictions so the result can be checked directly.

## Simple breakdown
| slice                                  |   n |   RAG EM |   RAG F1 |   NoRAG EM |   NoRAG F1 |
|:---------------------------------------|----:|---------:|---------:|-----------:|-----------:|
| Answerable                             |  23 | 0.434783 | 0.575362 |          0 |          0 |
| Unanswerable                           |   7 | 0        | 0        |          0 |          0 |
| Short (<= 10 words)                    |  20 | 0.35     | 0.478333 |          0 |          0 |
| Long (> 10 words)                      |  10 | 0.3      | 0.366667 |          0 |          0 |
| Answerable & retrieval success (top-1) |  21 | 0.47619  | 0.630159 |          0 |          0 |
| Answerable & retrieval failure (top-1) |   2 | 0        | 0        |          0 |          0 |

## Representative errors
| Question                                                                              | Gold answer                 | Retrieved context snippet (top-1)                                                                                                                                           | RAG answer                                  | No-RAG answer       | Error type                                     | Short explanation                                                                   |
|:--------------------------------------------------------------------------------------|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------|:--------------------|:-----------------------------------------------|:------------------------------------------------------------------------------------|
| What was the republican government amenable to?                                       | war reparations             | In March 1971, the residential office of an FBI agent in Media, Pennsylvania was burglarized by a group calling itself the Citizens' Commission to Investigate t...         | assassinations of political activists       | a reformed republic | Retrieval miss / absent-evidence hallucination | The exact top-1 passage seen by the generator was not the gold context.             |
| What are the helpers called that helped Buddha?                                       | disciples                   | Buddhism /ˈbudɪzəm/ is a nontheistic religion[note 1] or philosophy (Sanskrit: धर्म dharma; Pali: धम्म dhamma) that encompasses a variety of traditions, beliefs...           | the awakened one                            | saigon              | Retrieval miss / absent-evidence hallucination | The exact top-1 passage seen by the generator was not the gold context.             |
| Which recitation is the original basis of the Quran of Cairo?                         | ʻAsim's                     | Vocalization markers indicating specific vowel sounds were introduced into the Arabic language by the end of the 9th century. The first Quranic manuscripts lack...         | Asim's recitation                           | sacrimony           | Ignored-evidence hallucination                 | The gold context was top-1, but the generated answer did not match the gold answer. |
| How many causalities did the US Air Force suffer during WWII?                         | 68,000                      | The U.S. War Department created the first antecedent of the U.S. Air Force in 1907, which through a succession of changes of organization, titles, and missions ...         | 3                                           | 2                   | Ignored-evidence hallucination                 | The gold context was top-1, but the generated answer did not match the gold answer. |
| What was the decrease in non-German population between 2011 and 2013?                 | (unanswerable)              | In July 2013, there were 41,000 non-Germans by citizenship living in Thuringia (1.9% of the population − among the smallest proportions of any state in Germany)...         | 24%                                         | a year              | Unanswerable failure                           | SQuAD marks the question unanswerable, but the model produced a content answer.     |
| In what year did Heinz Kloss develop a framework for describing dialects genetically? | (unanswerable)              | A framework was developed in 1967 by Heinz Kloss, abstand and ausbau languages, to describe speech communities, that while unified politically and/or culturally...         | 1967                                        | 1897                | Unanswerable failure                           | SQuAD marks the question unanswerable, but the model produced a content answer.     |
| Who established the Tibetan law code?                                                 | Tai Situ Changchub Gyaltsen | The Ming court appointed three Princes of Dharma (法王) and five Princes (王), and granted many other titles, such as Grand State Tutors (大國師) and State Tutors (國師... | Phagmodru ruler Tai Situ Changchub Gyaltsen | samuel sai          | Correct                                        | The answer exactly matches or closely overlaps a gold answer.                       |
| How long did it take for Myanmar to recover from the collapse of it's first kingdom ? | 250 years                   | Pagan's collapse was followed by 250 years of political fragmentation that lasted well into the 16th century. Like the Burmans four centuries earlier, Shan migr...         | 250 years                                   | a few days          | Correct                                        | The answer exactly matches or closely overlaps a gold answer.                       |

## Limitations
The subset and corpus are small; the generator and prompt are simple;
metrics are lexical; hallucination labels and abstention detection are
rule-based; and the closed-corpus sampling makes retrieval scores optimistic
relative to a full open-corpus evaluation.

## Future Work
Improve the abstention prompt, test a stronger embedder or reranker, increase
the corpus and evaluation set, try a stronger generator, and conduct limited
human review. These are future directions and do not replace the Phase 1
baseline in this phase.

## Lightweight demo summary
The notebook displays two distinct answerable questions and one unanswerable
question, their top retrieved contexts, RAG answer, No-RAG answer, and gold.

## Input/output integration note
Input: `{"question": "...", "course_documents": ["..."]}`

Output: `{"answer": "...", "retrieved_contexts": ["..."],
"confidence_or_warning": "...", "error_type_if_known": "..."}`

## External AI tools acknowledgement
External AI assistants were used for drafting, debugging, and writing support.
All code, outputs, and interpretations were reviewed by the team.
