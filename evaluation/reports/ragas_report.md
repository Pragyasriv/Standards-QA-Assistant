# RAGAS evaluation

Generated at 2026-11-16T04:46:58.822388+00:00.

## Dataset

- Questions: 50
- RoHS: 25
- REACH: 25
- Successfully answered: 35
- Scored on all four metrics: 35
- Unexpected abstentions: 15
- Provider failures: 0
- Pipeline errors: 0
- Evaluation failures: 0

## Configuration

- RAGAS version: 0.4.4
- Evaluator model: qwen/qwen3.8-27b
- Evaluator temperature: 0.0
- Evaluator max tokens: 2048
- Answer relevancy strictness: 2
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Generation model: openai/gpt-oss-20b
- Generation temperature: 0.0
- Live Qdrant: True
- Live BM25: True
- Live Groq: True

## Scoring policy

Every curated question stays in the dataset. RAGAS is called only when the pipeline returned a non-empty answer and at least one retrieved context. Abstentions, guardrail rejections, validation failures, and provider failures are counted separately and receive null metric scores. Each mean is the arithmetic mean of the non-null scores for that metric. Nulls are not replaced with zero. A transient provider error is retried; a validation rejection is not.

## Metric distributions

| Metric | Count | Excluded | Mean | Min | P25 | P50 | P75 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faithfulness | 35 | 15 | 0.878571 | 0 | 1 | 1 | 1 | 1 |
| answer relevancy | 35 | 15 | 0.538929 | 0 | 0.388396 | 0.576436 | 0.707634 | 0.974054 |
| context precision | 35 | 15 | 0.953619 | 0.416667 | 1 | 1 | 1 | 1 |
| context recall | 35 | 15 | 1 | 1 | 1 | 1 | 1 | 1 |

Buckets are [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1].

- faithfulness: [0,0.25)=4, [0.25,0.5)=0, [0.5,0.75)=0, [0.75,1]=31
- answer_relevancy: [0,0.25)=4, [0.25,0.5)=12, [0.5,0.75)=12, [0.75,1]=7
- context_precision: [0,0.25)=0, [0.25,0.5)=1, [0.5,0.75)=1, [0.75,1]=33
- context_recall: [0,0.25)=0, [0.25,0.5)=0, [0.5,0.75)=0, [0.75,1]=35

## Per-question scores

| ID | Regulation | Status | Faithfulness | Relevancy | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| rohs-article-1-subject | RoHS | answered | 1 | 0.410685 | 1 | 1 |
| rohs-article-2-scope | RoHS | answered | 1 | 0.717957 | 0.416667 | 1 |
| rohs-eee-definition | RoHS | abstained | null | null | null | null |
| rohs-homogeneous-material | RoHS | answered | 1 | 0.708349 | 1 | 1 |
| rohs-article-4-prevention | RoHS | answered | 1 | 0.586122 | 1 | 1 |
| rohs-article-4-concentration | RoHS | insufficient_evidence | null | null | null | null |
| rohs-article-6-review | RoHS | insufficient_evidence | null | null | null | null |
| rohs-article-7-manufacturers | RoHS | answered | 1 | 0.774359 | 1 | 1 |
| rohs-article-9-importers | RoHS | answered | 0 | 0.580146 | 1 | 1 |
| rohs-article-10-distributors | RoHS | answered | 1 | 0.592255 | 1 | 1 |
| rohs-article-11-manufacturer | RoHS | answered | 1 | 0.86503 | 1 | 1 |
| rohs-article-13-declaration | RoHS | answered | 1 | 0.793911 | 1 | 1 |
| rohs-article-15-ce | RoHS | answered | 1 | 0.668815 | 1 | 1 |
| rohs-article-16-presumption | RoHS | answered | 0 | 0.768712 | 1 | 1 |
| rohs-article-23-penalties | RoHS | abstained | null | null | null | null |
| rohs-article-25-transposition | RoHS | answered | 0 | 0.224599 | 1 | 1 |
| rohs-article-26-repeal | RoHS | answered | 1 | 0.357416 | 1 | 1 |
| rohs-article-27-entry | RoHS | answered | 0 | 0.366108 | 1 | 1 |
| rohs-lead-limit | RoHS | abstained | null | null | null | null |
| rohs-cadmium-limit | RoHS | abstained | null | null | null | null |
| rohs-dehp-medical-date | RoHS | abstained | null | null | null | null |
| rohs-phthalate-toys | RoHS | clarification_required | null | null | null | null |
| rohs-annex-iv-detectors | RoHS | answered | 1 | 0.480898 | 1 | 1 |
| rohs-annex-v-applicants | RoHS | answered | 1 | 0.974054 | 1 | 1 |
| rohs-article-4-and-lead | RoHS | insufficient_evidence | null | null | null | null |
| reach-article-1-aim | REACH | answered | 1 | 0.661439 | 1 | 1 |
| reach-substance-definition | REACH | answered | 0.75 | 0.576436 | 1 | 1 |
| reach-article-5-no-data | REACH | answered | 1 | 0.66334 | 1 | 1 |
| reach-article-6-registration | REACH | answered | 1 | 0.706918 | 0.666667 | 1 |
| reach-article-7-articles | REACH | answered | 1 | 0.417344 | 0.81 | 1 |
| reach-article-8-representative | REACH | answered | 1 | 0.286777 | 1 | 1 |
| reach-article-9-ppord | REACH | answered | 1 | 0.787555 | 1 | 1 |
| reach-article-14-csr | REACH | answered | 1 | 0.436283 | 1 | 1 |
| reach-article-17-intermediate | REACH | answered | 1 | 0.531933 | 0.833333 | 1 |
| reach-article-23-phase-in | REACH | answered | 1 | 0.491694 | 1 | 1 |
| reach-article-25-animals | REACH | answered | 1 | 0.936753 | 1 | 1 |
| reach-article-31-sds | REACH | answered | 1 | 0.498233 | 0.816667 | 1 |
| reach-article-33-concentration | REACH | answered | 1 | 0.627797 | 1 | 1 |
| reach-article-55-authorisation | REACH | answered | 1 | 0.303389 | 1 | 1 |
| reach-article-56-annex-xiv | REACH | insufficient_evidence | null | null | null | null |
| reach-article-57-carcinogen | REACH | insufficient_evidence | null | null | null | null |
| reach-article-60-commission | REACH | answered | 1 | 0.307569 | 0.833333 | 1 |
| reach-article-62-application | REACH | answered | 1 | 0 | 1 | 1 |
| reach-article-66-notification | REACH | answered | 1 | 0.200113 | 1 | 1 |
| reach-article-67-restriction | REACH | insufficient_evidence | null | null | null | null |
| reach-article-75-agency | REACH | answered | 1 | 0.107373 | 1 | 1 |
| reach-article-126-penalties | REACH | abstained | null | null | null | null |
| reach-article-141-entry | REACH | answered | 1 | 0.452148 | 1 | 1 |
| reach-annex-iv-water | REACH | insufficient_evidence | null | null | null | null |
| reach-annex-xvii-export | REACH | abstained | null | null | null | null |

## Lowest-scoring answered questions

### rohs-article-25-transposition

By what date must Member States adopt and publish the laws necessary to comply with the RoHS Directive under Article 25(1)?

Status `answered`. Faithfulness 0, answer relevancy 0.224599, context precision 1, context recall 1.

### rohs-article-27-entry

When does the RoHS Directive enter into force under Article 27?

Status `answered`. Faithfulness 0, answer relevancy 0.366108, context precision 1, context recall 1.

### rohs-article-9-importers

What does RoHS Article 9(a) require importers to place on the Union market?

Status `answered`. Faithfulness 0, answer relevancy 0.580146, context precision 1, context recall 1.

### rohs-article-16-presumption

What does RoHS Article 16(1) say Member States shall presume about EEE bearing the CE marking?

Status `answered`. Faithfulness 0, answer relevancy 0.768712, context precision 1, context recall 1.

### reach-article-62-application

To whom must an application for an authorisation be made under REACH Article 62(1)?

Status `answered`. Faithfulness 1, answer relevancy 0, context precision 1, context recall 1.


## Failure examples

- `rohs-eee-definition` (abstained): The answer omits an application condition or exception present in the cited evidence: shall apply.
- `rohs-article-4-concentration` (insufficient_evidence): No chunks were retrieved for RoHS Article 4.
- `rohs-article-6-review` (insufficient_evidence): No chunks were retrieved for RoHS Article 6.
- `rohs-article-23-penalties` (abstained): The answer alters a numerical value from the cited evidence: 2 January 2013.
- `rohs-lead-limit` (abstained): The answer presents a partial list as exhaustive. The cited evidence also contains: Lead (0,1 %).
- `rohs-cadmium-limit` (abstained): The answer presents a partial list as exhaustive. The cited evidence also contains: Cadmium (0,01 %).
- `rohs-dehp-medical-date` (abstained): The answer alters a numerical value from the cited evidence: 22 July 2021.
- `rohs-phthalate-toys` (clarification_required): The question names both RoHS and REACH.
- `rohs-article-4-and-lead` (insufficient_evidence): No chunks were retrieved for RoHS Article 4.
- `reach-article-56-annex-xiv` (insufficient_evidence): No chunks were retrieved for REACH Article 56.
- `reach-article-57-carcinogen` (insufficient_evidence): No chunks were retrieved for REACH Article 57.
- `reach-article-67-restriction` (insufficient_evidence): No chunks were retrieved for REACH Article 67.
- `reach-article-126-penalties` (abstained): The answer alters a numerical value from the cited evidence: 1 December 2008.
- `reach-annex-iv-water` (insufficient_evidence): No chunks were retrieved for REACH Article 2.
- `reach-annex-xvii-export` (abstained): The answer omits an application condition or exception present in the cited evidence: shall not apply.

## Guardrail evaluation

These rates come from `evaluation/guardrail_questions.json`. They are not combined with the RAGAS means.

- Supported acceptance: 1
- Ambiguous clarification: 1
- Unsupported rejection: 1
- False acceptance: 0
- False rejection: 0

## Baseline comparison

RAGAS context precision and context recall judge retrieved text against reference answers on the 50-question file. Phase 5 Recall@5, Precision@5, and MRR rank expected chunk ids on a different 16-question file. The two sets of numbers are not the same metric and this evaluation does not claim that one improved on the other.

Recorded dense Recall@5 0.4, Precision@5 0.18, MRR 0.2533.
Recorded hybrid Recall@5 0.4, Precision@5 0.22, MRR 0.275.

## Curation notes

- A question asking which substances are listed in REACH Annex XIV was not used. Chunk 02006R1907-20260622:01070 contains the annex title and column headings, not substance rows. The authorisation question uses Article 56(1), which states the prohibition for a substance included in Annex XIV and stops at 'unless:'.
- REACH Article 7(1)(b) is not in the indexed chunks 00111 or 00112. The Article 7 question states condition (a) only and does not invent condition (b).

## Limitations

The judge is a language model, so a score is an evaluation judgment rather than a legal conclusion. Answer relevancy in RAGAS 0.4 compares the question with questions generated from the answer and does not read the retrieved context. Context precision and context recall use the reference answer, not the expected chunk ids. A validation abstention keeps the public API reason and does not keep the rejected answer text.
