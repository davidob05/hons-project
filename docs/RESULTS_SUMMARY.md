# Change Detection results so far

Written 23 November 2026. This document explains the track's four main measures and records what the experiments to date found, on the dev topics only.

## Headline findings

1. **Ranking is largely solved by BM25 with the right query.** BM25 over whole documents with the query `{label}: {question} {narrative}` returns every judged-relevant document in its top 20 and scores RR′ 0.884, AP′ 0.500, NDCG′ 0.705, against RR′ 0.756 for the plain question (section 3).
2. **Gating is the hard part.** Gates cut returned volume about 40-fold but find less than half of the relevant documents; every method sits on one trade-off line of about 4 to 5 documents returned per relevant document at 30% to 45% found (sections 4.4 and 6).
3. **Most false alarms are right-topic, wrong-question.** 63% to 85% of the documents returned on nil-answer days are relevant to a sibling question of the same topic that day. 82% of relevant judgments are for documents that answer more than one question (section 6).
4. **Windowed reranking is a real gain, and it now edges past BM25.** A cross-encoder that scores every window of the top 20 and takes the maximum reaches RR′ 0.887 (MiniLM-L12) and 0.890 (monoT5-base) on its own with 200-word windows and the label and question, against 0.884 for BM25's order and 0.775 to 0.816 on the first 100 words. Blended with BM25's order, the official scorer gives RR′ 0.902, AP′ 0.509 and NDCG′ 0.714, the first official improvement over BM25 alone (section 5).
5. **The narrative hurts the reranker, badly for monoT5; the question alone separates sibling questions best.** With the narrative monoT5's RR′ falls to 0.49 to 0.58. monoT5 with the question alone and windows separates a question's own relevant documents from its siblings' with AUC 0.747, the best figure so far (0.671 for the best ranking cell, 0.656 before windows), and gives the best day-level AUC (0.691) (section 5).
6. **No length bias from windowing.** The score's rank correlation with document length is about +0.06 to +0.13 within a question-day (up to +0.19 with the narrative), and the longest document is top only 1% to 3% of the time (up to 8% with the narrative) (section 5).
7. **Reading-comprehension models do not help.** SQuAD 2.0-style models (RoBERTa-base, MiniLM, ELECTRA-large) were tested as the question-level signal and are weaker than monoT5 on every measure (separation from sibling-only documents 0.64 to 0.67 against 0.747; right question first 34% to 42% against 53%). Combined with monoT5 they add almost nothing (section 5).
8. **Next experiment:** a two-signal design. One configuration (label + question, 200-word windows) decides what to return; a second (monoT5, question only, windows) decides which question a document belongs to (section 8 and `docs/PLAN_GX_FOLLOWUP.md`).

## 1. What these results are

- **Scorer.** Every number comes from the track's official scorer (`cdet-scorer 1.1.0`, kept in `ignored/`), run through the thin adapter `src/hons/scripts/score.py`. The adapter turns our judgments into the scorer's qrels format and calls the scorer on a run's `run.json`.
- **Judgments.** `dev` is the 377 relevant documents in `data/dev-topics-all-docs.jsonl`. `combined` adds our 23 silver judgments, giving 400. Unless a table says otherwise, numbers are on `combined`. Dev-only values differ by 0.01 or less.
- **Topics.** Two dev topics (`cd2026_001`, French election, 9 questions; `cd2026_002`, imaging satellite proliferation, 15 questions), 24 questions in all.
- **Evaluation points.** The scorer evaluates every question on every day that has at least one judgment: 137 days, 1,749 question-days. Of these, 368 are answerable (a relevant document exists that day) and 1,381 are nil-answer.
- **Lists.** Unless stated, a run returns its top 20 documents per question per day.

## 2. The four measures

The track's four prime measures are RR′, AP′, NDCG′ and RBP′ (p = 0.8). They are the modified measures of Liu, Moffat, Baldwin and Zhang, "Quit While Ahead: Evaluating Truncated Rankings" (SIGIR 2016), which allow a system to return a short list or nothing. I also report plain AP, which is our own addition, and the two nil-day measures, abstention and noise.

### 2.1 In plain words

- **RR′ (reciprocal rank).** How high up is the first right answer? 1.0 means it is first, 0.5 means second, 0.33 third, and 0 means it is not in the list.
- **AP′ (average precision).** How well are all the right answers packed near the top, without wrong ones in between? It also gives credit for stopping early, so a shorter list is worth more than a longer one with the same hits. Its best possible value is well below 1 at depth 20.
- **NDCG′ (normalised discounted cumulative gain).** Like AP′, but it gives more credit for more valuable documents (action over assessment over awareness) and discounts lower ranks smoothly. It's compared with the best possible ordering.
- **RBP′ (rank-biased precision).** Imagine a reader who moves to the next document with probability 0.8. It adds up the value of what the reader sees. With our binary dev judgments it sits near 0.02 for every run, so it tells us nothing and I leave it out of the dev tables.
- **Abstention and noise.** On nil-answer question-days (no relevant document exists), abstention is the share of days where the run returned nothing (higher is better), and noise is the average number of documents returned (lower is better).

### 2.2 In more detail

**Which question-days each measure covers.**
- RR′, AP′, NDCG′ and RBP′ average over **answerable** question-days only, pooled across all topics and days (368 here).
- Abstention and noise average over the **nil-answer** question-days of judged days (1,381 here). The roughly 970 days with no judgment at all are ignored.
- If a run returns nothing on an answerable day, that day scores 0 on all four measures. That is why gated runs score so much lower than ungated ones: the score is mostly the share of answerable days where the gate said anything.
- The scorer sorts each ranking by the `score` field and reads at most 100 documents.

**The terminal document.** The prime measures append one imaginary document to the end of a ranking. Its gain is r_t, the fraction of the relevant documents the ranking has retrieved (r_t = 1 when all of them were found). A short list that has found everything is therefore rewarded for stopping, and a long list is penalised for the documents it adds.

**RR′.** 1 / (rank of the first relevant document), or 0 if none is in the list.

**AP′.**
- Binary gains: a document counts 1 if relevant at any level, otherwise 0.
- Sum precision at each relevant position, including the terminal document with gain r_t, then divide by (R + 1), where R is the number of relevant documents.
- The consequence: with R = 1, the relevant document at rank 1 and a list of 20, AP′ = (1 + 2/21) / 2 = 0.548. So 0.548 is the best possible score in that case, and our best run's 0.500 is about 91% of it.

**NDCG′.**
- DCG = sum of gain / log2(rank + 1), with gains from the qrels (awareness 1, assessment 5, action 20) and the terminal document added.
- It is divided by the ideal DCG of the same documents, with an extra gain-1 document added to the ideal list.
- With R = 1, the relevant document at rank 1 and a list of 20, NDCG′ = 1.224 / 1.631 = 0.751. Our best run's 0.705 is about 94% of it. With graded qrels the ceilings differ.

**RBP′ (p = 0.8).** (1 − p) × sum of gain × p^(rank − 1), plus the terminal term r_t × p^n, with gains divided by 20 (the action level). Our dev qrels are all at the lowest level, so each document is worth 0.05 and RBP′ stays near 0.02. The official test qrels have graded levels, where it will carry information.

**Plain AP (ours).** The standard average precision: the mean of the precision at each relevant document, divided by R, without a terminal document. It lies in [0, 1]. Most answerable question-days here have a single relevant document, and for one relevant document AP equals RR, so plain AP stays very close to RR′.

**Other counts the scorer reports.** `num_ret` (documents returned), `num_rel_ret` (relevant documents returned), `unjudged` (returned documents with no judgment). Dev judgments list only relevant documents, so an unjudged document is treated as not relevant. All scores are therefore lower bounds.

## 3. Ranking results (top 20, no gate)

Whole-document BM25 (Terrier, per-day statistics) unless stated.

| Configuration | RR′ | AP | AP′ | NDCG′ | Relevant in top 20 (of 400) |
| --- | --- | --- | --- | --- | --- |
| BM25, plain question | 0.756 | 0.751 | 0.430 | 0.631 | 383 |
| BM25, question + narrative | 0.860 | 0.856 | 0.486 | 0.691 | 397 |
| BM25, label + question | 0.873 | 0.866 | 0.491 | 0.695 | 396 |
| **BM25, label + question + narrative** | **0.884** | **0.881** | **0.500** | **0.705** | **400** |
| Fused BM25 + MiniLM, 200-word windows, label (RRF) | 0.852 | 0.847 | 0.482 | 0.689 | 400 |
| Fused BM25 + MiniLM z-sum + MiniLM cross-encoder, plain question | 0.836 | 0.832 | 0.470 | 0.671 | 385 |
| Fused (RRF) + MiniLM cross-encoder blend, plain question | 0.816 | 0.810 | 0.459 | 0.659 | 382 |
| Dense MiniLM, 100-word windows, with narrative | 0.794 | 0.791 | 0.453 | 0.659 | 396 |
| Fused BM25 + MiniLM, 300-word windows (RRF), plain question | 0.781 | 0.777 | 0.443 | 0.643 | 382 |

The next rows rerank the top 20 of the best BM25 run (label + question + narrative), ordered by the reranker's score alone. The candidates are the same 20, so all of them contain the same 400 relevant documents. The full grid is in section 5.

| Reranker, query and passage (reranker order alone) | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- |
| monoT5-base, label + question, 200-word windows | **0.890** | **0.501** | **0.707** |
| MiniLM-L12, label + question, 200-word windows | 0.887 | 0.500 | 0.706 |
| monoT5-base, label + question, 100-word windows | 0.877 | 0.494 | 0.701 |
| MiniLM-L12, label + question, 100-word windows | 0.870 | 0.493 | 0.699 |
| monoT5-base, question only, 200-word windows | 0.851 | 0.481 | 0.688 |
| MiniLM-L12, label + question, first 100 words | 0.816 | 0.463 | 0.672 |
| monoT5-base, label + question, first 100 words | 0.764 | 0.436 | 0.645 |
| monoT5-base, label + question + narrative, 100-word windows | 0.536 | 0.323 | 0.537 |

Official scores of the windowed MiniLM-L12 order blended with BM25's order (reciprocal rank fusion):

| Blend | RR′ | AP | AP′ | NDCG′ |
| --- | --- | --- | --- | --- |
| BM25 label + question + narrative alone | 0.884 | 0.881 | 0.500 | 0.705 |
| + MiniLM-L12, label + question, 100-word windows | **0.902** | **0.899** | **0.509** | **0.714** |
| + MiniLM-L12, question only, 100-word windows | 0.881 | 0.878 | 0.498 | 0.703 |

The gain of the best blend over BM25 alone is +0.018 RR′ and +0.009 in AP′ and NDCG′. It has not been tested for significance on two topics.

Dev judgments only, for the best and the plain baseline: label + question + narrative BM25 scores RR′ 0.880, AP 0.877, AP′ 0.498, NDCG′ 0.704 with 377 of 377 relevant documents in the top 20. The plain-question BM25 scores 0.743, 0.737, 0.424 and 0.624 with 360 of 377.

**What this shows**
- The query text matters most. Adding the topic label and narrative to the BM25 query lifts RR′ by 0.13 and puts every judged relevant document in the top 20.
- Windows, dense scoring, fusion and reranking did not beat BM25 with the rich query.
- The first relevant document is typically at rank 1 or 2 (RR′ 0.88). AP′ and NDCG′ are within about 91% to 94% of their ceilings for one relevant document.
- Every ungated run returns 34,980 documents (1,749 question-days × 20), so the last column is also its recall. Documents returned against relevant ones is set out in section 4.4.

### 3.1 What maximises the four measures

RR′, AP′, NDCG′ and RBP′ are averaged over answerable question-days only, so nil-answer days never lower them (they only affect abstention and noise). Two things therefore raise them: the best ranking, and the list length, because AP′ and NDCG′ reward a short list that already holds the relevant document (the terminal-document design, section 2.2). Official scores, combined judgments, top-k of each ranking returned for every question on every day (no gate):

| Ranking | Depth | RR′ | AP′ | NDCG′ | Relevant kept (of 400) |
| --- | --- | --- | --- | --- | --- |
| BM25 label + question + narrative | 20 | 0.884 | 0.500 | 0.705 | 400 |
| monoT5, label + question, 200-word windows, alone | 20 | 0.890 | 0.501 | 0.707 | 400 |
| BM25 + monoT5 (rank fusion) | 20 | 0.902 | 0.509 | 0.714 | 400 |
| BM25 + MiniLM-L12 (rank fusion) | 20 | 0.904 | 0.509 | 0.714 | 400 |
| BM25 + monoT5 + MiniLM-L12 (rank fusion) | 20 | **0.926** | **0.520** | **0.724** | 400 |
| BM25 + monoT5 + MiniLM-L12 | 10 | 0.926 | 0.563 | 0.757 | 400 |
| BM25 + monoT5 + MiniLM-L12 | 5 | 0.926 | 0.637 | 0.802 | 398 |
| BM25 + monoT5 + MiniLM-L12 | 3 | 0.923 | 0.712 | 0.839 | 388 |
| BM25 + monoT5 + MiniLM-L12 | 2 | 0.916 | 0.776 | **0.864** | 374 |
| BM25 + monoT5 + MiniLM-L12 | 1 | 0.864 | **0.827** | 0.851 | 318 |
| BM25 alone | 3 | 0.882 | 0.691 | 0.817 | 386 |
| BM25 alone | 2 | 0.874 | 0.752 | 0.841 | 371 |

(monoT5 and MiniLM-L12 here are the label + question, 200-word-window cells; the fusion is reciprocal rank fusion with equal weights.)

- **The best ranking is a three-way fusion of BM25, monoT5 and MiniLM-L12:** RR′ 0.926, AP′ 0.520, NDCG′ 0.724 at depth 20, against 0.884, 0.500 and 0.705 for BM25 alone.
- **A shorter list raises AP′ and NDCG′ a great deal without losing much.** At depth 3 the same fusion scores RR′ 0.923, AP′ 0.712, NDCG′ 0.839; at depth 2, RR′ 0.916, AP′ 0.776, NDCG′ 0.864 (the NDCG′ maximum). RR′ saturates at depth 5, and AP′ is highest at depth 1 (0.827) where RR′ drops to 0.864.
- **There is no abstention in this setting.** Noise equals the depth (2 to 3 documents per nil-answer day, against 20 at full depth) and abstention is 0. Gating buys abstention and lower noise, but it lowers the four measures (the best gated run reaches RR′ about 0.41), so which to maximise depends on how the track weights the measures.
- **Caveats.** The depth and the equal fusion weights were chosen by looking at dev, on two topics. The official qrels are graded, so NDCG′ and RBP′ will behave differently there.

### 3.2 Adding the monoT5 sibling filter to the fusion (official scorer)

The filter keeps a document for a question only if monoT5 (question only, 200-word windows) scores that question within a margin (in log-probability units) of the document's best question. It was applied to the three-way fusion (BM25 + monoT5 + MiniLM-L12), alone and after the BM25 history gate.

**Straight onto the ungated fusion, filter applied after the depth cut:**

| Setting | Returned | Relevant | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| depth 3, no filter | 5,247 | 388 | 0.00 | 3.0 | 0.923 | 0.712 | 0.839 |
| depth 3, margin 8 | 3,875 | 325 | 0.03 | 2.2 | 0.782 | 0.622 | 0.718 |
| depth 3, margin 4 | 2,596 | 230 | 0.17 | 1.4 | 0.559 | 0.465 | 0.522 |
| depth 2, no filter | 3,498 | 374 | 0.00 | 2.0 | 0.916 | 0.776 | 0.864 |
| depth 2, margin 8 | 2,478 | 315 | 0.12 | 1.4 | 0.777 | 0.675 | 0.741 |
| depth 2, margin 4 | 1,581 | 222 | 0.34 | 0.8 | 0.556 | 0.498 | 0.536 |

This is a poor way to buy abstention: the filter removes relevant documents about as fast as wrong ones (monoT5 ranks a right question first only 53% of the time), so RR′ falls by 14 to 36 points for modest abstention. Applying the filter before the depth cut (so lower-ranked documents refill the list) gives no abstention at all and the same loss.

**After the BM25 history gate, then the fusion at depth 2 or 3, then the filter:**

| Setting | Returned | Relevant | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q0.8 gate, fusion depth 2, no filter | 1,363 | 257 | 0.38 | 0.7 | 0.639 | 0.605 | 0.631 |
| q0.8 gate, fusion depth 2, margin 8 | 936 | 220 | 0.61 | 0.5 | 0.543 | 0.518 | 0.539 |
| q0.8 gate, fusion depth 2, margin 6 | 725 | 190 | 0.70 | 0.3 | 0.469 | 0.443 | 0.465 |
| q0.9 gate, fusion depth 2, no filter | 826 | 194 | 0.62 | 0.4 | 0.501 | 0.470 | 0.492 |
| q0.9 gate, fusion depth 2, margin 8 | 588 | 169 | 0.75 | 0.3 | 0.433 | 0.410 | 0.427 |
| q0.9 gate, fusion depth 2, margin 6 | 463 | 143 | 0.81 | 0.2 | 0.368 | 0.343 | 0.362 |

(Depth 3 gives almost the same numbers: after a gate there are rarely more than two documents.) The history gate is a much better way to buy abstention than the filter alone (abstention 0.38 at RR′ 0.639, against 0.34 at RR′ 0.556 for the filter on the ungated fusion). After the gate, the filter adds value: at abstention about 0.61, the q0.8 gate with margin 8 scores RR′ 0.543 and NDCG′ 0.539, against 0.501 and 0.492 for the q0.9 gate alone. The full stack (gate, fusion, filter) with abstention 0.70 and RR′ 0.469 also beats the best earlier selective runs at similar abstention (MiniLM with a MAD gate: abstention 0.72, RR′ 0.408; monoT5 with a MAD gate: 0.75 and 0.400).

### 3.3 Document-first design: the gate and fusion pick documents, monoT5 on the question alone only picks questions (official scorer)

Here the BM25 history gate decides which documents are worth returning on a topic-day (a document that passes the gate for any question of the topic counts). The three-way fusion only orders documents within a question's list. monoT5 on the question alone (200-word windows) is used for one thing: deciding which questions a chosen document is returned for. A document goes to a question only if that question is within a margin (log-probability units) of the document's best question, or among its top m questions. Lists are cut at depth 3. Official scores, combined judgments:

| Setting | Returned | Relevant | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ungated fusion, depth 3 (for reference) | 5,247 | 388 | 0.00 | 3.0 | 0.923 | 0.712 | 0.839 |
| q0.8 gate, no assignment filter | 2,252 | 325 | 0.15 | 1.2 | 0.787 | 0.724 | 0.767 |
| q0.8 gate, margin 8 | 1,394 | 271 | 0.47 | 0.7 | 0.658 | 0.618 | 0.648 |
| q0.8 gate, margin 6 | 1,052 | 229 | 0.60 | 0.5 | 0.556 | 0.521 | 0.547 |
| q0.8 gate, margin 4 | 791 | 192 | 0.70 | 0.4 | 0.465 | 0.440 | 0.459 |
| q0.8 gate, top 3 questions | 613 | 143 | 0.77 | 0.3 | 0.359 | 0.335 | 0.352 |
| q0.8 gate, top 2 questions | 414 | 112 | 0.85 | 0.2 | 0.285 | 0.263 | 0.279 |
| q0.9 gate, margin 8 | 889 | 196 | 0.63 | 0.4 | 0.488 | 0.458 | 0.477 |
| q0.9 gate, margin 6 | 674 | 169 | 0.72 | 0.3 | 0.421 | 0.391 | 0.412 |
| q0.9 gate, margin 4 | 483 | 137 | 0.81 | 0.2 | 0.342 | 0.320 | 0.336 |

- **The margin is a single knob that trades the four measures against abstention and noise.** At abstention 0.60 it scores RR′ 0.556, AP′ 0.521, NDCG′ 0.547; at 0.70, 0.465, 0.440, 0.459.
- **It beats the earlier selective runs at the same abstention.** At abstention 0.70 to 0.72 it gives RR′ 0.465 (q0.8 gate, margin 4), against 0.408 for MiniLM with a MAD gate (0.72) and 0.400 for monoT5 with a MAD gate (0.75). At abstention about 0.60 it gives RR′ 0.556, NDCG′ 0.547, against 0.501 and 0.492 for the q0.9 gate alone (0.62).
- **Put the selectivity in the assignment, not in the gate.** A looser gate with a tighter margin beats a stricter gate: at abstention 0.70 to 0.72, q0.8 with margin 4 scores RR′ 0.465, against 0.421 for q0.9 with margin 6.
- **It matches the earlier "gate, then per-question filter" design at equal abstention** (q0.8 with margin 6 gave RR′ 0.469 at 0.70 there), so the document-first form is a cleaner way to get the same trade-off and slightly better in the middle (abstention 0.60 to 0.61: RR′ 0.556 here against 0.543).
- **Caveats:** the margins and the gate quantile were chosen by looking at dev, on two topics. Assignment uses monoT5 scores only for questions where the document was in the top 20, which is partial.

### 3.4 Does the document-first design behave the same on both topics? (official scorer, per topic)

The French election topic has 9 questions and the satellite topic has 15 near-identical ones. With the q0.8 gate and the margin rule:

| Setting | Topic | Returned | Relevant (% found) | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no assignment filter | French election | 902 | 126 (87%) | 0.06 | 1.7 | 0.871 | 0.753 | 0.827 |
| no assignment filter | Satellites | 1,350 | 199 (78%) | 0.18 | 1.1 | 0.746 | 0.710 | 0.738 |
| margin 8 | French election | 660 | 115 (79%) | 0.27 | 1.2 | 0.800 | 0.700 | 0.764 |
| margin 8 | Satellites | 734 | 156 (61%) | 0.54 | 0.5 | 0.589 | 0.579 | 0.592 |
| margin 6 | French election | 538 | 107 (74%) | 0.39 | 0.9 | 0.742 | 0.659 | 0.713 |
| margin 6 | Satellites | 514 | 122 (48%) | 0.67 | 0.4 | 0.466 | 0.454 | 0.467 |
| margin 4 | French election | 436 | 100 (69%) | 0.50 | 0.7 | 0.696 | 0.627 | 0.671 |
| margin 4 | Satellites | 355 | 92 (36%) | 0.78 | 0.2 | 0.353 | 0.350 | 0.356 |

- **The same margin gives very different operating points.** At margin 6 the election topic abstains on 39% of its no-answer days and keeps 74% of the relevant documents (RR′ 0.742), while the satellite topic abstains on 67% and keeps 48% (RR′ 0.466). The aggregate numbers in section 3.3 hide this.
- **Both topics sit on roughly the same abstention-against-RR′ curve,** with the election topic somewhat higher. What differs is where a given margin lands on it. With 15 near-identical questions, the satellite topic makes monoT5's scores for a document's questions closer together, so the same margin cuts more.

A rule that scales with the topic's size keeps the top fraction of a topic's questions for each document (q0.8 gate, depth 3):

| Fraction of the topic's questions kept | Topic (questions kept) | Relevant (% found) | Abstain | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- |
| 25% | French election (2 of 9) | 53 (37%) | 0.71 | 0.400 | 0.344 | 0.380 |
| 25% | Satellites (4 of 15) | 89 (35%) | 0.77 | 0.341 | 0.333 | 0.342 |
| 25% | both | 142 (36%) | 0.76 | 0.360 | 0.337 | 0.354 |
| 50% | French election (4 of 9) | 81 (56%) | 0.45 | 0.583 | 0.520 | 0.557 |
| 50% | Satellites (8 of 15) | 139 (55%) | 0.54 | 0.529 | 0.512 | 0.527 |
| 50% | both | 220 (55%) | 0.51 | 0.547 | 0.514 | 0.537 |

- **The fraction rule is much more consistent across topics** (at 50%: 56% against 55% of relevant documents found; RR′ 0.583 against 0.529), so its behaviour is more predictable on new topics.
- **It is less efficient overall.** At abstention about 0.70 it scores RR′ about 0.42, against 0.465 for the margin rule (margin 4). So the margin rule is better on average and the fraction rule is more predictable.
- **For the test topics (6 to 26 questions each),** I would expect the margin rule's operating point to vary a lot between topics. Choosing the rule by target behaviour, not a fixed margin, is the safer plan.

### 3.5 Topic first: pick the documents for the topic, then the questions

The design idea: search BM25 once per topic, shortlist a handful of documents for the topic-day, then rank those against the topic's questions. Tested with a stand-in, because we ran only a search per question: a document's topic score is its best BM25 score over the topic's questions (each query already contains the label and the narrative).

- **A topic-level shortlist is small.** The top 5 documents per topic-day (by that score) hold 156 of the 158 documents judged relevant to at least one question (99%); the top 10 hold all 158. Question-first lists hold about 42 distinct candidates per topic-day.
- **Ranking is unchanged.** Topic-first without a gate, with the fusion ordering and depth 3, scores RR′ 0.926, AP′ 0.714, NDCG′ 0.841 (top 10) and 0.923, 0.712, 0.839 (top 20), the same as question-first.
- **The assignment filter without a gate refills the lists** (documents are chosen per topic, then each question takes its first three assigned documents), so abstention stays at 0.00 and RR′ falls to 0.56 to 0.79 (margins 4 to 8). The abstention in section 3.3 comes from the gate, not from the question assignment alone.
- **The document-first design in section 3.3 is the topic-first design with a gate** (a document passing for any question counts for the topic-day). What has not been tested is BM25 on the topic text alone (label and narrative, no question words), which may miss documents that match a specific aspect.
- **A topic-level gate can be tested on dev** if days without a listed relevant document are treated as quiet (section 3.6). The official scorer, and our adapter, only score days that have a judgment, so they never see those days.
- **Cost.** 50 BM25 searches a day instead of 665 on the test topics, and 5 to 10 documents per topic-day to rerank instead of 20 per question.

**A real topic-level BM25 search, one query per topic per day** (a Python BM25 with k1 1.2, b 0.75, whole documents and day statistics, close to the Terrier setup but not identical; the 139 judged topic-days run in about 25 seconds). Recall of the 158 documents judged relevant to at least one question, within the topic's top N documents:

| Topic query | Top 3 | Top 5 | Top 10 | Top 20 | Top 30 | Worst rank |
| --- | --- | --- | --- | --- | --- | --- |
| label only | 87% | 92% | 97% | 98% | 99% | 31 |
| label + narrative | 89% | 94% | 98% | 99% | 100% | 25 |
| label + narrative + all the topic's questions | 92% | 98% | 100% | 100% | 100% | 10 |
| label + all the topic's questions (no narrative) | 92% | 97% | 100% | 100% | 100% | 9 |

- **One search per topic works.** Even the label and narrative alone put 94% of the relevant documents in the top 5 and 98% in the top 10; adding the questions' words gives 98% in the top 5 and 100% in the top 10.
- **The questions' words help a little** (98% against 94% in the top 5), so a topic query does not need to be question-free.
- **Limits:** two topics, active days only, and my own BM25 implementation (not Terrier).

**Topic first, end to end (official scorer).** One BM25 search per topic per day (label, narrative and the topic's questions), the top N documents as the shortlist ordered by their topic score, then a question chosen for each document by monoT5 on the question alone (200-word windows), lists cut at depth 3, no gate. Recall of the relevant (question, document) pairs, which sit on only 158 distinct documents, if the document is on the shortlist:

| Shortlist | Distinct relevant documents found | Relevant pairs reachable (of 400) | Shortlist-question pairs already scored by monoT5 |
| --- | --- | --- | --- |
| top 3 | 146 of 158 | 379 | 95% |
| top 5 | 155 of 158 | 395 | 89% |
| top 10 | 158 of 158 | 400 | 77% |
| top 20 | 158 of 158 | 400 | 58% |

| Setting | Returned | Relevant | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top 5 documents, every scored question | 5,196 | 379 | 0.00 | 3.0 | 0.908 | 0.697 | 0.824 |
| top 5, margin 8 | 4,624 | 320 | 0.02 | 2.6 | 0.779 | 0.603 | 0.703 |
| top 5, margin 4 | 3,459 | 228 | 0.12 | 1.9 | 0.560 | 0.446 | 0.510 |
| top 5, top 50% of questions | 3,539 | 273 | 0.11 | 2.0 | 0.674 | 0.552 | 0.622 |
| top 5, top 25% of questions | 2,012 | 177 | 0.37 | 1.1 | 0.450 | 0.387 | 0.426 |
| top 10, every scored question | 5,243 | 379 | 0.00 | 3.0 | 0.908 | 0.696 | 0.823 |
| top 10, top 25% of questions | 2,938 | 179 | 0.22 | 1.6 | 0.454 | 0.375 | 0.420 |

- **Recall is excellent from a short topic-level list.** With 5 documents per topic-day, 395 of the 400 relevant pairs (99%) are reachable; with 10 documents, all 400.
- **Topic-first without a gate nearly matches question-first** (RR′ 0.908 against 0.923, using only BM25's topic order and no neural fusion, and only questions where the document was already scored).
- **Choosing the question among the shortlist is the same moderate signal as before.** The selection rules cut RR′ steeply for modest abstention, because without a gate nothing makes a question stay silent unless every shortlisted document is assigned elsewhere. The document-first runs with the gate in section 3.3 reach abstention 0.60 at RR′ 0.556; here abstention 0.37 costs RR′ 0.450. Abstention has to come from a topic-level gate.
- **Coverage is incomplete.** 11% of the top-5 shortlist-question pairs have no monoT5 score (23% for the top 10), so these results understate what full scoring could do.

### 3.6 A topic-level gate (BM25 on the topic, history-normalised) separates active days from quiet days

Assumption: a day with no relevant document in the dev judgments is a quiet day (unjudged is treated as not relevant). On that basis the French election topic has 56 active days and 1,037 quiet days (5.1% active), and the satellite topic has 83 and 1,010 (7.6%). The signal is the topic-level BM25 score from section 3.5 (one search per topic per day, over all 1,093 days), compared with the topic's own previous 60 days after a 30-day warm-up. AUC of active days against quiet days:

| Topic query | Feature | French election: raw / history z | Satellites: raw / history z |
| --- | --- | --- | --- |
| label + narrative + questions | best score | 0.935 / 0.918 | 0.851 / 0.831 |
| label + narrative + questions | mean of top 5 | 0.936 / 0.919 | 0.816 / 0.787 |
| label + narrative + questions | gap, best minus 10th | 0.940 / 0.927 | 0.841 / 0.824 |
| label + narrative | best score | 0.882 / 0.871 | 0.844 / 0.830 |

History gate on the best topic score (label + narrative + questions):

| Gate | French election: active days fired / quiet days silent | Satellites: active days fired / quiet days silent | Share of all days fired (French / satellites) |
| --- | --- | --- | --- |
| MAD c=1 | 89% / 79% | 64% / 82% | 25% / 22% |
| MAD c=2 | 84% / 89% | 28% / 95% | 15% / 7% |
| quantile 0.8 | 78% / 82% | 60% / 83% | 21% / 20% |
| quantile 0.9 | 62% / 91% | 41% / 92% | 11% / 10% |
| quantile 0.95 | 36% / 96% | 23% / 96% | 5% / 5% |

- **Topic-level detection is far easier than question-level detection.** The topic gate reaches AUC 0.83 to 0.94, against 0.6 to 0.69 for the best question-level signals. The question-level figures were measured only on active topic-days, where the negatives are the topic's other questions, which is a different and harder problem.
- **The raw score is as good as the history-normalised one** here (0.935 against 0.918 for the French topic, 0.851 against 0.831 for satellites), so the history normalisation does not add much at topic level.
- **Newness against recent coverage does not work as a gate.** The novelty of the best topic document compared with the topic's top 5 of the previous 7 or 30 days has AUC 0.29 to 0.42, meaning active days are less new, not more: relevant documents arrive during stretches of continuing coverage. The similarity to recent documents carries signal in the opposite direction (AUC 0.59 to 0.71), which is not what a novelty gate is meant to use.
- **Limits:** two topics; the quiet days are assumed, not judged; the combination of this gate with the question assignment has not been scored end to end across all days.

### 3.7 Every gate tested on all days (monoT5 scores from the four topic-first pods)

Assumption throughout: a day with no relevant document in the dev judgments is a quiet day. For the end-to-end runs, every quiet day gets an added "not relevant" judgment, which is how the official judgments treat judged quiet days, so the official scorer counts abstention and noise over all days. Abstention figures here (over about 2,000 no-answer topic-days of 1,093 days each, mostly quiet days) are therefore not comparable with the abstention figures in sections 3.3 and 4, which only covered the days with a relevant document. Reranker gates can only fire on the days where a loose BM25 topic gate (0.6 quantile of the topic's last 60 days) fired, as in a cascade.

**Detection (AUC of active days against quiet days, French election / satellites; share of active days fired / share of quiet days silent at one setting):**

| Gate | AUC | Fired on active days | Silent on quiet days | Fires on all days |
| --- | --- | --- | --- | --- |
| BM25 topic best score, history | 0.918 / 0.831 | quantile 0.9: 62% / 41% | 91% / 92% | 11% |
| BM25 topic gap (best minus 10th), history | 0.927 / 0.824 | quantile 0.9: 62% / 41% | 91% / 93% | 10% |
| BM25 topic top-5 mean, history | 0.919 / 0.787 | quantile 0.9: 62% / 37% | 91% / 92% | 11% |
| reranker, best pair probability (label + question) | 0.907 / 0.812 | cut 0.2: 87% / 60% | 81% / 86% | 20% |
| reranker, best pair probability (question only) | 0.852 / 0.814 | cut 0.2: 78% / 81% | 77% / 73% | 29% |
| reranker, history gate on best pair (label + question) | 0.843 / 0.795 | quantile 0.8: 62% / 26% | 92% / 94% | 9% |
| reranker, history gate on best pair (question only) | 0.832 / 0.796 | quantile 0.8: 42% / 31% | 93% / 94% | 8% |
| reranker, at least 3 pairs with probability ≥ 0.2 (label + question) | 0.913 / 0.816 | 75% / 27% | 94% / 97% | 7% |
| cascade: BM25 q0.9 AND reranker history quantile 0.5 (label + question) | n/a | 58% / 33% | 94% / 95% | 8% |
| cascade: BM25 q0.95 OR reranker probability ≥ 0.5 (label + question) | n/a | 84% / 53% | 84% / 88% | 18% |

- **The cheap BM25 topic gate is as good as any reranker gate** (AUC 0.92 / 0.83 against 0.85 to 0.91 / 0.80 to 0.82), so the reranker does not improve topic-level detection.
- **MAD gates on the reranker scores almost never fire** (0% to 11% of active days at c = 1), because log-probabilities are heavy-tailed; quantile gates are the usable history form there.
- **Cascades give small trade-offs:** an AND adds silence at the cost of active days found, an OR adds active days found at the cost of silence.

**End to end (official scorer, all days, depth 3, topic shortlist of 5, question rule applied to each question):**

| Topic gate | Question rule | Returned | Relevant | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reranker, best pair probability ≥ 0.2 | none | 22,284 | 321 | 0.72 | 0.83 | **0.762** | 0.585 | 0.691 |
| cascade: BM25 q0.95 OR reranker ≥ 0.5 | none | 19,845 | 315 | 0.76 | 0.73 | 0.746 | 0.573 | 0.677 |
| BM25 topic best score, quantile 0.8 | none | 15,633 | 280 | 0.81 | 0.57 | 0.658 | 0.505 | 0.596 |
| cascade: BM25 q0.8 AND reranker ≥ 0.1 | none | 13,032 | 265 | 0.84 | 0.48 | 0.618 | 0.475 | 0.561 |
| BM25 topic best score, quantile 0.9 | none | 8,055 | 220 | 0.90 | 0.29 | 0.519 | 0.396 | 0.467 |
| BM25 topic best score, MAD c = 2 | none | 7,623 | 208 | 0.91 | 0.27 | 0.484 | 0.369 | 0.436 |
| BM25 topic best score, quantile 0.9 | question history gate 0.5 | 5,058 | 200 | 0.94 | 0.17 | 0.468 | 0.356 | 0.420 |
| reranker, history quantile 0.8 | none | 6,363 | 151 | 0.92 | 0.23 | 0.347 | 0.264 | 0.314 |
| BM25 topic best score, quantile 0.9 | question probability cut 0.1 | 1,719 | 122 | 0.98 | 0.05 | 0.279 | 0.210 | 0.248 |
| BM25 topic best score, quantile 0.95 | margin 6 | 3,767 | 100 | 0.95 | 0.13 | 0.227 | 0.175 | 0.204 |
| BM25 topic best score, quantile 0.8 | margin 6 | 14,320 | 204 | 0.81 | 0.53 | 0.488 | 0.371 | 0.435 |
| BM25 topic best score, quantile 0.9 | margin 6 | 7,361 | 160 | 0.91 | 0.26 | 0.379 | 0.286 | 0.336 |

- **The topic gate sets the abstention, and the question assignment margin no longer helps.** Once quiet days are counted, most no-answer points are questions of silent topics. Adding the margin rule after a topic gate raises abstention by 0 to 1 point and costs 0.14 to 0.17 RR′ and 40 to 76 relevant pairs (quantile 0.9: RR′ 0.519 to 0.379; quantile 0.8: 0.658 to 0.488). The assignment result of sections 3.3 and 3.4 holds only on the active days of the earlier evaluation.
- **A question-level history gate after the topic gate is a better trade-off than the margin:** after the quantile 0.9 topic gate it takes abstention from 0.90 to 0.94 and noise from 0.29 to 0.17 for RR′ 0.519 to 0.468; a fixed probability cut gives abstention 0.98 at RR′ 0.279.
- **The frontier for the four measures against abstention** (no question rule, official scorer, assumed quiet days): reranker cut 0.2 abstains on 72% at RR′ 0.762; the OR cascade 76% at 0.746; the BM25 quantile 0.8 gate 81% at 0.658; the BM25 quantile 0.9 gate 90% at 0.519; the question history gate after it 94% at 0.468.
- **Not tested here:** duplicate suppression and similarity to recent coverage as a topic signal; the LLM verifier; the evaluation is on two dev topics with assumed quiet days, and the question rules use monoT5 scores only on days where the loose gate fired.

### 3.8 What in the topic-first design actually checks that a document is close to a question

In the end-to-end rows of section 3.7 with no question rule, nothing does. When the topic gate fires, the topic's top 3 documents are returned for every question of the topic, so the lists are identical. The only components that compare a document with a specific question are the per-question BM25 search (the question's own words), the per-pair neural scores (monoT5 and MiniLM on one question and one document), the question assignment margin, and the question-level history gate and probability cut.

Putting a question-specific check back into the lists changes the four measures very little (official scorer, all days, depth 3, assumed quiet days):

| Topic gate | List for each question | Relevant | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 quantile 0.9 | the topic's top 3, same for every question | 220 | 0.90 | 0.29 | 0.519 | 0.396 | 0.467 |
| BM25 quantile 0.9 | the topic's top 5 ordered by monoT5 (question only) | 220 | 0.90 | 0.29 | 0.489 | 0.382 | 0.452 |
| BM25 quantile 0.9 | the topic's top 5 ordered by monoT5 (label + question) | 223 | 0.90 | 0.29 | 0.512 | 0.394 | 0.465 |
| BM25 quantile 0.9 | the question's own BM25 top 3 | 225 | 0.90 | 0.29 | 0.509 | 0.396 | 0.465 |
| BM25 quantile 0.8 | the topic's top 3, same for every question | 280 | 0.81 | 0.57 | 0.658 | 0.505 | 0.596 |
| BM25 quantile 0.8 | the topic's top 5 ordered by monoT5 (label + question) | 282 | 0.81 | 0.57 | 0.656 | 0.505 | 0.597 |
| BM25 quantile 0.8 | the question's own BM25 top 3 | 285 | 0.81 | 0.57 | 0.650 | 0.507 | 0.597 |
| reranker probability ≥ 0.2 | the topic's top 3 | 321 | 0.72 | 0.83 | 0.762 | 0.585 | 0.691 |
| reranker probability ≥ 0.2 | the question's own BM25 top 3 | 328 | 0.72 | 0.83 | 0.750 | 0.587 | 0.692 |

- **The four measures cannot tell a question-specific list from the topic's shared list.** Relevant documents are shared across a topic's questions (82% of judgments are for documents that answer several), so the topic's best few documents are relevant for most of the questions that have an answer that day.
- **The measures reward question specificity only through noise on a topic's other questions when it fires.** Counting quiet days, those points are about 5% of all no-answer points, so abstention is set almost entirely by the topic gate.
- **A defensible design costs nothing.** Using each question's own BM25 list after the topic gate gives the same scores as the shared list (RR′ 0.509 against 0.519 at quantile 0.9) and is an actual per-question selection, which the shared list is not.

### 3.9 A per-question system, scored on all days (official scorer, assumed quiet days)

Every list below is built per question. A topic-level gate decides when the topic speaks, then each question gets its own list: the topic's top 5 documents ordered by reciprocal rank fusion of that question's own BM25 score (label, question and narrative) and the monoT5 score for that question (label + question), cut at depth 3. A per-question history gate (the question's best pair score above its own median recent history) decides whether that question speaks.

| System | Returned | Relevant | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| purely per-question: fused + MiniLM, MAD c=3 (earlier run) | 3,590 | 178 | 0.89 | 0.13 | 0.408 | 0.384 | 0.405 |
| purely per-question: fused + monoT5, MAD c=3 (earlier run) | 2,892 | 171 | 0.91 | 0.10 | 0.400 | 0.384 | 0.401 |
| topic gate BM25 q0.8 + per-question lists | 15,633 | 286 | 0.81 | 0.57 | 0.659 | 0.512 | 0.601 |
| topic gate BM25 q0.8 + per-question lists + question history gate | 8,871 | 256 | 0.89 | 0.32 | 0.587 | 0.456 | 0.535 |
| topic gate BM25 q0.9 + per-question lists | 8,055 | 226 | 0.90 | 0.29 | 0.517 | 0.400 | 0.469 |
| topic gate BM25 q0.9 + per-question lists + question history gate | 5,058 | 206 | 0.94 | 0.17 | 0.468 | 0.362 | 0.424 |
| topic gate reranker probability ≥ 0.2 + per-question lists | 15,093 | 297 | 0.82 | 0.55 | 0.686 | 0.532 | 0.626 |
| topic gate reranker probability ≥ 0.2 + per-question lists + question history gate | 8,979 | 267 | 0.89 | 0.32 | 0.617 | 0.478 | 0.561 |

- **At the same abstention (0.89), a topic gate followed by per-question lists and a per-question gate scores RR′ 0.587 to 0.617 against 0.408 for the best purely per-question gate**, at higher noise (0.32 against 0.13), because each firing question returns up to 3 documents. The topic gate is a stronger when-signal than any per-question gate.
- **The per-question history gate is worth keeping:** after the quantile 0.9 topic gate it takes abstention from 0.90 to 0.94 and noise from 0.29 to 0.17 for RR′ 0.517 to 0.468.

**Which question a document answers is still the weak link.** On the 158 documents that answer at least one question, the monoT5 score ranks a right question first for about 54% (question only) and 52% (label + question):

| Score | AUC, relevant to this question against relevant only to a sibling | Right question first |
| --- | --- | --- |
| monoT5, question only (full coverage, active days) | 0.748 | 54% |
| the same, normalised per question | 0.752 | 48% |
| the same, margin to the document's best question | 0.745 | 54% |
| monoT5, label + question | 0.705 | 52% |
| the same, normalised per question | 0.711 | 49% |

- **Normalising the score per question does not help** (0.748 to 0.752, and the right-question-first rate falls from 54% to 48%).
- **On 105 sampled pairs (30 relevant to the question, 75 relevant only to a sibling), the LLM's verdicts separate them better than the automatic scores:** AUC 0.77 (yes / partly / no) and 0.80 (with its confidence), against 0.70 for monoT5, 0.69 for MiniLM and 0.52 for BM25. The sample is small (the interval is about ±0.06), the model is later than the 2021 cut-off, and it is a lead, not a result.

### 3.10 The chosen design: BM25 for finding, monoT5-base for everything neural

Decision: the only neural model is monoT5-base (`castorini/monot5-base-msmarco`), used in two configurations that the experiments picked as best for their jobs.

| Role | monoT5 configuration | Why |
| --- | --- | --- |
| Ordering a question's documents (relevance) | label + question, 200-word windows, best window | highest RR′ of any single reranker (0.890 alone, official scorer); separates relevant from unjudged documents (AUC 0.96) |
| Deciding whether a question speaks (question-level gate) | question only, 200-word windows, best window | best question-level separation (AUC 0.747 against sibling-only documents) and best day-level AUC (0.69) |

Pipeline, per day:

1. **Topic gate (when):** one BM25 search per topic over the whole day (label, narrative and all the topic's questions), then a history gate on the topic's best score (quantile 0.8 for a balanced operating point, 0.9 for higher abstention).
2. **Shortlist:** the topic's top 5 documents.
3. **Each question's list (what):** the 5 documents ordered by reciprocal rank fusion of that question's own BM25 rank and the monoT5 (label + question) rank, cut at depth 3.
4. **Question gate (whether this question speaks):** a history gate on that question's best monoT5 (question only) score over the shortlist, comparing it with the question's own recent days on which the topic fired (quantile 0.5).

Dev results of this design (official scorer, all days, assumed quiet days): quantile 0.8 topic gate, abstention 0.89, noise 0.32, RR′ 0.587, AP′ 0.456, NDCG′ 0.535; quantile 0.9 topic gate, abstention 0.94, noise 0.17, RR′ 0.468, AP′ 0.362, NDCG′ 0.424. Without the question gate: abstention 0.81, RR′ 0.659 (q0.8) and 0.90, RR′ 0.517 (q0.9).

Cost at test scale (estimates): 50 topic searches a day instead of 665, monoT5 only on the days where the topic gate fires (about 20% of days at quantile 0.8) and only for 5 documents against each topic's questions, which is about 3 million window pairs for 50 topics, roughly 2 hours across four GPU pods.

### 3.11 The build: one config, one command

`src/hons/scripts/topic_system.py` runs the whole design of 3.10 day by day (each day only sees its own documents and the earlier days' history) and writes a track-format `run.json` (header with runtag, description, `automatic`, model name and external-data statement, then one line per topic with every day). Every setting is in a TOML file (`configs/final_dev.toml`); the table below is its score with the official scorer.

| Setting | Value |
| --- | --- |
| Topic gate | BM25 best score, history window 60, warm-up 30, quantile 0.8 |
| Shortlist, depth | 5, 3 |
| Question gate | monoT5 (question only) best score, window 60, warm-up 20, quantile 0.5 |

Dev, all days, assumed quiet days, reusing the monoT5 scores of the topic-first pods: abstention 0.895, noise 0.316, RR′ 0.587, AP′ 0.454, NDCG′ 0.533, RBP′ 0.325, 255 of 400 relevant (question, document) pairs found with 8,871 returned. It reproduces the 3.10 figures (the small differences are from using our own BM25 instead of the Terrier run for each question's own rank).

Commands: dev demo `python -m hons.scripts.topic_system --config configs/final_dev.toml`; held-out test topics on the pods `deploy/okd/gx.sh configs/final_test_1.toml hons.scripts.topic_system` (and 2 to 4), then `--merge` the four run files and score with the official qrels.

### 3.12 Comparison with the TREC 2026 participant results

Official figures: 31 assessed test topics, 461 questions, 84,064 judged question-days, judgments provisional (overview Table 1, 35 runs). BM25 here is whole-document, per question (label, question, narrative), day statistics, top 20, no gate, run after the judgments were released. Question order is by top BM25 score. Depth 1 and 3 are cuts of the same list, looked at after seeing the test judgments.

| Run | RBO | RR′ | AP′ | NDCG′ | RBP′ | Abstain | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Our BM25, top 20 | .215 | .833 | .468 | .708 | .036 | .000 | 20.0 |
| Our BM25, depth 3 (post hoc) | .215 | .820 | .636 | .768 | .481 | .000 | 3.0 |
| Our BM25, depth 1 (post hoc) | .215 | .756 | .739 | .748 | .613 | .000 | 1.0 |
| dmnr-harr-t58z05-tr | .235 | .619 | .607 | .615 | .503 | .645 | – |
| ncsu-local | .230 | .469 | .463 | .474 | .390 | .812 | – |
| milgro-cd26-dense | .225 | .841 | .512 | .734 | .128 | .022 | 9.8 |
| milgro-cd26-bm25 | .209 | .829 | .450 | .688 | .045 | .003 | 69.2 |
| milgro-cd26-hybrid | .171 | .769 | .578 | .714 | .398 | .181 | 3.1 |
| bm25-spd-5-22 | .121 | .565 | .545 | .564 | .449 | .595 | – |
| Dave bm25 gemma3 | .161 | .275 | .268 | .272 | .222 | .946 | – |

Among the 35 runs plus ours, the top-20 BM25 run would place 2nd on RR′, 3rd on NDCG′, 4th on RBO, 7th on AP′, 35th on RBP′ and last on abstention.

The gated systems of 3.11 have not been run on the test topics (they need monoT5 on the pods), so their dev figures are not comparable with this table. The one shared dev measure is MilanoGroningen's (377 dev judgments, all 1,093 days, absence of a judgment counted as quiet):

| System | Relevant found of 377 | Proxy negative abstention |
| --- | --- | --- |
| milgro BM25 / dense / hybrid | 370 / 355 / 361 | .121 / .324 / .268 |
| Topic-first, depth 3 | 241 | .894 |
| Topic-first, depth 1 | 189 | .894 |
| Per-question gate, MiniLM / monoT5 | 166 / 158 | .890 / .910 |

## 4. Gating results

A gate decides, for each question and day, which of the top 20 (if any) to return. MAD means the question's bar is median + c × 1.4826 × MAD of its own last 60 daily best scores (silent for the first 30 days). Quantile q uses the q-th quantile of that history instead. A fixed cut uses a constant.

### 4.1 Official-scorer gated runs

Noise is the average documents returned per nil-answer question-day. An ungated run returns 20.0.

| Run | Returned | Relevant | Abstain | Noise | RR′ | AP | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No gate (BM25 label + question + narrative) | 34,980 | 400 | 0.00 | 20.0 | 0.884 | 0.881 | 0.500 | 0.705 |
| BM25 plain question, MAD c=3 | 195 | 63 | 0.94 | 0.1 | 0.148 | 0.144 | 0.141 | 0.147 |
| BM25 label + question, MAD c=3 | 220 | 70 | 0.93 | 0.1 | 0.174 | 0.169 | 0.165 | 0.172 |
| BM25 question + narrative, MAD c=3 | 249 | 78 | 0.91 | 0.1 | 0.192 | 0.181 | 0.180 | 0.190 |
| BM25 label + question + narrative, MAD c=3 | 309 | 82 | 0.88 | 0.1 | 0.196 | 0.185 | 0.180 | 0.193 |
| BM25 label + MiniLM cross-encoder, MAD c=3 | 859 | 175 | 0.71 | 0.4 | 0.399 | 0.390 | 0.377 | 0.396 |

### 4.2 The twelve pod runs (fused candidates, top 20 reranked, plain-question first stage)

| Run | Reranker and score | Gate | Returned | Relevant | Abstain | Noise | RR′ | AP | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `frr_ce_mad3` | MiniLM-L12 logit | MAD c=3 | 857 | 178 | 0.72 | 0.4 | 0.408 | 0.400 | 0.384 | 0.405 |
| `frr_t5_mad3` | monoT5-base | MAD c=3 | 720 | 171 | 0.75 | 0.3 | 0.400 | 0.392 | 0.384 | 0.401 |
| `frr_t5_p1e4` | monoT5-base | fixed cut, P(true) > 0.0001 | 915 | 174 | 0.75 | 0.4 | 0.387 | 0.380 | 0.364 | 0.386 |
| `frr_ce_mad4` | MiniLM-L12 logit | MAD c=4 | 688 | 153 | 0.78 | 0.3 | 0.347 | 0.339 | 0.328 | 0.345 |
| `frr_sum05_mad3` | MiniLM logit + 0.5 × fused z-sum | MAD c=3 | 586 | 139 | 0.80 | 0.3 | 0.323 | 0.317 | 0.308 | 0.322 |
| `frr_sum_q95` | MiniLM logit + fused z-sum | quantile 0.95 | 379 | 113 | 0.83 | 0.2 | 0.298 | 0.296 | 0.294 | 0.298 |
| `frr_t5L_mad3` | monoT5-large | MAD c=3 | 541 | 128 | 0.82 | 0.2 | 0.288 | 0.284 | 0.277 | 0.289 |
| `frr_sum_mad3` | MiniLM logit + fused z-sum | MAD c=3 | 434 | 114 | 0.85 | 0.2 | 0.277 | 0.273 | 0.264 | 0.274 |
| `frr_bge_mad3` | BGE reranker base | MAD c=3 | 786 | 123 | 0.70 | 0.4 | 0.259 | 0.254 | 0.251 | 0.264 |
| `frr_sum_mad4` | MiniLM logit + fused z-sum | MAD c=4 | 284 | 79 | 0.91 | 0.1 | 0.190 | 0.185 | 0.178 | 0.187 |
| `frr_sum_open` | MiniLM logit + fused z-sum | none | 34,980 | 385 | 0.00 | 20.0 | 0.836 | 0.832 | 0.470 | 0.671 |
| `frr_rrf_open` | RRF blend with MiniLM | none | 34,980 | 382 | 0.00 | 20.0 | 0.816 | 0.810 | 0.459 | 0.659 |

### 4.3 History gates replayed on BM25 label + question + narrative

These are exact replays of the pipeline's gates on the ungated run's saved scores. I checked the method by replaying a gate on a reranker run and getting the pod run's result exactly (434 returned and 114 relevant for MAD c=3, 379 and 113 for the quantile 0.95 gate). They are replays, not official-scorer outputs, and the RR′ column is my own calculation.

| Gate | Returned | Relevant | Abstain | Detect | RR′ |
| --- | --- | --- | --- | --- | --- |
| MAD c=1 | 1,554 | 273 | 0.38 | 0.72 | 0.630 |
| MAD c=1.5 | 1,118 | 226 | 0.56 | 0.60 | 0.536 |
| MAD c=2 | 746 | 164 | 0.71 | 0.43 | 0.388 |
| MAD c=3 | 309 | 82 | 0.88 | 0.21 | 0.196 |
| quantile 0.8 | 1,437 | 260 | 0.38 | 0.72 | 0.632 |
| **quantile 0.9** | **829** | **194** | **0.62** | **0.55** | **0.495** |
| quantile 0.95 | 510 | 132 | 0.75 | 0.38 | 0.351 |

"Detect" is the share of answerable question-days where the run returned anything.

**What this shows.** At about 830 documents, BM25 with the rich query and a history gate returns more relevant documents (194) than the best reranker run (178 for 857). The reranker runs abstain more (0.72 to 0.75 against 0.62), but they find fewer relevant documents.

### 4.4 Total documents returned versus relevant documents returned

"Relevant" is the number of judged-relevant documents a run returned, out of the 400 that exist on the 1,749 question-days. "Docs per relevant" is returned divided by relevant (lower is better; 1.0 would mean every returned document is relevant). Documents that are not judged count as not relevant, so these are lower bounds.

**Ungated (the whole top 20 on every question-day)**

| Run | Returned | Relevant | % of the 400 found | Docs per relevant | Share of returned that is relevant |
| --- | --- | --- | --- | --- | --- |
| BM25, label + question + narrative | 34,980 | 400 | 100.0% | 87.5 | 1.1% |
| BM25, label + question | 34,980 | 396 | 99.0% | 88.3 | 1.1% |
| BM25, plain question | 34,980 | 383 | 95.8% | 91.3 | 1.1% |
| Fused + MiniLM z-sum (`frr_sum_open`) | 34,980 | 385 | 96.3% | 90.9 | 1.1% |
| Fused + RRF blend (`frr_rrf_open`) | 34,980 | 382 | 95.5% | 91.6 | 1.1% |
| Top 30 instead of top 20 (fused, MiniLM blend) | 52,470 | 389 | 97.3% | 134.9 | 0.7% |

Going from 20 to 30 documents per question added 17,490 documents and only 7 relevant ones.

**Gated, official-scorer runs, by relevant documents found**

| Run | Returned | Relevant | % of the 400 found | Docs per relevant | Abstain | RR′ |
| --- | --- | --- | --- | --- | --- | --- |
| Fused + MiniLM cross-encoder, MAD c=3 (`frr_ce_mad3`) | 857 | 178 | 44.5% | 4.8 | 0.72 | 0.408 |
| BM25 label + MiniLM cross-encoder, MAD c=3 | 859 | 175 | 43.8% | 4.9 | 0.71 | 0.399 |
| monoT5-base fixed cut, P(true) > 0.0001 (`frr_t5_p1e4`) | 915 | 174 | 43.5% | 5.3 | 0.75 | 0.387 |
| monoT5-base, MAD c=3 (`frr_t5_mad3`) | 720 | 171 | 42.8% | 4.2 | 0.75 | 0.400 |
| MiniLM cross-encoder, MAD c=4 (`frr_ce_mad4`) | 688 | 153 | 38.3% | 4.5 | 0.78 | 0.347 |
| BM25 label + MiniLM, fixed threshold −6 | 876 | 151 | 37.8% | 5.8 | 0.78 | 0.323 |
| MiniLM logit + 0.5 × fused z-sum, MAD c=3 (`frr_sum05_mad3`) | 586 | 139 | 34.8% | 4.2 | 0.80 | 0.323 |
| monoT5-large, MAD c=3 (`frr_t5L_mad3`) | 541 | 128 | 32.0% | 4.2 | 0.82 | 0.288 |
| BGE reranker, MAD c=3 (`frr_bge_mad3`) | 786 | 123 | 30.8% | 6.4 | 0.70 | 0.259 |
| MiniLM logit + fused z-sum, MAD c=3 (`frr_sum_mad3`) | 434 | 114 | 28.5% | 3.8 | 0.85 | 0.277 |
| MiniLM logit + fused z-sum, quantile 0.95 (`frr_sum_q95`) | 379 | 113 | 28.3% | 3.4 | 0.83 | 0.298 |
| Dense MiniLM, 100-word windows with narrative, MAD c=3 | 392 | 92 | 23.0% | 4.3 | 0.85 | 0.220 |
| BM25 (Python, all-day statistics), MAD c=3 | 336 | 87 | 21.8% | 3.9 | 0.88 | 0.204 |
| BM25 label + question + narrative, MAD c=3 | 309 | 82 | 20.5% | 3.8 | 0.88 | 0.196 |
| MiniLM logit + fused z-sum, MAD c=4 (`frr_sum_mad4`) | 284 | 79 | 19.8% | 3.6 | 0.91 | 0.190 |
| BM25 question + narrative, MAD c=3 | 249 | 78 | 19.5% | 3.2 | 0.91 | 0.192 |
| BM25 label + question, MAD c=3 | 220 | 70 | 17.5% | 3.1 | 0.93 | 0.174 |
| BM25 plain question, MAD c=3 | 195 | 63 | 15.8% | 3.1 | 0.94 | 0.148 |
| Fused z-sum (no reranker), MAD c=3 | 179 | 50 | 12.5% | 3.6 | 0.92 | 0.132 |
| BM25 plain question, MAD c=4 | 115 | 40 | 10.0% | 2.9 | 0.96 | 0.099 |
| Fused z-sum (no reranker), MAD c=4 | 81 | 23 | 5.8% | 3.5 | 0.96 | 0.061 |

**Computed by us rather than run through the pipeline**

These are exact replays of the history gates on an ungated run's saved scores (†), and fixed cuts on monoT5-base scores of the top 20 on the judged days (‡). They are not official-scorer outputs.

| Setting | Returned | Relevant | % of the 400 found | Docs per relevant | Abstain | RR′ |
| --- | --- | --- | --- | --- | --- | --- |
| † BM25 rich, MAD c=1 | 1,554 | 273 | 68.3% | 5.7 | 0.38 | 0.630 |
| † BM25 rich, quantile 0.8 | 1,437 | 260 | 65.0% | 5.5 | 0.38 | 0.632 |
| † BM25 rich, MAD c=1.5 | 1,118 | 226 | 56.5% | 4.9 | 0.56 | 0.536 |
| † **BM25 rich, quantile 0.9** | **829** | **194** | **48.5%** | **4.3** | **0.62** | **0.495** |
| † BM25 rich, MAD c=2 | 746 | 164 | 41.0% | 4.5 | 0.71 | 0.388 |
| † BM25 rich, quantile 0.95 | 510 | 132 | 33.0% | 3.9 | 0.75 | 0.351 |
| ‡ monoT5-base, P(true) > 0.0001 (BM25 candidates) | 887 | 173 | 43.3% | 5.1 | 0.75 | 0.385 |
| ‡ monoT5-base, P(true) > 0.001 (fused candidates) | 472 | 121 | 30.3% | 3.9 | 0.87 | 0.265 |
| ‡ monoT5-base, P(true) > 0.05 | 226 | 74 | 18.5% | 3.1 | 0.95 | 0.169 |
| ‡ monoT5-base, P(true) > 0.5 (the plain yes/no cut) | 126 | 51 | 12.8% | 2.5 | 0.98 | 0.117 |

**Where the returned documents go**

| Run | Returned | On answerable days | On nil-answer days | Relevant | Relevant to a sibling question | Unjudged |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 rich, quantile 0.9 (†) | 829 | 240 | 589 | 194 | 516 | 116 |
| MiniLM cross-encoder, MAD c=3 | 857 | 308 | 549 | 178 | 385 | 292 |
| monoT5-base, MAD c=3 | 720 | 264 | 456 | 171 | 310 | 237 |

"Relevant to a sibling question" counts documents judged relevant to another question of the same topic that day, on answerable and nil-answer days together. Reviewed not-relevant and on-topic-only documents (a handful per run) are included in no column.

**How to read this**
- An ungated list returns about 88 to 92 documents for every relevant one (about 1.1% of what is returned is relevant). Gating cuts the volume about 40-fold (34,980 down to about 860) but loses more than half of the relevant documents (100% down to 44%).
- Every setting sits on one trade-off line: about 3 documents per relevant when we find 15% to 20% of them, about 4 to 5 per relevant at 30% to 45%, and about 5.5 to 6 at 65%. None of the gates, rerankers or scores tested moves far off that line, except that BGE (6.4 documents per relevant at 31%) sits clearly above it, meaning worse.
- The best rows on that line are the quantile 0.9 and MAD gates on BM25 with the rich query (829 returned for 194 relevant), monoT5-base with MAD (720 for 171) and MiniLM with MAD (857 for 178).
- Noise per nil-answer day falls from 20.0 documents (ungated) to 0.1 to 0.4 for gated runs.

## 5. What each family of experiments showed

**Query text.** The label and narrative improve BM25 ranking (section 3) but make its score a worse day-level signal. Day-level AUC of the history z-score: rich query 0.591, label only 0.647, MiniLM + fused 0.651.

**Windows, dense and fusion.** Windowing helped the dense scorers, and fusing BM25 with MiniLM over windows matched or beat BM25 on the plain question. None of them beat BM25 with the rich query.

**Reranking (offline, on the plain-question fused candidates, top 20).**
- Blending the cross-encoder rank with the fused rank by reciprocal rank fusion lifted RR′ from 0.781 to 0.812–0.815 (MiniLM-L12). The bootstrap interval on the gain was about +0.02 to +0.05, with question-days improved outnumbering worsened ones 59 to 16.
- Replacing the order with the cross-encoder's alone never beat the baseline (RR′ 0.761 for MiniLM-L12, 0.736 for monoT5-base).
- Top 30 was no better than top 20. monoT5-base was no better than MiniLM-L12, and the blend gain was within noise. This was measured on a weaker base than the rich-query BM25, and it may not carry over to it.

**Windowed reranking of the BM25 top 20 (18 cells: two models, three queries, three passages).** The reranker saw the top 20 of the best BM25 run on the judged days. The passage was the first 100 words, or every 100-word window (stride 50) or 200-word window (stride 100), with the maximum window score taken. The median top-20 document is 639 words long and 89% are over 300 words, so the first 100 words is about a sixth of a typical document.

| Model | Query | Passage | RR′ | AP′ | NDCG′ | RR blended with BM25 | AUC vs sibling-only | AUC vs unjudged | Day-level AUC | Length correlation | Longest document on top |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiniLM-L12 | question | first 100 words | 0.775 | 0.442 | 0.652 | 0.852 | 0.667 | 0.934 | 0.639 | −0.21 | 2% |
| MiniLM-L12 | question | 100-word windows | 0.816 | 0.464 | 0.672 | 0.882 | 0.713 | 0.955 | 0.660 | +0.06 | 3% |
| MiniLM-L12 | question | 200-word windows | 0.828 | 0.470 | 0.678 | 0.882 | 0.712 | 0.957 | 0.661 | +0.07 | 3% |
| MiniLM-L12 | label + question | first 100 words | 0.816 | 0.463 | 0.672 | 0.878 | 0.633 | 0.951 | 0.624 | −0.20 | 2% |
| MiniLM-L12 | label + question | 100-word windows | 0.870 | 0.493 | 0.699 | 0.908 | 0.671 | 0.967 | 0.653 | +0.07 | 2% |
| MiniLM-L12 | label + question | 200-word windows | 0.887 | 0.500 | 0.706 | 0.910 | 0.666 | 0.971 | 0.646 | +0.06 | 2% |
| MiniLM-L12 | label + question + narrative | first 100 words | 0.724 | 0.415 | 0.628 | 0.867 | 0.632 | 0.876 | 0.621 | −0.06 | 4% |
| MiniLM-L12 | label + question + narrative | 100-word windows | 0.733 | 0.422 | 0.633 | 0.874 | 0.660 | 0.889 | 0.629 | +0.19 | 6% |
| MiniLM-L12 | label + question + narrative | 200-word windows | 0.774 | 0.443 | 0.653 | 0.895 | 0.651 | 0.910 | 0.621 | +0.18 | 5% |
| monoT5-base | question | first 100 words | 0.776 | 0.441 | 0.651 | 0.857 | 0.678 | 0.915 | 0.663 | −0.13 | 1% |
| monoT5-base | question | 100-word windows | 0.847 | 0.479 | 0.686 | 0.892 | 0.745 | 0.960 | 0.688 | +0.08 | 3% |
| monoT5-base | question | 200-word windows | 0.851 | 0.481 | 0.688 | 0.897 | **0.747** | 0.962 | **0.691** | +0.08 | 3% |
| monoT5-base | label + question | first 100 words | 0.764 | 0.436 | 0.645 | 0.841 | 0.653 | 0.892 | 0.647 | −0.15 | 2% |
| monoT5-base | label + question | 100-word windows | 0.877 | 0.494 | 0.701 | 0.908 | 0.713 | 0.960 | 0.679 | +0.13 | 2% |
| monoT5-base | label + question | 200-word windows | **0.890** | **0.501** | **0.707** | **0.912** | 0.706 | 0.960 | 0.674 | +0.11 | 2% |
| monoT5-base | label + question + narrative | first 100 words | 0.487 | 0.296 | 0.510 | 0.664 | 0.520 | 0.569 | 0.511 | −0.07 | 4% |
| monoT5-base | label + question + narrative | 100-word windows | 0.536 | 0.323 | 0.537 | 0.766 | 0.552 | 0.633 | 0.534 | +0.18 | 8% |
| monoT5-base | label + question + narrative | 200-word windows | 0.583 | 0.347 | 0.561 | 0.785 | 0.571 | 0.676 | 0.542 | +0.10 | 6% |

The RR′, AP′ and NDCG′ columns are official-scorer figures on the reranker's order alone (combined judgments). The "RR blended" column is my own calculation of the reranker's order fused with BM25's order, so it is not official (the official figure for the MiniLM label + question 100-word blend is 0.902, in section 3). BM25's own order scores RR′ 0.884, AP′ 0.500 and NDCG′ 0.705.

- **Windows beat the first 100 words on every measure, and by more for monoT5.** monoT5 with label + question rises from RR′ 0.764 to 0.877 (100-word windows) and 0.890 (200-word windows); MiniLM rises from 0.816 to 0.870 and 0.887. A windowed reranker alone now matches or just exceeds BM25's own order (0.884).
- **200-word windows are slightly better than 100-word ones** (about +0.01 to +0.02 RR′), at roughly the same cost, since there are half as many windows per document.
- **The best ranking cell is label + question with windows,** for either model (RR′ 0.887 to 0.890; separation from unjudged documents 0.96 to 0.97). The two models are within 0.003 of each other.
- **The best question-level cell is monoT5 with the question alone and windows.** It separates a question's own relevant documents from its siblings' with AUC 0.745 to 0.747, against 0.706 to 0.713 for monoT5 with the label, 0.712 for MiniLM with the question alone, and 0.656 for the earlier monoT5 first-window figure. It also gives the best day-level AUC (0.688 to 0.691).
- **The label helps ranking but costs question specificity** for both models.
- **The narrative hurts both, and badly for monoT5.** monoT5 with the narrative falls to RR′ 0.49 to 0.58 and AUC against unjudged documents to 0.57 to 0.68. The narrative is a long paragraph, far from the short queries monoT5 was trained on (MS MARCO).
- **No length bias.** The rank correlation between a document's score and its length within a question-day is +0.06 to +0.13 with windows (−0.2 to −0.1 with the first 100 words, where shorter documents have more focused openings), up to +0.19 with the narrative. The longest document is top only 1% to 3% of the time (up to 8% with the narrative), against 5% by chance.
- **The day-level AUC stays below 0.7.** These gains improve ranking and question specificity more than the choice of when to stay silent.

**Reading-comprehension models (SQuAD 2.0-style) as the question-level signal.** These models read a question and a passage and score how strongly the passage contains an answer (best answer span against the "no answer" option), the case the question assignment needs. They were run on the same top 20 of the BM25 run and the same judged days, with the question alone (the form they were trained on) over 200-word windows, with the best window taken.

| Model | Query | Passage | RR′ | AP′ | NDCG′ | AUC vs sibling-only | AUC vs unjudged | Day-level AUC | Length correlation | Longest document on top |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RoBERTa-base | question | first 100 words | 0.648 | 0.376 | 0.590 | 0.648 | 0.862 | 0.598 | −0.16 | 3% |
| RoBERTa-base | question | 200-word windows | 0.688 | 0.398 | 0.610 | 0.665 | 0.868 | 0.588 | +0.17 | 5% |
| RoBERTa-base | label + question | 200-word windows | 0.674 | 0.390 | 0.603 | 0.641 | 0.865 | 0.567 | +0.25 | 6% |
| MiniLM | question | 100-word windows | 0.670 | 0.390 | 0.603 | 0.667 | 0.869 | 0.574 | +0.24 | 7% |
| MiniLM | question | 200-word windows | 0.686 | 0.398 | 0.610 | 0.674 | 0.888 | 0.587 | +0.22 | 6% |
| ELECTRA-large (Dec 2020) | question | 200-word windows | 0.663 | 0.386 | 0.599 | 0.666 | 0.864 | 0.567 | +0.23 | 7% |
| *monoT5-base, for comparison* | question | 200-word windows | 0.851 | 0.481 | 0.688 | 0.747 | 0.962 | 0.691 | +0.08 | 3% |

- **Weaker than monoT5 on every measure.** They separate a question's own relevant documents from its siblings' with AUC 0.64 to 0.67 (monoT5 0.747), put the right question first for 34% to 42% of documents (monoT5 53%), and rank the top 20 far worse (RR′ 0.66 to 0.69 against 0.85).
- **A bigger model did not help.** ELECTRA-large is no better than the small MiniLM, and the label did not help either.
- **Some length bias.** The score's rank correlation with document length is +0.17 to +0.25 with windows, and the longest document is top 5% to 7% of the time (monoT5: +0.08 and 3%).
- **They make different mistakes from monoT5 but add almost nothing.** Their scores correlate at 0.46 to 0.49 with monoT5's (MiniLM and monoT5 correlate at 0.77). The best combination, the average of monoT5 and ELECTRA-large (z-scores), raises sibling separation from 0.747 to 0.751 and the right-question-first rate from 53% to 56%, but lowers the day-level AUC (0.681 against 0.691). Requiring both to agree (the minimum) is worse.
- **In the gated test (quantile 0.8 gate, then a margin filter at 1.0) the combination only moves along the same trade-off line:** 508 to 522 returned for 144 to 151 relevant (3.5 documents per relevant), against monoT5 alone at 465 returned for 136 relevant (3.4).
- **Official scorer, monoT5 alone against the joint score (monoT5 + ELECTRA-large):**

| Run | Returned | Relevant | Abstain | Noise | RR′ | AP′ | NDCG′ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| monoT5 alone, ranking | 34,980 | 400 | 0.00 | 20.0 | 0.851 | 0.481 | 0.688 |
| joint, ranking | 34,980 | 400 | 0.00 | 20.0 | 0.845 | 0.477 | 0.685 |
| monoT5 alone, gate q0.8 + margin 1.0 | 465 | 136 | 0.82 | 0.2 | 0.337 | 0.318 | 0.332 |
| joint, gate q0.8 + margin 1.0 | 522 | 151 | 0.79 | 0.2 | 0.372 | 0.353 | 0.368 |

The joint ranking is slightly worse than monoT5 alone. The joint gated run finds 15 more relevant documents for 57 more returned, which is the same exchange rate as before, so it is not a better operating point. The reading-comprehension models were trained on short factual questions about Wikipedia, while ours include yes/no and open questions, and that is the likely reason they do not help here. They are dropped as the question-level signal, and monoT5 with the question alone and windows stays the best.

**Rerankers as gates.** MiniLM-L12 and monoT5-base are about equal. monoT5-large is no better than base (it only ties BM25 alone at its volume), and BGE is the weakest. A history gate beat a fixed cut on the same score (monoT5: 171 relevant for 720 documents with MAD, against 174 for 915 with a fixed cut).

**Combining gates.**
- Merging the relevant documents found by monoT5-large and monoT5-base adds only 16 over base (187 against 171), for 17% more documents.
- An OR of BM25's gate and a reranker's gate gives about +7% to +10% relevant at the same volume. An AND gives about +5% to +8%.
- Averaging day-level scores from different scorers did not lift the AUC above the best single signal (0.652 against 0.651).
- Gating on a different signal from the one that ranks (label-only BM25 or MiniLM + fused, returning the rich BM25 top 3) gave about +5% to +10% relevant at the same volume.

**A learned day-level detector.** Features were the history z-score and percentile, the gap between the top two documents, and how many sibling questions fire the same day. A logistic model trained on one topic and tested on the other reached AUC 0.566, worse than the plain z-score (0.591). An earlier probe's 0.83 did not reproduce, probably because it compared answerable days with quiet days rather than with the other questions on active days.

## 6. Why gating is hard

Of the documents returned on nil-answer days, the large majority are relevant to a different question of the same topic that day:

| Gate | Returned on nil days | Relevant to a sibling question | Unjudged |
| --- | --- | --- | --- |
| BM25 rich, quantile 0.9 | 589 | 499 (85%) | 87 |
| MiniLM cross-encoder, MAD c=3 | 549 | 350 (64%) | 197 |
| monoT5-base, MAD c=3 | 456 | 289 (63%) | 165 |

- The documents are about the right topic. They are false alarms only for the question they were returned for.
- The relevant documents are shared: 158 distinct relevant documents back the 400 relevant judgments (about 2.5 questions each), and 82% of judgments are for documents that answer more than one question that day.
- On a judged day a median of only 2 of the topic's 9 to 15 questions have a relevant document. The other 7 to 13 sibling questions are lexically close, and 71% of a day's top-20 documents appear for several questions.
- How well each score separates a document relevant to this question from one relevant only to a sibling question (AUC): BM25 plain 0.600, BM25 rich 0.555, BM25 question + narrative 0.527, MiniLM cross-encoder 0.632, MiniLM + fused 0.634, monoT5-base 0.656 (first 100 words). With windows, MiniLM-L12 reaches 0.713 with the question alone, 0.671 with the label and 0.660 with the narrative, and monoT5-base reaches 0.745 to 0.747 with the question alone, 0.706 to 0.713 with the label, and 0.55 to 0.57 with the narrative. The narrative lowers question specificity even as it raises topic coverage.
- Competing the questions against each other failed. Giving a document only to its best-scoring question removed most of the relevant documents (cross-encoder, quantile 0.95: 379 returned and 113 relevant became 90 and 40). A softer penalty did worse than no penalty at the same volume.
- The best day-level signal separates answerable from nil days with AUC about 0.66 (monoT5-base on BM25 candidates; PR-AUC 0.39 against a 0.21 base rate).

### 6.1 How much of the noise is sibling returns, and the ceiling

Classifying every returned document on the judged days:

| Run | Returned | Relevant | Relevant to a sibling question | Other (unjudged or off-topic) |
| --- | --- | --- | --- | --- |
| BM25 rich, ungated top 20 | 34,980 | 400 (1%) | 1,536 (4%) | 33,044 (94%) |
| BM25 rich, quantile 0.9 gate (replay) | 829 | 194 (23%) | 516 (62%) | 119 (14%) |
| BM25 rich, quantile 0.95 gate (replay) | 510 | 132 (26%) | 341 (67%) | 37 (7%) |
| MiniLM cross-encoder, MAD c=3 | 857 | 178 (21%) | 385 (45%) | 294 (34%) |
| monoT5-base, MAD c=3 | 720 | 171 (24%) | 310 (43%) | 239 (33%) |

- **Ungated, sibling returns are only 4% of what is returned.** The problem there is volume: 20 documents per question and about one relevant.
- **After gating, nearly all of BM25's remaining noise is siblings.** 81% of the non-relevant documents returned by the quantile 0.9 gate (90% at quantile 0.95) are relevant to another question that day. The rerankers cut the sibling share (56% to 57% of their non-relevant returns) but return more off-topic documents (33% to 34%).
- **Oracle ceiling if every sibling-relevant document were removed** (uses the judgments, so it overstates what a real method could do):

| Gate | Docs per relevant now | If siblings were removed | Abstention now | If siblings were removed |
| --- | --- | --- | --- | --- |
| BM25 rich, quantile 0.9 | 4.3 | 1.6 (313 returned for 194 relevant) | 0.62 | 0.94 |
| BM25 rich, quantile 0.95 | 3.9 | 1.3 | 0.75 | 0.98 |
| MiniLM, MAD c=3 | 4.8 | 2.7 | 0.72 | 0.89 |
| monoT5-base, MAD c=3 | 4.2 | 2.4 | 0.75 | 0.90 |

Detection stays about the same, because siblings are only removed from questions that have no answer that day. A perfect question assignment would therefore take BM25's gate from 4.3 to 1.6 documents per relevant and from 62% to 94% abstention. A real method (AUC about 0.75 on sibling separation) would recover only part of that, and some "sibling-relevant" documents may truly answer the question without having been judged for it.

### 6.2 First test of question assignment (monoT5, question only, 200-word windows)

Each (question, document) pair that the system could return is already in some question's top 20, and the grid scored those pairs, so assignment can be tested without new scoring. Over the 1,936 pairs where the document is relevant to at least one question that day (400 relevant to this question, 1,536 sibling-only; keeping all of them is 21% precision):

| Rule | Typical result (recall of relevant pairs at precision) |
| --- | --- |
| Absolute threshold on P(true) | 0.30 at 0.52; 0.47 at 0.44 |
| Closeness to the document's best question (log-probability margin) | 0.34 at 0.51; 0.47 at 0.44 |
| Gap chain: keep questions in score order until the first gap larger than g | 0.38 at 0.51; 0.59 at 0.35; 0.80 at 0.29 |
| Keep the top m questions of each document | 0.45 at 0.38 |

- **There is no sharp separation.** The best precision is about 0.5, against 0.21 for keeping every pair.
- **An absolute threshold is not consistent across topics.** The median P(true) of relevant pairs is 0.27 for the French election topic and 0.003 for the satellite topic, and the same cut (P ≥ 0.5) keeps 41% and 23% of relevant pairs. The gap chain with a generous gap is the most consistent (recall 0.85 and 0.76 at g = 3), at low precision.

Applied to the output of the BM25 history gate (rich query), the filter moves off the earlier trade-off line:

| Setting | Returned | Relevant (% of 400) | Docs per relevant | Abstain | Earlier line at similar recall (docs per relevant) |
| --- | --- | --- | --- | --- | --- |
| quantile 0.9 gate + P(true) ≥ 0.02 | 285 | 115 (29%) | 2.5 | 0.90 | 3.4 to 3.9 |
| quantile 0.8 gate + P(true) ≥ 0.01 | 485 | 156 (39%) | 3.1 | 0.82 | about 4.5 |
| quantile 0.8 gate + margin ≤ 6 | 755 | 193 (48%) | 3.9 | 0.70 | 4.3 |
| quantile 0.8 gate + gap chain ≤ 3 | 908 | 212 (53%) | 4.3 | 0.62 | 4.9 |

Documents per relevant fall by roughly 10% to 30%, most at low volume, against an oracle ceiling of 1.6. Filter settings are read off a dev sweep on two topics, and topics disagree on absolute thresholds, so a margin or gap rule is the safer choice for new topics.

On each topic separately, the margin rule gives the most consistent documents per relevant (a difference of 0.1 to 0.3 between topics, against 0.3 to 0.9 for the gap chain and absolute cuts), and the history gate underneath is topic-robust (5.5 and 5.5 at quantile 0.8). The recommended starting point is the quantile 0.8 gate with margin ≤ 4 (567 returned, 160 relevant, 40% found, 3.5 documents per relevant, abstention 0.78), with margin ≤ 6 as the higher-recall variant (755 returned, 193 relevant, 48% found, 3.9 documents per relevant, abstention 0.70, RR′ 0.468). Compared with the quantile 0.9 gate alone (829 returned, 194 relevant, abstention 0.62, RR′ 0.495), margin ≤ 6 finds the same number of relevant documents with 9% fewer returned and 8 points more abstention, at a small cost in RR′.

**Assignment quality.** For the 158 documents that are relevant to at least one question and were also scored for a question they are not relevant to, I checked whether the score ranks a relevant question first:

| Score used to rank the questions | Top-1 correct | Chance | R-precision | Best relevant question's average rank |
| --- | --- | --- | --- | --- |
| BM25 rich score | 36% | 21% | 32% | 3.8 of 12.3 |
| monoT5, question only, first 100 words | 59% | 21% | 49% | 2.4 |
| monoT5, question only, 100-word windows | 56% | 21% | 50% | 2.3 |
| monoT5, question only, 200-word windows | 53% | 21% | 47% | 2.3 |
| monoT5, label + question, 200-word windows | 52% | 21% | 45% | 2.3 |
| MiniLM, question only, 200-word windows | 47% | 21% | 43% | 2.7 |
| MiniLM, label + question, 200-word windows | 56% | 21% | 45% | 2.6 |

The score ranks a relevant question first for about 53% to 59% of documents, against 21% by chance, so assignment is a moderate signal. The first 100 words does as well as windows for ranking the questions of a single document (the differences are within noise on 158 documents), although windows help the pooled separation across documents. The two topics are close (for monoT5 question only with windows, 47% and 58%).

**A learned combination of the scores, and a question-level gate.** A small model (logistic regression; also a depth-3 boosted tree) was trained to predict whether a pair is relevant to its question. Features were each scorer's score, how it compares with the document's other questions (margin to the best, rank), how it compares with the question's other documents that day, and optionally the BM25 score and rank. To avoid fitting the two dev topics, it was trained on one topic and tested on the other, in both directions.

| Features | AUC vs sibling-only (mean of the two topics) | Per topic | Right question first |
| --- | --- | --- | --- |
| monoT5 question only, one score, no training | 0.747 | n/a | 53% |
| logistic: monoT5 question only | 0.760 | 0.790 / 0.730 | 55% |
| logistic: + monoT5 with label | 0.742 | 0.785 / 0.699 | 53% |
| logistic: + ELECTRA-large reading comprehension | 0.766 | 0.793 / 0.738 | 49% |
| logistic: all three scorers | 0.752 | 0.792 / 0.712 | 54% |
| logistic: all three + BM25 + number of questions | 0.749 | 0.794 / 0.705 | 53% |
| boosted tree: monoT5 question only | 0.715 | 0.749 / 0.680 | 51% |
| boosted tree: all three + BM25 + number of questions | 0.711 | 0.741 / 0.681 | 52% |

- **A learned combination gains very little** over monoT5's single score (0.747 to 0.760 for monoT5's own features, and 0.766 at best with the reading-comprehension score). The boosted tree is worse, which suggests overfitting to the two topics.
- **Gated result** (BM25 quantile 0.8 gate, then keep pairs whose predicted probability, trained on the other topic, is at least a cut):

| Features | Cut | Returned | Relevant (% of 400) | Docs per relevant | Abstain | RR |
| --- | --- | --- | --- | --- | --- | --- |
| margin rule, monoT5 (for comparison) | margin ≤ 4 | 567 | 160 (40%) | 3.5 | 0.78 | 0.388 |
| margin rule, monoT5 (for comparison) | margin ≤ 6 | 755 | 193 (48%) | 3.9 | 0.70 | 0.468 |
| monoT5 question only | 0.10 | 608 | 179 (45%) | 3.4 | 0.75 | 0.438 |
| all three scorers | 0.10 | 691 | 190 (48%) | 3.6 | 0.69 | 0.465 |
| all three + BM25 + number of questions | 0.20 | 750 | 200 (50%) | 3.8 | 0.65 | 0.496 |
| all three + BM25 + number of questions | 0.10 | 973 | 233 (58%) | 4.2 | 0.54 | 0.571 |

  The learned probability beats the simple margin rule by about 4% to 7% in documents per relevant at matched volume. That is a small gain, from a model trained on the dev labels of the other topic, and two topics are a thin basis.
- **A question-level gate does worse than the document-level margin filter.** After the BM25 quantile 0.8 gate, keeping a question only if its best monoT5 score is within a margin of the best question that topic-day gives 423 returned for 127 relevant (3.3 documents per relevant, 32% found) at margin 1, and 711 for 175 (4.1, 44% found) at margin 4. The document-level margin filter gives about 3.7 at 44% found.
- **History gates (MAD, quantile) cannot be replayed on these scores.** They need each question's top score on every day (or at least every day it passes an earlier gate), and the windowed monoT5 and reading-comprehension scores exist only for the judged days. Faithful replays need new scoring on the pods (section 8).

### 6.3 LLM review of 150 returned documents (first pass, blind)

An LLM judged each of 150 sampled returned (question, document) pairs without seeing our label (prompt in `docs/llm_review/judge_prompt.md`, data in `docs/llm_review/false_alarms_for_llm.jsonl`, verdicts in `docs/llm_review/llm_first_pass.txt`). The sample holds 30 control pairs that the track judges marked relevant, 75 sibling-relevant false alarms and 45 unjudged ones.

| Our label | Day type | Rows | LLM yes | LLM partly | LLM no | yes + partly |
| --- | --- | --- | --- | --- | --- | --- |
| relevant (controls) | answerable | 30 | 37% | 47% | 17% | 83% |
| sibling-relevant | no-answer day | 60 | 10% | 20% | 70% | 30% |
| sibling-relevant | answerable day | 15 | 13% | 27% | 60% | 40% |
| unjudged | no-answer day | 30 | 0% | 3% | 97% | 3% |
| unjudged | answerable day | 15 | 13% | 20% | 67% | 33% |

- **The LLM is stricter than the track judges.** It says "yes" to only 37% of the pairs the judges marked relevant (83% yes or partly), so "yes or partly" is the comparable standard.
- **Most of what we counted as false alarms really are not answers.** 70% of the sibling-relevant returns on no-answer days and 97% of the unjudged ones on no-answer days got "no". The scoring was right to discount them.
- **A minority are not clearly wrong.** 10% (strict) to 30% (yes or partly) of the sibling returns on no-answer days bear on the question asked. In this sample, many confident "yes" verdicts are launch-date or launch-site articles returned for a multi-part question ("when was it launched and what is its life expectancy?") that the judges marked relevant only to other questions. On answerable days, 2 of 15 unjudged documents were a confident "yes" (Le Pen's promises, who Le Pen is), which suggests the judgments are incomplete on days where other documents were judged.
- **Rough effect on precision.** For the BM25 quantile 0.9 gate (829 returned, 194 judged relevant), weighting these rates by the run's own mix of returned documents gives about 56 more documents that answer their question under the strict reading and about 170 under the lenient one. Precision of 23% by the track's judgments would be about 30% (strict) to 44% (lenient).
- **Limits.** 150 rows, two topics, one LLM pass with a 400-word excerpt (only 4 of the 55 "partly" or "yes" verdicts said the answer is probably later in the document). The LLM's own false-positive rate was not measured, since there are no known-wrong controls. The official scorer will keep counting these documents as noise because the judges did not mark them.

## 7. Caveats

- Two dev topics, 24 questions and 368 answerable question-days. Differences under about 0.03 are not reliable, and bootstrap intervals computed over question-days are too narrow, because the question-days are not independent.
- Dev judgments list only relevant documents, so every score is a lower bound and unjudged documents count as not relevant. Abstention and noise are measured only on the nil questions of judged days.
- The two dev topics (French election, imaging satellite proliferation) are not among the test topics. The test topics file reuses the ids cd2026_001 and cd2026_002 for different topics (AI degrees and Mars travel), so the official qrels cannot be used to score the dev runs, and the 31 judged test topics are fully held out from everything tuned here. (An earlier version of this document said the dev topics were among the 31, based on matching ids, which was wrong.)
- The official test qrels hold 6,903 relevant judgments (2,924 distinct documents) across 31 topics, with graded levels (4,960 awareness, 1,689 assessment, 254 action) and 3,495 explicit not-relevant judgments. They make RBP′ informative and score the question ranking, which we have not addressed.
- MiniLM, the Hugging Face monoT5 weights (paper March 2020, Hugging Face copies dated late 2021) and BGE (2023) are labelled "may be contaminated" under the 2021-08-01 rule.
- The AP′ and NDCG′ ceilings quoted in section 2 are for one relevant document and a list of 20.

## 8. In progress and next

- **Done:** the full reranker grid, 18 cells (section 5), and the official score of two blend runs (section 3).
- **Next proposed experiment: a two-signal design.** Use different configurations for different jobs, which the grid now supports.
  - **Job R, what to return:** label + question with 200-word windows (MiniLM-L12 or monoT5-base; RR′ 0.887 to 0.890, separation from unjudged documents 0.96 to 0.97, blended with BM25 it scores officially above BM25 alone). Its best score per question feeds the history gate that decides whether to speak and which documents clear the bar.
  - **Job Q, which question it belongs to:** monoT5-base with the question only and windows (AUC 0.745 to 0.747 against sibling-only documents, the best measured). Each document that passes is scored against every question of its topic and kept only for the questions where it scores high relative to its siblings.
  - **Soft assignment, not hard.** 82% of relevant documents answer more than one question, and a hard "best question only" rule removed most true positives in earlier tests with weaker scores. So keep a document for a question if its score is within a margin of its best, or if the question is in its top one to three.
  - **Cost on dev:** scoring each candidate against all sibling questions is 76,371 document-question pairs, about 1.18M window pairs at 100 words, or roughly half that at 200 words: about 85 minutes for monoT5-base on a pod (estimate). It needs a new cross-question option in the rerank script.
  - **Success means** moving off the trade-off line (about 4 to 5 documents per relevant document at 30% to 45% found), not just along it. If it does not, the assignment idea is dropped and the remaining levers are novelty (Stage 4) and state (Stage 7).
  - Full plan, decision rules and risks are in `docs/PLAN_GX_FOLLOWUP.md`.
- **False alarms for LLM review:** `docs/llm_review/false_alarms_for_llm.jsonl` and `.csv` hold 150 sampled returned documents (60 sibling-relevant and 30 unjudged on nil-answer days, 15 and 15 on answerable days, and 30 correct returns as controls) from the main gated runs, with the question, topic, day, a 400-word excerpt, our label and the other questions each document was judged relevant to.
- **Other candidate steps:** a frozen test run on the official topics (BM25-only first). Seed similarity was set aside as not robust.

## 9. References

- Liu, Moffat, Baldwin, Zhang. "Quit While Ahead: Evaluating Truncated Rankings." SIGIR 2016, pp. 953–956 (DOI 10.1145/2911451.2914737, as cited in the scorer).
- Nogueira, Jiang, Lin. "Document Ranking with a Pretrained Sequence-to-Sequence Model" (monoT5). arXiv 2003.06713, 2020.
- Cormack, Clarke, Büttcher. "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods." SIGIR 2009, pp. 758–759 (recalled from memory, not verified here).
- Dai, Callan. "Deeper Text Understanding for IR with Contextual Neural Language Modeling." SIGIR 2019 (the MaxP passage aggregation for long documents; recalled from memory, not verified here).
