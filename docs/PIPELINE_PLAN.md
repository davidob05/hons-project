# Pipeline implementation plan

Companion to the CDET briefing. This file says how to lay out the code so that every pipeline stage is a swappable, toggleable component. Stage behaviour and options are in the briefing (section 8); they are not repeated here.

## 1. What the codebase gives us (verified, not from the briefing)

- `src/cdet_api/` is upstream code. Leave it unchanged. `pyproject.toml` uses setuptools with a `src` layout, so a new sibling package `src/hons/` is picked up by `pip install -e .` with no build changes.
- `docs.db` exists: 1,000,095 documents over **1,093 days**, not the 1,096 the briefing uses. Use 1,093 in cost estimates.
- Java needs to be installed.
- Settings are read from `settings.toml` with `tomllib`. Use TOML for experiment configs too, to match the repository idiom (the briefing suggests YAML; either works, TOML needs no extra dependency).
- `CDetClient` pitfalls the runner must handle:
  - `next_day()` returns `None` on any non-404 failure (it only prints). The runner must treat `None` as an error and stop, otherwise a day is silently skipped.
  - `retrieval()` prints and swallows errors. A failed post leaves a date missing, which the track reads as "all questions rank 0". Subclass the client in `hons/io/client.py` to raise instead of printing. Do not edit `client.py`.
  - `QuestionResults.question_text` is `str | None` with no default, so always pass it.
- `check-cdet-run.py` already validates a finished run (dates, topics, qids, 100-document limit, duplicates). Reuse it as the Stage 6 validator rather than writing a second one. It now writes its `.errlog` to `logs/`.

## 2. Design

### 2.1 Causality by construction

The live loop never touches `docs.db` directly. Only a `DaySource` does, and it yields one `DayBatch` at a time. Stages receive the `DayBatch` and a `State` handle and nothing else. Offline analysis lives in `hons/analysis/` and is the only code allowed to open `docs.db` whole. Add a test that greps `hons/stages/` and `hons/runner.py` for `sqlite3`, `peewee` and `docs.db` and fails if found.

### 2.2 Stage contract and registry

Every stage has the same shape: an interface (a `Protocol`) with one `run` method, plus a registry keyed by name.

The registry maps a stage name to its implementations, each stored under an implementation name. It needs two operations:
- **Register:** attach an implementation class to a (stage, implementation name) pair, usable as a class decorator.
- **Build:** take a stage's config section and return an instance. If the section has `enabled = false`, return the stage's `off` implementation (the null object, see 2.3). Otherwise look up the class by `impl` and construct it from the remaining config keys as parameters.

### 2.3 Toggle semantics

Each stage config has `enabled` (bool) and `impl` (name) plus its own parameters. `enabled = false` builds the stage's `off` implementation, which is a null object with a defined meaning. Every stage must ship an `off`, so any combination of toggles runs end to end.

| Stage | `off` means |
| --- | --- |
| 0 fragment | one fragment per document (the whole text), no boilerplate flags, no groups |
| 1 questions | version 0 only: question text, no narrative, no seeds |
| 2 score | cannot be off; `impl` chooses (`bm25_day` is the toolkit rule) |
| 3 tail/bar | cannot be off; `impl = "empty"` returns no candidates, `impl = "topk"` is the simplest rule |
| 4 novelty | weight fixed at 1 |
| 5 gate/rank | cannot be off; `impl = "all_candidates"` is the simplest rule |
| 6 output | cannot be off |
| 7 propose | never propose |
| 8 state | stateless: every day recomputed from nothing (carried state is the thing being ablated) |

### 2.4 Data contracts (dataclasses in `hons/types.py`)

| Type | Fields | Notes |
| --- | --- | --- |
| `DayBatch` | day, sequence number, documents | Immutable. Documents use the toolkit's document type |
| `Fragments` | fragment rows; per-document info | Per-document info holds group id, page family and boilerplate share |
| `QuestionVersion` | qid, tid, text, the sequence number it takes effect from, query terms, embeddings, seeds | A question can have several versions over time |
| `Scores` | per (topic:question, document) score | Each score holds the raw score, the best fragment and the standardised score |
| `Candidates` | per question, a list of candidates | A candidate holds doc id, score, excess over the bar, p-value, novelty and weight |
| `DayOutput` | per topic, the toolkit's day-results type | Toolkit type, used only at the boundary |

`Stage 3` returns `Candidates`; `Stage 4` rewrites `.weight`; `Stage 5` turns candidates into `DayResults`. Keep the toolkit types at the boundary only.

### 2.5 Config

One TOML file per experiment in `configs/`. Layout mirrors the stages, so a comparison from briefing section 9 is a one-line diff.

Sections and their options (the first value of each list is not a default; defaults are your choice):

| Section | Key | Options |
| --- | --- | --- |
| `[run]` | `runtag`, `description`, `topics` | free text; `topics` is a path |
| `[run]` | `days` | optional replay range, start and end |
| `[run]` | `mode` | `replay`, `live` |
| `[run]` | `seed` | integer (see section 4) |
| `[s0_fragment]` | `enabled` | on or off |
| `[s1_questions]` | `impl` | `question`, `question_narrative`, `general_template` |
| `[s1_questions]` | `seeds` | on or off |
| `[s2_score]` | `impl` | `bm25_day`, `bm25_acc`, `dense`, `fused` |
| `[s2_score]` | `aggregate` | `best_fragment`, `lead`, `mean_top2`, `weighted` |
| `[s2_score]` | `normalise` | `none`, `zscore_window`, `s1pct` |
| `[s3_tail]` | `impl` | `empty`, `topk`, `fixed`, `quantile`, `surprise`, `spot`, `dspot`, `hybrid` |
| `[s3_tail]` | `q` | tail probability, for example 1e-5 |
| `[s4_novelty]` | `enabled` | on or off (options in section 7) |
| `[s5_gate]` | `impl` | `all_candidates`, `top1`, `best_excess`, `sum_top`, `entropy` |
| `[s5_gate]` | `correction` | `none`, `bonferroni`, `sidak`, `bh` |
| `[s5_gate]` | `group_policy` | `list_all`, `list_best` |
| `[s7_propose]` | `enabled` | on or off (options in section 8) |
| `[s8_state]` | `enabled`, `window_days` | on or off; window length in days |
| `[models]` | `used` | list of models; copied into run metadata automatically |

The config is hashed into the run directory name (`runs/<runtag>-<hash>/`) so results are always traceable to the exact toggles. Metadata `models` and `extern` are filled from whichever components declare them (each class has `models = [...]` and `extern = "..."` attributes), so the run file cannot omit a model that was actually used.

### 2.6 Layout

```
src/hons/
  types.py  config.py  registry.py  runner.py
  io/        source.py (DaySource: db replay)  client.py (strict CDetClient subclass)  topics.py
  stages/    s0_fragment.py  s1_questions.py  s2_score.py  s3_tail.py
             s4_novelty.py   s5_gate.py       s6_output.py s7_propose.py  s8_state.py
  state/     sqlite.py (registry, versions, memory, caches)   cache.py (keyed score cache)
  metrics/   proxies.py  (P1-P5)  qwa.py (later)
  analysis/  date_dev.py (Appendix A2)  day_stats.py  summarise_run.py
configs/     m4_min.toml  empty.toml  toolkit_baseline.toml ...
runs/        (gitignored)
tests/       test_causality.py  test_registry.py  test_proxies.py  test_s3_*.py ...
```

Add `runs/`, `*.jsonl` and `dev_dated.json` to `.gitignore` (`eng-docs.jsonl` and `src/cdet_api.egg-info/` currently show as untracked).

### 2.7 Runner

The runner loops over the days from the `DaySource`, in order, from day 0. For each `DayBatch` it does these steps in this order:

1. Stage 0 fragments the day's documents.
2. Stage 1 activates any examples that arrive today. The new question version takes effect from the next sequence number.
3. Stage 2 scores the fragments against the question versions currently in force.
4. Stage 3 turns scores into candidates.
5. Stage 4 rewrites the candidate weights using novelty.
6. Stage 5 turns candidates into the day's output.
7. Stage 7 adds proposed questions to the output. This happens before Stage 6, so a proposal appears on the same day.
8. Stage 6 writes the output. Live mode calls `client.retrieval` for every topic. Replay mode appends to the run file.
9. Stage 8 updates state, and only after the output has been written.

Replay and live differ only in `DaySource` and `s6.write`. Add a test that runs three days in both modes with the baseline config and compares outputs.

## 3. Build order

Matches the briefing milestones. Each row is shippable and testable before the next.

| Step | Work | Done when |
| --- | --- | --- |
| P0 | `types.py`, `config.py`, `registry.py`, `io/` (source, strict client, topics), runner skeleton with all `off` stages, `s6` writer | empty-run config produces a file that `check-cdet-run.py` accepts, in replay and live mode |
| P1 | `analysis/date_dev.py` (briefing A2), `day_stats.py` (per-day counts, id prefixes), `.gitignore` | active days per question table; confirms 1,093 days |
| P2 | `metrics/proxies.py` P1-P5 with unit tests from briefing B10 | empty run = max on inactive, 0 on active; toolkit baseline false-alarm rate reported |
| P3 | `s2: bm25_day` (wrap the toolkit baseline), `s3: topk/fixed`, `s5: all_candidates` | reproduces `my-run.json` under the new runner (sanity check of the whole harness) |
| P4 | M4 minimum system: `s2: bm25_acc` + `zscore_window`, `s3: quantile`, `s5: best_excess`, `s8` windows | beats the toolkit baseline on false-alarm rate |
| P5 | `s3`: surprise, spot, dspot, hybrid; q sweep; leave-one-topic-out tuning script | ladder table |
| P6 | `s1` seeds and example arrival, with the causality test (briefing B11) | an example never affects days before its arrival |
| P7 | `s4` novelty variants (**focus stage**, see section 6) | ablation table including weight 0 |
| P8 | `s0` full: shingles, boilerplate registry, page families, groups | retention check and audit script |
| P9 | `s7` propose, leave-one-question-out (**focus stage**, see section 7) | proposals per topic, held-out recall |
| P10 | LM and dense variants behind the same registry, cached by (question, doc, model, prompt) | cost per day reported, rows labelled "may be contaminated" |

P0 to P4 are the minimum worth showing a supervisor: they give a working harness, floors, and one real system.

## 4. Cross-cutting rules

- **Cache**: `state/cache.py` keys every expensive score by (kind, question version, fragment hash, model, params). Threshold sweeps then rerun without rescoring.
- **Checkpoints**: `s8` writes a daily checkpoint (SQLite) so a run resumes after a crash. A mid-collection start without a checkpoint is refused by the runner, since peaks, windows and memory would be wrong.
- **Logging**: per day and per question, write raw score, bar, flags, novelty and the reason a question was returned or dropped to `runs/<id>/trace.jsonl`.
- **Seeds**: one `seed` in `[run]`, passed to anything random.
- **Tests**: add `pytest` to a `dev` extra in `pyproject.toml`. Property tests for the metrics (briefing 6.3), unit tests from Appendix B for SPOT, Surprise, multiple testing and novelty weights.

## 5. Decisions (resolved 26 October 2026)

- **Compute:** university GPU fleet available. Dense and language-model variants are realistic, so P10 is no longer last: pull the Stage 4 and Stage 7 model variants forward (sections 6 and 7). Still cache every model score and record every model with its cutoff.
- **Focus:** Stages 4 (novelty) and 7 (new-question proposal) get the depth. Everything else stays at the simplest working option (P0 to P4) until these two have ablation tables. Revisit later.
- **Config format:** TOML.
- **Scorer and test material (checked 26 October 2026):** none is public. The toolkit has no scoring code or judgments (latest commit 35d96b6, 31 July 2026; only the two dev topic files). The guidelines still say the scoring script is unreleased. The 15 October "scores released" date is a delivery to participants, not a public release, so briefing A.5 misread it. No participant papers exist yet; the workshop is 16 to 18 November. Consequences:
  - Proxy metrics (P2) stay the working evaluation. Even a scorer alone would not help on dev, because the dev labels are binary, undated and non-exhaustive, with no 1/5/10 gains and no question-level judgments per day. Importance ordering and silent-day behaviour need test judgments or silver labels.
  - Implement the Quit While Ahead metric family from the paper (read it in full first). The track's aggregation of question and document scores, and the sentinel treatment, cannot be reconstructed; state this as an assumption.
  - Possible routes to more: the supervisor offered to try for early non-public access (ask for the scoring script, then the metric aggregation rules, then test topics, in that order; judgments are hardest and cannot be redistributed before public release); a NIST public data release after the workshop (not promised).

## 6. Reference convention (applies to all code)

Every implementation that comes from a paper carries a comment giving the paper, taken from `References by Pipeline Stage.md`. The comment sits above the class or function it describes, starts with `Ref:`, and has four parts: the full citation (authors, year, title, venue, pages); what the paper contributes to this code; the verification status; and where it is listed in the references document (stage number).

Rules:
- Cite the paper for the mechanism actually implemented, and say what is taken from it and what is adapted (for example "adapted: applied to fragments, not tweets").
- Copy the verification status honestly (verified, partial, unverified, abstract only). Do not upgrade it.
- Where there is no paper, write `Original design [Guess]` or `Standard technique, no source read`, as the references document does. Do not invent a citation (Broder 1997 and multiple-testing corrections are the known cases).
- Where the source has a known flaw, say so in the comment (Aggarwal and Subbian's inverted novelty direction; Kumar's garbled retirement rule; Surprise text against Algorithm 1).
- Registry entries get a class attribute `ref = "..."` as well, so the run metadata and trace can list the references used by a run.

## 7. Stage 4 (novelty): toggles and references

The stage sees only candidates that cleared the Stage 3 bar, so it is cheap; most variants can use a GPU model without cost concerns. Memory is read from previous days only (briefing Stage 4, memory timing).

Config keys in `[s4_novelty]`:

| Key | Options |
| --- | --- |
| `enabled` | on or off |
| `impl` | `exact_hash`, `nn_cosine`, `lsh_backoff`, `answer_pool`, `s7_rule`, `lm_judge`, `nli_judge` |
| `weighting` | `soft`, `hard` |
| `tau` | novelty threshold, for example 0.30 |
| `floor` | lowest weight a repeat can get, for example 0.2; use 0.0 for the zero-weight ablation |
| `memory` | `reported` (only what was returned), `all_candidates` |
| `horizon_days` | memory length in days, for example 30 |
| `group_policy` | for example `one_per_group` |

| `impl` | What it does | Reference to cite | Notes |
| --- | --- | --- | --- |
| `exact_hash` | Novelty 0 if the fragment hash is in the question's memory | None; engineering | Cheapest floor |
| `nn_cosine` | 1 minus the best cosine to any remembered fragment in the window | Petrovic et al. 2010 (NAACL HLT) | Exact search is enough at about 913 documents a day |
| `lsh_backoff` | 13-bit LSH with exact back-off to the 2,000 most recent | Petrovic et al. 2010 | Compare against `nn_cosine`; only matters at scale |
| `answer_pool` | Novel if an unseen entity of the wanted type appears | Li and Croft (year and venue unconfirmed; "2004" is a template stamp) | Report entity-type questions separately (about 7 of 24 dev questions) |
| `s7_rule` | Novel if similarity is below mean minus 3 spreads | Aggarwal and Subbian 2012 (SDM, DOI 10.1137/1.9781611972825.54) | Paper states the direction backwards in one place and its "sigma" is a variance; implement the corrected rule and say so |
| `lm_judge` | Model asked whether the fragment adds anything beyond earlier ones | Original design; evaluator tiers from Ai et al. (NovAScore, arXiv:2409.09249, partial) | Leakage caveat; list the model |
| `nli_judge` | NLI model decides entailment by remembered fragments | Ai et al. (NovAScore): NLI agreed best with humans (page summary, partial) | Cheaper and more controllable than free-form prompts |

Weighting and policy references: soft weight and floor are original design; hard filters hurt precision at depth in Li and Croft. Repeat credit in the track is unknown, so `weighting` ablations cover zero, half and full credit; context from Lin et al. 2016 (RTS: repeats scored zero), Hubert et al. 2017 and Buntain et al. 2023 (CrisisFACTS: redundant 0.5). Evaluation: the two dev measures in briefing Stage 4, plus TDT5 Cmin as a component benchmark only if the data is accessible.

## 8. Stage 7 (propose): toggles and references

Stage 7 is the most contamination-exposed use of a model (question writing and the new, rephrase or update decision). For every model variant use either a pre-2021-08-01 model or label rows "may be contaminated" (briefing A.3). Candidate clean options from the references document: T5 and monoT5 (C4 from April 2019), ChronoGPT-Instruct or the HZ0619 2020-cutoff instruct model (the latter is unverified; check its card first).

Config keys in `[s7_propose]`:

| Key | Options |
| --- | --- |
| `enabled` | on or off |
| `orphan_test` | `narrative_bar`, `below_all_bars` |
| `cluster` | `single_pass` |
| `cluster_cutoff` | `fixed`, `radius` |
| `cutoff` | number; 0.8 was Kumar et al.'s value, which will need re-tuning |
| `min_domains` | 1, 2 or 3 |
| `support` | `count`, `entropy` |
| `question_text` | `template`, `lm_zero_shot`, `diffqg` |
| `change_check` | `embedding`, `qa_equivalence`, `lm_judge` |
| `timing` | `first_supported`, `wait_n` |
| `max_per_topic_per_day` | integer, for example 1 |

| Toggle | Option | Reference to cite | Notes |
| --- | --- | --- | --- |
| `orphan_test` | `narrative_bar` | Original design ; reuses the Stage 3 machinery | Narrative is a pseudo-question with its own bar |
| `cluster` | `single_pass` | Kumar, Liu, Mehta, Subramaniam (arXiv:1405.1392; venue unconfirmed) | Cutoff 0.8 was for a compression distance on tweets; a cosine cutoff must be tuned |
| `cluster_cutoff` | `radius` | Son et al. 2026 (CREAM, arXiv:2601.02708; KDD 2026 only from the supplied PDF) | Lambda 8 main, best value 4 or 16 by dataset |
| `cluster_cutoff` | alternative streaming clusterer | Yoon, Lee, Zhang, Han 2023 (USTORY, SIGIR 2023) | Page summary only; candidate comparison |
| `support` | `entropy` | Kumar et al.: H = minus sum of p log p | Validated on eight earthquakes only |
| `question_text` | `template` | Li and Croft (general-question formulation) | Granularity matters |
| `question_text` | `lm_zero_shot` | Original design; related: Wu et al. 2025 (arXiv:2501.00888, self-questioning for timelines, page summary) | Check answerability against the cluster fragments |
| `question_text`, `change_check` | `diffqg` / `qa_equivalence` | Cole, Jain, Eisenschlos, Zhang, Choi, Dhingra (DiffQG, arXiv:2303.00242; venue unconfirmed) | QA-equivalence F1 80.9 against 44.1 random on Wikipedia; prone to spurious changes; T5-based, so clean on cutoff |
| `change_check` | `embedding` | Original design  | Cheapest |
| `change_check` | `lm_judge` | Original design | Contaminated unless time-locked |

Outcome handling follows the briefing table (new, rephrase, update, noise). Evaluation: never-propose baseline, propose-every-orphan, then leave-one-question-out on dev (project design, untested) and the DiffQG decision step on its own data as a component test. Related background for the framing: RAGTIME 2025 overview (Autonuggetization, Lawrie et al., arXiv:2602.10024, verified).

**Gap to state in the dissertation:** no read paper evaluates deciding new question, rephrase or update on news, so this stage is novel work built from components, not a reimplementation.

## 9. Revised build order for the focus stages

Do P0 to P4 first, then:

1. P5-lite: only `topk` and `quantile` in Stage 3 (full Surprise and SPOT ladder can wait).
2. P6: Stage 1 seeds and arrival-day handling (Stage 4 memory and Stage 7 seeds depend on it).
3. P7: Stage 4, cheapest to richest (`exact_hash`, `nn_cosine`, `answer_pool`, `nli_judge`, `lm_judge`).
4. P8-lite: Stage 0 exact-hash groups and boilerplate registry only (Stage 7 support counts need domains and groups).
5. P9: Stage 7, in the order orphan test, clustering, `template` text, `embedding` check; then model variants.
6. Dense scorer and encoder in Stage 2 (needed for `nn_cosine` embeddings anyway); cache embeddings by fragment hash.
