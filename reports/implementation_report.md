# Week 6–7 Final Implementation Report

## Commands and exit status

- `python -m pytest -q`: PASS (exit 0; 35 passed).
- `python run_pipeline.py --config configs/pipeline.yaml --mode final`: PASS (exit 0).
- `python run_pipeline.py --config configs/pipeline.yaml --stage validate-only`: PASS (exit 0).

## Frozen invariants

- Raw SHA-256 before: `afef69828165619ced6661ca14a95444a1db82c7b25ae9c2619bbbb108844a44`
- Raw SHA-256 after: `afef69828165619ced6661ca14a95444a1db82c7b25ae9c2619bbbb108844a44`
- Eligible/excluded: 3063 / 2
- Split counts: {'train': 2144, 'validation': 460, 'test': 459}
- Cross-split template groups: 0
- Raw exact duplicate groups/records: 0 / 0
- Sanitised exact duplicate groups/records: 15 / 31
- Mixed-label sanitised duplicate groups: 0
- Duplicate policy: `keep_all_grouped`.
- Keyword configuration: version 1.0, status frozen, date 2026-07-30.
- Feature contract: exactly eight columns in the frozen order.

## Final matrices

| Matrix | Rows | Columns | SHA-256 |
|---|---:|---:|---|
| all | 3063 | 15 | `e2d8517230151c2fd40afa39b58aecc8e597c89d6ad143673de166db18d44866` |
| train | 2144 | 15 | `a9a55f1cf690c881673eaa8dab82c4b6c382ea9308ca3f31be2de9b70ffb77ba` |
| validation | 460 | 15 | `64688aec1c3918f15fd622c972e989094fe63774d537b38f2f4c33df3c684768` |
| test | 459 | 15 | `ea1b8737c77a2f03574457f8d4c3f8e7fe947cf9e01867ee6ccf0173d5233bad` |

## Definition of Done

| Item | Status | Evidence |
|---|---|---|
| Dataset and raw protection | PASS | Read-only raw file; before/after SHA-256 identical. |
| Labels | PASS | Only configured 0=legitimate and 1=phishing mappings accepted. |
| Audit and sanitisation | PASS | Dataset audit, deterministic/idempotent safe text, exclusions recorded. |
| Duplicate/template grouping | PASS | Content-only grouping; keep-all policy; review report generated. |
| Fixed split | PASS | 2144/460/459, disjoint union, zero cross-group violations. |
| Frozen eight features | PASS | Names, order, source, ranges and model interface validated. |
| Frozen keywords | PASS | Version/status/date/lists/rules match the approved specification. |
| Final outputs | PASS | Non-provisional all/train/validation/test matrices generated. |
| Automated tests | PASS | 35 tests passed with exit 0. |
| Validate-only | PASS | Required post-final bundle validation completed with exit 0. |

## Remaining risks

- Any mixed-label sanitised duplicate group listed in `researcher_review_required.md` remains a researcher decision; records stay grouped and unchanged.
- Near-template grouping uses the documented engineering defaults (seed 42, char 3–5gram TF-IDF, cosine >=0.95) and is not a model-performance choice.
