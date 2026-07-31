# Final Eight-Feature Report

- Feature contract: version 1.0, status frozen
- Keyword version: 1.0
- Keyword status: frozen
- Frozen date: 2026-07-30
- Text source: `sanitised_subject` + newline + `sanitised_body`

## Feature statistics

| Feature | Min | Max | Mean | Median | Zero ratio |
|---|---:|---:|---:|---:|---:|
| url_count | 0 | 652 | 1.35292 | 0 | 0.667973 |
| ip_address_url_count | 0 | 1 | 0.000326477 | 0 | 0.999674 |
| urgency_word_count | 0 | 274 | 0.956579 | 0 | 0.639896 |
| credential_word_count | 0 | 1846 | 3.08978 | 0 | 0.559909 |
| action_word_count | 0 | 1311 | 3.37023 | 2 | 0.331701 |
| money_related_word_count | 0 | 597 | 1.44042 | 0 | 0.761345 |
| uppercase_letter_ratio | 0 | 0.864198 | 0.105977 | 0.0876532 | 0.003265 |
| exclamation_mark_count | 0 | 177 | 0.822396 | 0 | 0.760692 |

## Keyword-feature non-zero email counts by ground-truth label

These descriptive counts are reported only after the researcher-approved freeze and were not used to select or modify any keyword.

| Feature | phishing | legitimate |
|---|---:|---:|
| urgency_word_count | 919 | 184 |
| credential_word_count | 1167 | 181 |
| action_word_count | 1346 | 701 |
| money_related_word_count | 531 | 200 |
