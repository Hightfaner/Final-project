# Dataset Audit

## File contract

- Rows: 3065
- Columns: sender, receiver, date, subject, body, label, urls
- Encoding: utf-8-sig
- MD5: `45db8330ea4aabbf72f5199949ae03e5`
- SHA-256: `afef69828165619ced6661ca14a95444a1db82c7b25ae9c2619bbbb108844a44`

## Labels and text quality

- Label counts: {'legitimate': 1500, 'phishing': 1565}
- Blank subjects: 50
- Blank bodies before sanitisation: 2
- Body length range: 1–4599644
- HTML-like records: 209

## Safety indicators

- URL-like records: 1022
- Active-tag records: 2
- Email-like records: 1538

Only counts and email IDs are reported; raw email bodies are not reproduced.
