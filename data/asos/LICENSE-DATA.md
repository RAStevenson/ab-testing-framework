# Data license: ASOS Digital Experiments Dataset

The file `asos_digital_experiments_dataset.csv` in this directory is the
ASOS Digital Experiments Dataset, redistributed here UNMODIFIED under
its own license, which is separate from this repository's MIT license
for code.

- **Source:** Open Science Framework, DOI 10.17605/OSF.IO/64JSB
  (https://osf.io/64jsb/), direct download https://osf.io/download/hq8sc/
- **Creators:** the AI and Data Science Platform at ASOS.com, with the
  EPSRC CDT in Modern Statistics and Statistical Machine Learning
  (Imperial College London and University of Oxford). Copyright 2021.
- **License:** Creative Commons Attribution 4.0 International (CC BY
  4.0), https://creativecommons.org/licenses/by/4.0/, as stated in the
  dataset's own datasheet and on its OSF record. Redistribution with
  attribution is permitted by that license; no changes have been made to
  the file.
- **Citation:** C. H. Bryan Liu, Angelo Cardoso, Paul Couturier,
  Emma J. McCoy. "Datasets for Online Controlled Experiments." NeurIPS
  2021 Datasets and Benchmarks Track. arXiv:2111.10198.
- **Integrity:** MD5 `acdb1bebeede91cad1545f52d9a8a246` (2,737,991
  bytes, as downloaded 2026-08-12). The analysis config pins this digest
  and the analyze command refuses to run against a file that does not match it.
  The dataset's maintainers state that minor corrections would be
  published in place under the same filename, so a future mismatch means
  a new upstream version should be reviewed, re-pinned, and re-analyzed.
- **Metric descriptions:** the dataset's OSF wiki (the schema page at
  https://osf.io/64jsb/wiki/home/, retrieved 2026-08-31) states that
  metric 1 accepts binary responses, metrics 2 and 3 accept count-based
  responses, metric 4 accepts non-negative real number responses, and
  that "The responses for metrics 2, 3, and 4 demonstrate various
  degrees of right skewness." The metrics are otherwise anonymized.
- **Primacy flag behind the single-experiment selection:** the
  datasheet ("Datasheet - ASOS Digital Experiments Dataset.pdf" on the
  OSF record, retrieved 2026-08-31) lists possible data pitfalls and
  states: "The treatment mean entries for Experiment '54a85a' are
  increasing faster than the control mean entries of the same
  experiment, indicating a possible primacy effect in play." It is the
  only experiment the datasheet flags for a possible primacy effect
  (its other example, 162a38, is flagged for possible telemetry loss).
  The selection rule's second condition, variances present at every
  checkpoint, was checked against the committed CSV on 2026-08-31:
  all 1,152 rows for 54a85a carry both variance entries.
