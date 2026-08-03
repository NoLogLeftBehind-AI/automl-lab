# Dataset provenance

Nothing in this repository redistributes raw data. The demo notebooks and
benchmarks fetch each dataset at run time from a stable public mirror (the
benchmarks cache a local parquet copy under `results/`, which is gitignored);
what is committed is derived material only — metrics, figures, and reports.

| Dataset | Used by | Fetched from | Original source | License |
|---|---|---|---|---|
| **Adult census income** (48,842 rows) | classification demo, AutoML benchmark | [`jbrownlee/Datasets`](https://github.com/jbrownlee/Datasets) (`adult-all.csv`) | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) (Becker & Kohavi, 1996) | CC BY 4.0 |
| **Diamonds** (53,940 rows) | regression demo, AutoML benchmark | [`mwaskom/seaborn-data`](https://github.com/mwaskom/seaborn-data) (`diamonds.csv`) | the [ggplot2](https://ggplot2.tidyverse.org/reference/diamonds.html) R package's built-in dataset | distributed with ggplot2 (MIT) |
| **ERCOT regional electricity load** (daily, 8 regions, 2016–2021) | forecasting demo, forecasting benchmark | [`ourownstory/neuralprophet-data`](https://github.com/ourownstory/neuralprophet-data) (`load_ercot_regions.csv`, MIT-licensed repo) | ERCOT's public grid operations data | public grid data, via the MIT-licensed mirror |

Why GitHub mirrors rather than the original portals: no authentication or
API tokens, plain CSV over HTTPS, and effectively-stable files (the
benchmarks additionally cache their first download as a local parquet). The
URLs reference branch heads, not commit SHAs, so they are stable in practice
but not cryptographically pinned. That trade-off is accepted and localized:
each loader is a single `read_csv(URL)` line that is trivial to repoint — or
to pin to a commit SHA if upstream ever changes.
