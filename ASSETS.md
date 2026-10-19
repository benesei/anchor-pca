# Data and licenses

This repository's original code is licensed under [MIT](LICENSE). Raw datasets
are not distributed.

## Gas-sensor drift data

The scripts download the [Gas Sensor Array Drift Dataset at Different
Concentrations](https://doi.org/10.24432/C5MK6M) from UCI and verify the archive
checksum. UCI lists the dataset under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the linked dataset
page for the provider's terms and requested citations, including Vergara et al.
(2012) and Rodríguez-Luján et al. (2014).

## Fama–French industry portfolios

The scripts download [49 Industry Portfolios, daily value-weighted returns](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library.html)
from the Kenneth R. French Data Library, Tuck School of Business at Dartmouth
College. We use 1970–2024; the paper used the May 2026 CRSP revision, accessed
July 26, 2026. The library reconstructs the history when updating its data, so
historical returns may change. A supplied original archive can be used instead
of a fresh download. The scripts record the revision and input hashes.

Data remain subject to the source's terms and are not covered by this
repository's MIT license.

## minPCA baseline

The wcPCA baselines use [minPCA](https://github.com/anyafries/minPCA), pinned to
commit `9b90ebd0b56fc8eb88a7217e38c49b35a6927f82` in
`requirements-repro.txt`. That package is licensed under AGPL-3.0 and is installed
as an external dependency; its source is not included here.
