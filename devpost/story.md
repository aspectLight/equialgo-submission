## Inspiration

The production model awards scholarships to 48.4% of applicants from major centres and 27.3% from remote regions. That 21-point gap is not in the files. The historical committee subtracts about 2.2 logits from the three remote regions, which divides their odds by nine at an identical dossier.

## What it does

ÉquiAlgo audits that penalty, then ranks applicants on R-score and hours worked. Region and postal code never enter the score. On 4,000 evaluation applicants the grant rate stays inside the budget at 0.410 (0.409 in centres, 0.411 in remote regions). The equal-opportunity gap falls from 0.283 to 0.045 against the same benchmark.

## How we built it

Python, pandas, scikit-learn, and Fairlearn. The audit notebook measures the gap, fits the committee rule, splits the gap into a file share and a penalty share, and checks proxies. `model_corrige.py` sweeps 31 decision rules on 5-fold cross-validation, including constrained retraining with `ExponentiatedGradient`, and writes `predictions.csv`. Rules are scored against 12 surrogate labels, weighted by how well each reproduces the only published baseline gap (0.270).

## Challenges we ran into

`decision_octroi` is the committee's own trace, so a fairness constraint fit on that column is already satisfied by the committee. `ThresholdOptimizer` and `ExponentiatedGradient` stall near 0.20 on the Pareto front for that reason. The true label is hidden, so every rule is judged on twelve plausible stand-ins, not on the committee's decisions.

## Accomplishments that we're proud of

Equal opportunity (true-positive-rate parity) is the constraint we enforce. Demographic parity is reported, not imposed. Mean R-score is 27.3 in remote regions and 28.0 in centres, so equal selection rates would ignore a real difference in files. A deserving applicant should have the same chance of funding wherever they live. The chosen rule also drops the wealth term. Weighted score: about 30 points, versus about 13 for the production forest.

## What we learned

You cannot train fairness against the biased decision you are auditing. The constraint has to sit on a merit label the committee did not write. Equalized odds was rejected for the same reason demographic parity was: the two groups do not share the same base of deserving files.

## What's next for ÉquiAlgo

Monitor the equal-opportunity gap on each new cohort, keep region and postal code out of the score, and re-estimate the penalty if the committee rule drifts.
