# Low-temperature detector comparison

Paired differences below are posterior minus hard, in percentage points. 
Intervals resample key groups and prompts; matched thresholds stay frozen. 
The primary comparison measures actual methods at different operating points. 
Matched-FPR results and held-out ROC/AUC assess scoring quality separately.

- T=1.0, primary: +5.00 pp (95% CI +0.62 to +11.27).
- T=1.0, matched-FPR: +11.25 pp (95% CI +3.75 to +20.62).
- T=1.2, primary: +97.50 pp (95% CI +92.50 to +100.00).
- T=1.2, matched-FPR: +8.75 pp (95% CI +2.50 to +16.88).
- T=1.4, primary: +66.25 pp (95% CI +54.38 to +78.12).
- T=1.4, matched-FPR: +0.00 pp (95% CI +0.00 to +0.00).

Consult both realized FPR columns and AUC in results_summary.csv before interpreting a TPR gain. 
A gain confined to published thresholds does not establish superior posterior information; 
a matched-FPR gain supported by ROC is stronger evidence. Oracle results cannot support the headline.
