# US AI Token TAM, Interactive Dashboard

A static, single-page dashboard for exploring the AI-token TAM model. No backend
and no build step, just two files.

## Files
- `index.html`: the dashboard (React via CDN, all CSS inline).
- `dashboard_data.json`: 27 precomputed scenarios plus the convergence frontier (~60 KB).

## Deploy to Netlify
Both files must sit in the **same directory** (the JSON is fetched relatively).

**Drag-and-drop:** zip the folder containing `index.html` and
`dashboard_data.json`, drop it on the Netlify dashboard. Done.

**Git:** commit both files at the repo root (or any folder), point Netlify at it
with no build command and the publish directory set to that folder.

That is it. There is no compile step. The page loads React and Babel from
cdnjs at runtime.

## What the controls do
Three dropdowns vary the enterprise inputs that Phase 3 sensitivity analysis
flagged as driving about 95% of TAM uncertainty:
- **Enterprise AI adoption:** how large a share of IT budget goes to AI.
- **Spend concentration:** how unevenly that spend is distributed across firms.
- **Enterprise scale:** the typical size of enterprise IT budgets.

Consumer inputs are held at central values (they barely move the total).

## Regenerating the data
If the calibration changes, re-run (from the repo root so the model modules import):
```
python3 assets/precompute_dashboard.py
```
which rewrites `dashboard_data.json`. The dashboard picks up the new numbers
with no code change. Adjust the lever levels in the `LEVELS` dict at the top of
that script.

## Note on the numbers
The consumer income distribution is calibrated to CPS income data and the macro
sanity window is anchored to the 2026/equilibrium ceiling ($40B to $60B). The
enterprise IT-budget distribution and the spend-concentration levels are still
estimates and will firm up once fitted to Census/BLS firm-size microdata. The
durable findings are the shapes and relationships (skew, concentration, what
drives the tail), not the last dollar of any single scenario.

## Production hardening (optional)
The dashboard loads React and Babel from a CDN and transpiles JSX in the browser,
which is fine for a personal or Substack embed. For a higher-traffic deploy you
may want to pre-transpile the JSX and vendor React locally so there is no runtime
Babel step. Happy to provide that build if needed.
</content>
