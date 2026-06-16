# US AI Token TAM — Interactive Dashboard

A static, single-page dashboard for exploring the AI-token TAM model. No backend,
no build step — just two files.

## Files
- `index.html` — the dashboard (React via CDN, all CSS inline).
- `dashboard_data.json` — 27 precomputed scenarios + convergence frontier (~60 KB).

## Deploy to Netlify
Both files must sit in the **same directory** (the JSON is fetched relatively).

**Drag-and-drop:** zip the folder containing `index.html` and
`dashboard_data.json`, drop it on the Netlify dashboard. Done.

**Git:** commit both files at the repo root (or any folder), point Netlify at it
with no build command and the publish directory set to that folder.

That's it — there's no compile step. The page loads React and Babel from
cdnjs at runtime.

## What the controls do
Three dropdowns vary the enterprise inputs that Phase 3 sensitivity analysis
flagged as driving ~95% of TAM uncertainty:
- **Enterprise AI adoption** — how large a share of IT budget goes to AI.
- **Spend concentration** — how unevenly that spend is distributed across firms.
- **Enterprise scale** — the typical size of enterprise IT budgets.

Consumer inputs are held at central values (they barely move the total).

## Regenerating the data
If the calibration changes, re-run:
```
python3 precompute_dashboard.py
```
which rewrites `dashboard_data.json`. The dashboard picks up the new numbers
with no code change. Adjust the lever levels in the `LEVELS` dict at the top of
that script.

## Note on the numbers
Figures are illustrative output from placeholder calibrations. The shapes and
relationships (skew, concentration, what drives the tail) are the durable
findings; the exact dollar values firm up once the input distributions are
fitted to ACS/CPS income data and Census/BLS firm-size data.

## Production hardening (optional)
The dashboard loads React + Babel from a CDN and transpiles JSX in the browser,
which is fine for a personal/Substack embed. For a higher-traffic deploy you may
want to pre-transpile the JSX and vendor React locally so there's no runtime
Babel step — happy to provide that build if needed.
