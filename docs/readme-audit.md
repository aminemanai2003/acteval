# README audit for 1.0

The pre-1.0 README accurately described the point, probabilistic, and decision
APIs, but it had become a chronological implementation narrative rather than a
library landing page. The 1.0 review found these gaps:

| Area | Earlier state | 1.0 resolution |
|---|---|---|
| Project signals | No badges or supported-version summary | Added PyPI, Python, CI, license, and typing badges |
| Value proposition | Correct but spread across version paragraphs | Led with model-agnostic purpose and actuarial trade-offs |
| Feature discovery | Long sequential prose | Added a capability-to-diagnostics map |
| Quick start | Point evaluation only | Kept a minimal runnable example and explicit metric selection |
| Inference | Not present | Added interval and paired-comparison workflows with interpretation |
| Monitoring | Not present | Added segment, temporal, and drift examples |
| Reporting | Plot calls only | Added HTML, CSV, JSON, and plot export examples |
| Contracts | Exposure contract was present | Consolidated scale, weighting, and observed-tail limitations |
| Navigation | Three documentation links | Added focused API, metric, inference, monitoring, reporting, decision, and stability links |
| Maintenance | Development commands only | Added contribution, security, versioning, and license paths |
| Terminology | “Supported MVP metrics” remained after v0.3 | Replaced with a version-neutral capability overview |

## Verification criteria

- Installation uses the published `acteval-insurance` distribution and
  `import acteval` module.
- Every shown public function is exported from `acteval.__all__`.
- Runnable examples use defined variables or clearly named placeholders in
  model-comparison contexts.
- Scientific limitations appear next to the feature they constrain.
- Relative documentation links use absolute GitHub targets so they work on
  both GitHub and PyPI.
- Development commands match CI.
- The README makes no universal model-quality or PSI-threshold claim.
