# Decision-aware reference

ActEval's decision layer implements the architecture

```text
predictive distribution -> explicit decision -> financial loss
```

It does not assert that one loss function is appropriate for every insurer,
jurisdiction, product, or business objective.

## Generic regret

Given model decision `d_m`, benchmark `d_b`, realized outcome `y`, financial
loss `L`, and effective weights `w`:

```text
model_loss     = weighted mean L(d_m, y)
benchmark_loss = weighted mean L(d_b, y)
regret         = model_loss - benchmark_loss
```

Negative regret means the model decision outperformed the supplied benchmark.
Relative regret is only returned when benchmark loss is positive.

## Pricing

`premium_from_distribution()` applies

```text
mean loss * (1 + profit loading) / (1 - expense ratio)
```

`pricing_regret()` uses asymmetric absolute consequence:

```text
c_under * max(y - premium, 0) + c_over * max(premium - y, 0)
```

These costs must represent the user's economic view. This simplified loss does
not model demand elasticity, regulation, expenses that vary by policy, or
multi-period customer behavior.

## Loss ratio

```text
sum w_i loss_i / sum w_i premium_i
```

The signed impact is realized ratio minus target. It is a portfolio consequence,
not a proper statistical score.

## Reserve and capital shortfall

Per observation:

```text
max(realized loss - held amount, 0)
```

ActEval reports weighted aggregate and mean shortfall, weighted frequency, and
conditional mean when shortfall occurs. Reserve and capital functions share the
formula but retain different decision labels because their governance and time
horizons differ.

## Reinsurance

`ReinsuranceOption` represents a quoted excess-of-loss contract with retention
`r` and premium `pi`. Ceded loss is `max(loss-r, 0)` and retained loss is
`min(loss, r)`.

Projected selection minimizes

```text
pi + E[retained loss] + capital_cost_rate * rho(retained loss)
```

where `rho` is VaR or expected shortfall at an explicit quantile. This mirrors
actuarial retention work that combines reinsurance premiums, retained losses,
and tail risk measures, but it is only one possible business rule.

Realized option regret compares `pi + min(loss, r)` to the supplied benchmark
quote. It excludes taxes, reinstatements, limits, counterparty default,
commissions, and contract wording unless users incorporate them in a custom
loss function.

## References

- Cai, J. and Tan, K. S. (2007), *Optimal Retention for a Stop-Loss Reinsurance
  under the VaR and CTE Risk Measures*, ASTIN Bulletin 37(1), 93–112.
- Major, J. A. and Mildenhall, S. J., *Introduction to Capital Modeling and
  Portfolio Management*, Casualty Actuarial Society Monograph No. 15.
- Blanchet, J., Lam, H., Tang, Q. and Yuan, Z. (2016), *Applied Robust
  Performance Analysis for Actuarial Applications*.
