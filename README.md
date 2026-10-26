# Statistical Estimation of Canteen Arrival Overload Probability Using Poisson Distribution

**System name:** CanteenFlow, an interactive statistical analysis dashboard
**Course:** PAMAT501, Probability and Statistics
**Base paper:** Tang, T.-Q., Zhang, B.-T., Zhang, J., & Wang, T. (2019). *Statistical analysis and modeling of pedestrian flow in university canteen during peak period.* Physica A, 521, 29–40. https://doi.org/10.1016/j.physa.2019.01.060

| Presented by | Register No. |
| :--- | :--- |
| Mayur Krishna | 26MCA0191 |
| Faraz Akhtar Khan | 26MCA0197 |
| Malhar Bhosale | 26MCA0199 |

**Faculty guide:** Dr. Manimaran
**Technology:** Python 3, Pandas, NumPy, SciPy, Plotly, Streamlit, Pytest

---

## Abstract

University canteens see sharp changes in arrival intensity during the lunch peak, and a sudden rise produces crowded
service areas and long queues. This project asks a narrow, practical question: *what is the probability that arrivals
exceed a selected level during a particular time interval?* Arrival counts are recorded in fixed short slots across
several days, a separate Poisson rate λₜ is estimated for each slot with an exact confidence interval, the Poisson
assumption is tested slot by slot, and the probability of overload P(Xₜ > C) is converted into a low / medium / high
risk label. The analysis also identifies the least-crowded slot and tests whether it is genuinely quieter than the
alternatives. On the 90-day prototype dataset (one semester with two exam weeks), the riskiest slot is 13:15–13:30
(P = 73.9% at capacity C = 30) and the least crowded is 14:15–14:30 (λ̂ ≈ 16.3, significantly quieter than the
runner-up, p ≈ 5 × 10⁻⁶).

The second review extends the work from description to **prediction of the peak time**. A multiple linear regression
gives slot rates for a chosen weekday and day type (regular or exam); simulation turns them into the probability that
each slot is the day's peak; Bayes theorem with a Gamma prior updates the forecast from the morning's counts; a
Poisson–Gamma model corrects the overload probability for day-to-day variation; and a rolling backtest checks every
forecast on days the model has not seen, using hypothesis tests from the course. On a regular Monday the predicted
peak is 13:15–13:30 with probability 42%, and the peak falls in 13:00–13:30 with probability 75%. **Every method maps
to the PAMAT501 syllabus (Modules 1–5).**

The project focuses on **arrival overload risk, not exact waiting time**.

---

## 1. Problem Statement and Motivation

University canteens experience changing arrival intensity during lunch and other peak periods. When the number of
students arriving in a short interval exceeds what the counters can serve, queues grow, service areas become crowded
and food runs short. Management decisions are often based on intuition rather than measured arrival behaviour.

**Practical question.** What is the probability that arrivals will exceed a selected level during a particular time
interval, and which intervals are least likely to be crowded?

**Project direction.**

```
Observed arrivals → Estimate λₜ → Test Poisson fit → P(Xₜ > C) → Risk label → Least-crowded time
                 → Predict the peak for a chosen day → Update during the day → Validate on unseen days
```

---

## 2. Objectives

1. **Collect and organise arrival data.** Record arrivals at fixed 5-, 10- or 15-minute intervals across m days.
2. **Estimate time-specific rates.** Calculate λ̂ₜ for every slot and examine Poisson suitability.
3. **Calculate overload probability.** Estimate P(Xₜ ≥ K) = P(Xₜ > C) for a selected threshold, with its uncertainty.
4. **Identify useful time windows.** Classify intervals as relatively low, medium or high risk, and identify the
   least-crowded slot.
5. **Predict the peak time** *(second review)*. Forecast each slot's rate and P(Xₜ > C) for a chosen weekday and day
   type, the probability that each slot is the day's peak, and update the forecast from the day's first counts.
6. **Validate the forecasts** *(second review)*. Forecast each day from earlier days only and test the forecasts with
   syllabus hypothesis tests.

---

## 3. Base Paper

Tang et al. (2019a) study pedestrian flow in the second canteen of Beihang University during the lunch peak.

| Data collection | Detailed model | Main outcome |
| :--- | :--- | :--- |
| Camera-based observation; peak period of about 10:30–13:30; short time segments; arrival and departure counts | Arrival and departure rates, service windows and service time, window selection, movement, congestion, meal duration and seating | A complete pedestrian-flow model, calibrated with data and validated against observations |

The base paper is broad and system-level. **This project focuses on the arrival-count component.**

---

## 4. Literature Survey

Eighteen studies were reviewed. Studies 1–10 come from the first review (canteen modelling, canteen queueing,
arrival-process statistics, time-varying demand, model checking and campus crowd sensing); each was checked against
OpenAlex and its DOI and summarised from its published abstract. Studies 11–18 were added for the second review on
**forecasting arrivals and checking forecasts**; each was read in full, and the summaries cite sections and tables.

| # | Study | Area | Focus and method | Relevance to this project |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Tang et al. (2019a), *Physica A* (base paper) | Canteen pedestrian flow | Camera data from the Beihang canteen at lunch peak; statistical analysis and a pedestrian-flow model validated against observations | Shows arrivals can be observed in short segments and analysed statistically; we isolate the arrival-count component |
| 2 | Tang, Zhang & Xie (2019b), *Simulation Modelling Practice and Theory* | Canteen pedestrian flow | Extended cellular-automaton model of each pedestrian's motion, calibrated on the same canteen | Shows how detailed full canteen models are; arrival intensity is an input, not a tested quantity |
| 3 | Chen & Wang (2011), *J. Huangshi Inst. of Technology* | Canteen queueing | M/M/n/m queueing model to set serving windows, seats and opening times | Typical queueing treatment; Poisson arrivals are assumed rather than tested |
| 4 | Yuan, Wang & Chen (2023), *Transactions on Comparative Education* | Canteen queueing | Survey data, queueing theory and MATLAB simulation of staggered dining times | Confirms timing is the main crowding lever; does not test when crowds are smallest |
| 5 | Brown et al. (2005), *JASA* | Arrival-process statistics | Call-by-call data for a year; a test that a point process is Poisson and models for forecasting Poisson arrival rates | Methodological model: treat arrivals as their own component and test the Poisson assumption |
| 6 | Green, Kolesar & Whitt (2007), *Production and Operations Management* | Time-varying demand | Review of staffing methods when demand varies during the day | Justifies one λ per short slot instead of one rate for the whole peak |
| 7 | Kim & Whitt (2014), *M&SOM* | Arrival-process statistics | KS tests of nonhomogeneous Poisson arrivals with piecewise-constant rates | Poisson holds only with careful analysis; long subintervals and pooled days cause overdispersion |
| 8 | Leemis (1991), *Management Science* | Time-varying demand | Nonparametric estimate of a time-varying Poisson intensity from repeated realisations | Our per-slot λ̂ from m days is its piecewise-constant counterpart; basis for Monte Carlo generation |
| 9 | Cameron & Trivedi (1990), *Journal of Econometrics* | Model checking | Regression-based tests of Poisson equidispersion | Supports checking Var ≈ Mean before using Poisson probabilities |
| 10 | Variskhanov & Arreeras (2026), *Urban Science* | Campus crowd sensing | Wi-Fi telemetry from five canteens; k-means++ demand shapes and an "Early Peak" badge at 11:28 | Shows real demand for simple lunchtime crowd signals; needs Wi-Fi data and gives a day-level badge |
| 11 | Jiao, Wang, Song & Du (2022), *ITM Web of Conferences* | Canteen queueing | M/M/C canteen queue simulated 2000 times with one rate (19.3 students/min); seven windows found optimal | Poisson arrivals assumed, one rate for the whole period; the authors list period-specific rates as the next step |
| 12 | Shah, Myller & Islas Sedano (2025), *Applied Computing and Intelligence* | Canteen demand forecasting | RFID data from a university self-service restaurant, 2019–2024; LR, random forest, XGBoost, LSTM; XGBoost best | Forecasts daily totals only, although its Fig. 3 shows 48% of customers in one hour; no slot forecast or overload probability |
| 13 | Weinberg, Brown & Stroud (2007), *JASA* | Arrival forecasting | Multiplicative Poisson model (daily level × within-day shape per weekday), Bayesian, within-day learning at 10 am and 12 pm; PIT calibration | Template for slot-rate forecasts, same-day updating and calibration checks; its normal approximation needs ~300 arrivals per slot |
| 14 | Shen & Huang (2008), *Annals of Applied Statistics* | Arrival forecasting | Poisson factor model of quarter-hour rate profiles; AR(1) day-to-day forecasts; intraday updating cut staffing error by 14.5% and interval width by 41.5% | Shows early counts carry information about the rest of the day |
| 15 | Aldor-Noiman, Feigin & Mandelbaum (2009), *Annals of Applied Statistics* | Arrival forecasting | Poisson log-linear model with weekday and billing-cycle indicators, then a mixed model; compared with the "industry model" (average of past same-weekday slots); slot-length comparison | Motivates weekday and exam-day terms; its industry model is the per-slot average used at the first review |
| 16 | Matteson, McLean, Woodard & Henderson (2011), *Annals of Applied Statistics* | Arrival forecasting | Hourly ambulance calls (~24/hour) modelled as Poisson counts directly; beats "average of four past periods" on held-out data | Supports Poisson probabilities for small counts and testing against an averaging baseline |
| 17 | Ibrahim, Ye, L'Ecuyer & Shen (2016), *International Journal of Forecasting* | Arrival forecasting | Survey: overdispersion, within- and between-day correlation, Gamma busyness factor with Var(X) = λ(1 + (1 + λ)Var W) | Basis for the Poisson–Gamma model and the same-day update |
| 18 | McNulty (1988), NWS Technical Attachment 88-26 | Forecast verification | Brier score and a skill score against a no-skill reference forecast | Scoring P(X > C) as a yes/no forecast against the per-slot average |

**Positioning.** Previous work models the broader pedestrian-flow or queueing system, or studies Poisson arrivals in
other service settings. It does not directly provide a simple, statistically checked arrival-risk indicator for a
canteen. The forecasting literature (studies 13–17) predicts arrival rates for staffing in call centres and ambulance
services; no reviewed study predicts *when* a canteen's lunch peak will occur, with what probability, or tests such a
forecast on unseen days.

---

## 5. Research Gaps

Eleven gaps were identified. G1 was stated at the first review; G2 to G6 come from the extended survey; G7 to G11
come from the second-review reading on forecasting. Each is a **focus and application gap**, not a claim that Poisson
modelling of arrivals is new.

### G1. Focus and application gap
**What it is.** The reviewed canteen studies model the complete meal process (movement, windows, seating,
congestion) and output simulated flows or queue sizes. They do not turn time-specific arrival rates into a
threshold-exceedance probability with a risk label, so the question "how likely is overload at 13:15?" has no direct
answer.
**Evidence.** Tang et al. (2019a, 2019b); Chen & Wang (2011); Yuan et al. (2023).
**Addressed by.** λ̂ₜ per slot → P(Xₜ > C) → Low / Medium / High label (dashboard: *Overload risk*).

### G2. Validation gap
**What it is.** Queueing models of canteens start from Markovian (Poisson) arrivals, so the assumption is an input,
not a finding. Kim & Whitt (2014) show the assumption must be tested and that careless tests mislead. Testing many
slots also creates a multiple-testing problem, since some rejections occur by chance.
**Evidence.** Chen & Wang (2011); Yuan et al. (2023); Kim & Whitt (2014); Brown et al. (2005); Cameron & Trivedi (1990).
**Addressed by.** Per-slot chi-square goodness of fit with Cochran pooling, an index-of-dispersion test, and a
binomial check of the number of rejections (dashboard: *Model check*).

### G3. Time-resolution gap
**What it is.** Lunch demand changes within minutes. A single rate for an hour or the whole peak hides the rush and
makes pooled counts overdispersed. Green et al. (2007) show peak-hour and stationary approximations need refinement
under time-varying demand; Kim & Whitt (2014) trace failed Poisson tests to subintervals where the rate varies too
much.
**Evidence.** Green et al. (2007); Kim & Whitt (2014); Leemis (1991).
**Addressed by.** Piecewise-constant rates on 5-, 10- or 15-minute slots, plus an hourly-pooling view that shows the
resulting overdispersion (dashboard: *Arrival pattern*, *Model check*).

### G4. Uncertainty gap
**What it is.** λ̂ is estimated from a limited number of days, so it carries sampling error. A slot near a threshold
can move between Medium and High within the plausible range of λ. Carrying the uncertainty in the estimated arrival
rate through to the reported risk is not part of the reviewed canteen studies' output.
**Evidence.** Tang et al. (2019a); Brown et al. (2005).
**Addressed by.** Exact Garwood confidence intervals, P(X > C) evaluated at both ends of the interval, and a Monte
Carlo check (dashboard: *Overload risk*, *Model check*).

### G5. Decision-support gap
**What it is.** Variskhanov & Arreeras (2026) observe that many smart-campus tools fail to support routine, day-to-day
decisions; their crowd signal needs Wi-Fi telemetry and gives a day-level badge. Queueing studies give window counts
to managers but tell students nothing about when to go. Guidance that needs only arrival counts (when crowding is
least likely, and whether that slot is reliably quieter) is missing.
**Evidence.** Variskhanov & Arreeras (2026); Yuan et al. (2023); Chen & Wang (2011).
**Addressed by.** Least-crowded slot with an exact conditional test against the runner-up, a visit planner, and
capacity recommendations (dashboard: *When to go*, *Arrival pattern*, *Overload risk*).

### G6. Data and context gap
**What it is.** The canteen evidence reviewed comes from specific campuses, mainly Beihang University, and arrival
patterns depend on local timetables. We did not find openly available interval-level arrival counts for an Indian
college canteen, so results from elsewhere cannot be reused directly.
**Evidence.** Tang et al. (2019a, 2019b); Variskhanov & Arreeras (2026).
**Addressed by.** A documented collection protocol, a validated CSV import for field counts, and a clearly labelled
synthetic prototype dataset (dashboard: *Arrival pattern*).

### G7. Prediction gap *(second review)*
**What it is.** The first-review method averages each slot over past days, which Aldor-Noiman et al. (2009) call the
"industry model". Tang et al. (2019a) note that days differ with the timetable, so averaging all days is unsuitable,
and their arrival rate is an input, never a prediction. Shah et al. (2025) forecast campus dining, but only daily totals.
**Evidence.** Tang et al. (2019a); Shah et al. (2025); Aldor-Noiman et al. (2009); Weinberg et al. (2007).
**Addressed by.** Multiple linear regression of log counts on slot, weekday and exam-day dummies, coefficient t-tests,
and ANOVA / Kruskal-Wallis / Welch t / Mann-Whitney tests of the day effects (dashboard: *Peak forecast*).

### G8. Peak-time uncertainty gap *(second review)*
**What it is.** The busiest slot is reported, but not how likely it is to be the peak on a given day. Forecasting
papers produce full predictive distributions (Weinberg et al., 2007; Shen & Huang, 2008) but do not turn them into a
probability that each slot is the peak, or that at least one slot overflows.
**Evidence.** Weinberg et al. (2007); Shen & Huang (2008); Tang et al. (2019a).
**Addressed by.** Monte Carlo simulation of whole lunch periods: P(slot t is the peak), the most likely peak window,
P(at least one overload) (dashboard: *Peak forecast*).

### G9. Same-day updating gap *(second review)*
**What it is.** Forecasts are fixed in the morning, although early counts reveal how busy the day is (Weinberg et al.,
2007; Shen & Huang, 2008). Their methods use large-count approximations, which Matteson et al. (2011) warn fail for
small counts like a canteen's.
**Evidence.** Weinberg et al. (2007); Shen & Huang (2008); Matteson et al. (2011); Ibrahim et al. (2016).
**Addressed by.** Bayes theorem with a Gamma prior on the day's busyness factor and Poisson counts; the posterior is
Gamma and the updated P(X > C) is exact (dashboard: *Peak forecast*, *Update during the day*).

### G10. Overdispersion gap *(second review)*
**What it is.** The first review tested for overdispersion but still used the plain Poisson tail for P(X > C). Arrival
counts vary more than Poisson when whole days are busier or quieter (Ibrahim et al., 2016; Aldor-Noiman et al., 2009).
**Evidence.** Ibrahim et al. (2016); Aldor-Noiman et al. (2009); Cameron & Trivedi (1990).
**Addressed by.** Poisson–Gamma mixture (theorem of total probability), Var(X) = λ + vλ², v estimated by moments, and a
Pearson chi-square test of the plain Poisson model (dashboard: *Forecast check*, *Variation between days*).

### G11. Forecast validation gap *(second review)*
**What it is.** Canteen results are checked on the same days used to estimate them. Every forecasting paper read for
this review judges a method on held-out days (Weinberg et al., 2007; Aldor-Noiman et al., 2009; Matteson et al., 2011),
and McNulty (1988) adds the Brier skill score for probability forecasts.
**Evidence.** Weinberg et al. (2007); Aldor-Noiman et al. (2009); Matteson et al. (2011); McNulty (1988); Jiao et al. (2022).
**Addressed by.** Rolling one-day-ahead backtest: Brier score and skill score, coverage and peak hit-rate z-tests,
paired t and sign tests, Kolmogorov-Smirnov and chi-square tests of the PIT, predicted-vs-actual correlation
(dashboard: *Forecast check*).

---

## 5A. Syllabus Map (PAMAT501)

Every statistical component and the syllabus module it uses. The second-review methods were chosen so that **only
syllabus methods** are needed; factor models, MCMC, integer-GARCH and machine learning from the forecasting papers are
cited as context only.

| Component | Syllabus topics | Module(s) | Gap |
| :--- | :--- | :-: | :-: |
| Poisson model of slot arrivals, PMF and CDF | Discrete random variables, PMF, CDF; Poisson distribution | M1, M2 | G1 |
| Overload probability P(X > C) = 1 − P(X ≤ C) | Complement rule, CDF; Poisson distribution | M1, M2 | G1 |
| Estimating λ by the sample mean (maximum likelihood) | Estimation of parameters | M4 | G4 |
| Exact confidence interval for λ | Estimation of parameters; chi-square and Gamma distributions | M2, M4 | G4 |
| Index-of-dispersion test (Var = Mean) | Chi-square test for variance | M5 | G2 |
| Chi-square goodness of fit of the Poisson model | Chi-square goodness of fit | M5 | G2 |
| Least-crowded slot vs runner-up | Conditional distribution; Binomial distribution; hypothesis test | M1, M2, M5 | G5 |
| Monte Carlo check of P(X > C) | Random variables; independence; Poisson distribution | M1, M2 | G4 |
| Day-specific rates: regression on slot, weekday, exam day | Multiple linear regression | M3 | G7 |
| t-tests of the regression coefficients | t-test | M5 | G7 |
| Weekday effect on daily arrivals | One-way ANOVA; Kruskal-Wallis test | M5 | G7 |
| Exam-day effect on daily arrivals | Two-sample t-test; Mann-Whitney U test | M5 | G7 |
| P(slot t is the day's peak), P(any overload) | Joint distributions; independence; Gamma and Poisson | M1, M2 | G8 |
| Same-day update of the busyness factor | Bayes theorem; conditional probability; Gamma and Poisson | M1, M2 | G9 |
| Poisson–Gamma (negative binomial) overload probability | Theorem of total probability; moments; Gamma and Poisson | M1, M2 | G10 |
| Pearson chi-square test of overdispersion | Chi-square test | M5 | G10 |
| Coverage of 95% prediction intervals | Large-sample test of a single proportion | M4 | G11 |
| Peak-slot hit rate against chance | Large-sample test of a single proportion | M4 | G11 |
| Model vs per-slot average, day by day | Paired t-test; sign test | M5 | G11 |
| Calibration of forecast distributions (PIT) | Kolmogorov-Smirnov test; chi-square goodness of fit | M5 | G11 |
| Predicted vs actual counts | Correlation; rank correlation | M3 | G11 |
| Brier score and skill score of P(X > C) | Mathematical expectation (mean squared error) | M3 | G11 |
| Carry-over between consecutive slots | Rank correlation (Spearman) | M3 | G9 |
| Day-to-day dependence of busy days | Run test | M5 | G11 |

| Module | Components |
| :--- | :-: |
| M1: Foundations of Probability and Random Variables | 7 |
| M2: Probability Distributions | 8 |
| M3: Correlation and Regression Analysis | 4 |
| M4: Statistical Inference: Large Samples | 4 |
| M5: Statistical Inference: Small Samples and Non-parametric Tests | 10 |

Module 6 (contemporary issues, guest lectures) has no statistical content to map.

---

## 6. Proposed Statistical Framework

| Step | What happens | Implemented in |
| :--- | :--- | :--- |
| 1. Observe | Count arrivals at the entry point (manual count or permitted video) in fixed slots across m days | `data_generator.generate_dataset`, `utils.load_dataset` |
| 2. Clean | Drop missing, negative, fractional, unreadable and duplicate counts (no imputation) | `utils.clean_dataset` |
| 3. Group | Separate records by time slot | `analysis.summarize_by_group` |
| 4. Estimate | λ̂ₜ = Σx / m with an exact confidence interval | `poisson_model.estimate_lambda`, `poisson_model.lambda_confidence_interval` |
| 5. Test fit | Dispersion test and chi-square goodness of fit; multiple-testing check | `goodness_of_fit.dispersion_test`, `goodness_of_fit.chi_square_gof`, `analysis.gof_rejection_summary` |
| 6. Compute | P(Xₜ > C) by the textbook sum and by SciPy; Monte Carlo check | `poisson_model.overload_probability(_manual)`, `poisson_model.simulate_overload_frequency` |
| 7. Present | Risk labels, least-crowded slot, recommendations, interactive charts | `poisson_model.classify_risk`, `analysis.quietest_slots`, `analysis.generate_recommendations`, `app.py` |
| 8. Forecast *(2nd review)* | Day-specific rates by regression; Poisson–Gamma P(X > C); simulated P(peak); Bayesian same-day update | `forecast.fit_rate_model`, `forecast.estimate_day_variance`, `forecast.peak_time_distribution`, `forecast.same_day_update` |
| 9. Validate *(2nd review)* | Forecast each day from earlier days only; Brier skill, z-tests, paired t and sign tests, KS and χ² tests of the PIT | `forecast.rolling_backtest`, `forecast.summarize_backtest` |

The reference study motivates examining Poisson arrival behaviour, but the collected data decide whether the model
suits this canteen. It is **not assumed by default**, and an unsupported fit is reported rather than forced.

---

## 7. Mathematical Foundation

### 7.1 Poisson model
Xₜ is the number of students arriving during slot t, with Xₜ ~ Poisson(λₜ):

$$P(X_t = x) = \frac{e^{-\lambda_t}\lambda_t^{x}}{x!},\qquad x = 0, 1, 2, \dots \qquad E[X_t] = \mathrm{Var}(X_t) = \lambda_t$$

λₜ is estimated separately for each slot because arrival intensity changes across the peak.

### 7.2 Estimating λₜ
For slot t observed on m days, maximising $\ln L(\lambda) = -m\lambda + \ln\lambda\sum x_i - \sum\ln x_i!$ gives

$$\hat\lambda_t = \frac{1}{m}\sum_{i=1}^{m} x_i.$$

Exact (Garwood) confidence interval, using $S = \sum x_i \sim \mathrm{Poisson}(m\lambda)$:

$$\left[\frac{\chi^2_{\alpha/2}(2S)}{2m},\ \frac{\chi^2_{1-\alpha/2}(2S+2)}{2m}\right]$$

### 7.3 Overload probability
Overload means more than C students arrive in a slot. With threshold K = C + 1 (the first-review notation):

$$P(X_t > C) = P(X_t \ge K) = 1 - \sum_{x=0}^{C}\frac{e^{-\hat\lambda_t}\hat\lambda_t^{x}}{x!}$$

It is computed both by the explicit sum (log-gamma form, for explanation) and by `scipy.stats.poisson.sf`; automated
tests check that they agree.

### 7.4 Dispersion test
$$D = \frac{(m-1)s^2}{\bar x} \sim \chi^2_{m-1}\ \text{under } H_0,\ \text{two-sided}$$
Var/Mean > 1 suggests overdispersion (for example group arrivals); < 1 underdispersion. Equal mean and variance is
necessary for a Poisson model but not proof of one.

### 7.5 Chi-square goodness of fit
H₀: counts follow Poisson(λ̂). H₁: they do not.

$$\chi^2 = \sum_{i=1}^{k}\frac{(O_i - E_i)^2}{E_i},\qquad E_i = m\,P(X \in \text{category } i \mid \hat\lambda),\qquad df = k - 2$$

Adjacent counts are merged left to right until every expected frequency is at least 5 (Cochran's rule); a leftover
tail joins the previous category. Binning depends only on λ̂ and m. If fewer than three categories remain, the test is
reported as not valid. Across all slots, the number of rejections is compared with Binomial(number of slots, α).

### 7.6 Comparing the two quietest slots
Under H₀: λ₁ = λ₂, conditional on the combined total,

$$S_1 \mid (S_1 + S_2) \sim \mathrm{Binomial}\left(S_1 + S_2,\ \frac{m_1}{m_1 + m_2}\right),$$

and the one-sided p-value is P(Binomial ≤ S₁) (Przyborowski & Wileński, 1940).

### 7.7 Risk classification
| Risk | Rule | Default | Meaning |
| :-: | :-: | :-: | :--- |
| Low | P < τ_low | < 20% | Relatively unlikely to exceed capacity |
| Medium | τ_low ≤ P ≤ τ_high | 20%–50% | Moderate chance of exceeding capacity |
| High | P > τ_high | > 50% | More likely than not to exceed capacity |

The thresholds are project-defined decision rules, adjustable in the dashboard.

### 7.8 Day-specific rates by regression *(second review, Module 3)*
With dummy variables for slot t, weekday d and day type e (reference levels dropped), least squares on log counts:

$$\ln(x_{dt} + 0.5) = \beta_0 + \alpha_t + \gamma_d + \delta_e + \varepsilon_{dt},\qquad \hat\beta = (X^\top X)^{-1}X^\top y$$

$$\hat\lambda_{dt} = \exp(x_{dt}^\top\hat\beta)\cdot\overline{e^{\hat\varepsilon}} - 0.5,\qquad t_j = \hat\beta_j/\mathrm{SE}(\hat\beta_j) \sim t_{n-p}$$

The factor $\overline{e^{\hat\varepsilon}}$ (Duan's smearing estimate) turns the fitted log-mean back into a mean count.
Weekday effects on daily totals are tested with one-way ANOVA and Kruskal-Wallis; exam days with Welch's t-test and
Mann-Whitney U (Module 5).

### 7.9 Day-to-day variation: Poisson–Gamma model *(Modules 1, 2, 5)*
Each day has a busyness factor W with mean 1 and variance v. By the theorem of total probability,

$$X \mid W \sim \mathrm{Poisson}(W\lambda),\quad W \sim \mathrm{Gamma}(1/v,\,1/v)\ \Rightarrow\ X \sim \mathrm{NegBin},\quad E[X] = \lambda,\quad \mathrm{Var}(X) = \lambda + v\lambda^2$$

$$\hat v = \frac{\chi^2_P - (n-p)}{\sum\hat\lambda},\qquad \chi^2_P = \sum\frac{(x-\hat\lambda)^2}{\hat\lambda} \sim \chi^2_{n-p}\ \text{under the plain Poisson model}$$

With v = 0 this is the ordinary Poisson model of Section 7.1.

### 7.10 Peak-time probability and same-day update *(Modules 1, 2)*
P(slot t is the peak) is estimated by simulating whole days: draw W, then $X_t \sim \mathrm{Poisson}(W\lambda_t)$ for
every slot, and record which slot is largest (ties broken at random). After the first k slots of today, Bayes theorem
with the Gamma prior gives

$$W \mid x_1,\dots,x_k \sim \mathrm{Gamma}\Big(a + \sum_{t\le k} x_t,\ a + \sum_{t\le k}\lambda_t\Big),\qquad a = 1/v,$$

and each later slot has a negative binomial predictive distribution, which gives the updated P(Xₜ > C).

### 7.11 Scoring and testing forecasts *(Modules 3, 4, 5)*
Overload in a slot is a yes/no event o with forecast probability p (Brier, 1950; McNulty, 1988):

$$B = \frac{1}{N}\sum (p_i - o_i)^2,\qquad \mathrm{BSS} = 1 - B_{\text{model}}/B_{\text{per-slot average}}$$

| Check | Test | Module |
| :--- | :--- | :-: |
| 95% prediction intervals cover 95% of counts | z-test of a single proportion | M4 |
| Predicted peak slot is right more often than chance | z-test of a single proportion (one-sided) | M4 |
| Model error smaller than the per-slot average | paired t-test; sign test | M5 |
| Forecast distributions calibrated | Kolmogorov–Smirnov and chi-square tests of the PIT (randomised for counts) | M5 |
| Predicted and actual counts agree | Pearson and Spearman correlation | M3 |
| Carry-over between consecutive slots | Spearman rank correlation of observed ÷ predicted | M3 |
| Busy days follow busy days | Wald–Wolfowitz run test on daily residual signs | M5 |

---

## 8. Dataset and Data Collection

**Collection protocol.** Arrivals are counted at the canteen entry point (manual count or permitted video), following
the observation style of Tang et al. (2019a), in fixed 5-, 10- or 15-minute slots during the lunch peak, on m separate
working days. Recorded attributes: date, day, time slot, arrival count and, optionally, the day type (Regular / Exam).

**Status.** Field data collection is in progress. Until it is complete, the analysis runs on **synthetic prototype
data**, clearly labelled as such in the dashboard. These are not observed results.

**Prototype data.** `data/canteen_arrivals.csv` holds 90 working days (one 18-week semester) × 8 fifteen-minute slots
(12:30–14:30), generated with a fixed seed (42):
- baseline demand: 14 arrivals per 15 minutes plus a Gaussian lunch rush centred at 13:15;
- day-to-day variation: a Gamma(k, 1/k) attendance factor with 5% coefficient of variation;
- weekday effects: Monday 1.02, Tuesday 1.00, Wednesday 1.01, Thursday 1.00, Friday 0.94;
- exam weeks: week 9 (mid-semester) and week 18 (end-semester) are marked `day_type = Exam` with demand × 0.85;
- counts drawn from Poisson(rate × day factor); 0.5% of counts blanked to exercise the cleaning step.

The analysis code never sees the generating profile; it re-estimates every rate from the counts alone.

**Schema.**
```csv
date,day,day_type,time_interval,hour,arrivals,period,capacity
2026-01-05,Monday,Regular,12:30-12:45,12,32,Lunch,30
```
Only `date`, `time_interval` and `arrivals` are required for field data; other columns are derived. A missing
`day_type` is treated as `Regular`.

**Other slot lengths.** `python data_generator.py --interval 10` (or 5) generates 10- or 5-minute slots with rates and
the default capacity scaled to the slot length. The dashboard can generate these directly (sidebar, Dataset).

---

## 9. Dashboard

`streamlit run app.py` opens the dashboard. It is built to be self-explanatory: nine pages in three sections, each
page titled with the question it answers and the answer shown directly underneath.

| Section | Page | Question it answers | What is on it (at most three tabs) |
| :--- | :--- | :--- | :--- |
| Plan | **When to go** (home) | When should I go to the canteen? | Day and exam-week picker; busiest time, times to avoid and the quietest time in plain words; chance of overcrowding per slot; "find a time that suits you"; animated **3D canteen** of a simulated lunch hour |
| Analysis | Arrival pattern | When do students arrive? | λ̂ per slot with confidence intervals and the quietest-slot test; day-to-day variation (box plot, weekday profile, heatmap); data cleaning and records |
| | Model check | Is the Poisson model right? | Chi-square and dispersion tests for one slot; all-slot results with the multiple-testing check; Monte Carlo simulation |
| | Overload risk | How likely is overcrowding in each slot? | P(X > C) per slot with risk levels and recommendations; what-if capacity curve, worked calculation and optional 3D surface; capacity planning |
| | Peak forecast | Can we predict the peak time? | P(each slot is the peak) and **3D risk surface** by weekday and time; updating during the day with today's counts; regression model and day-effect tests |
| | Forecast check | Can we trust the forecast? | Hypothesis-test table with syllabus modules; calibration, PIT and daily errors (optional 3D view); day-to-day variation and carry-over tests |
| Report | About the project | — | Title, team, summary, objectives, syllabus map; literature (18 studies) and research gaps (G1–G11) |
| | Methodology | How is the analysis done? | Nine-step framework; formulas by topic; interactive Poisson explorer |
| | Limitations & references | — | Limitations and future scope; full reference list; CSV downloads |

**Design rules.** The sidebar shows only the data source and the counter capacity; significance level, risk
thresholds, slot grouping, demand-level method and date/weekday filters sit under one collapsed *Advanced settings*
panel with sensible defaults. Every chart has a one-line caption and an *About this chart* button for details, instead
of side notes. The student page uses plain words ("chance of overcrowding"); the analysis pages show the notation.
3D is used where it explains something (the canteen, the weekday × time risk surface); other 3D views are optional
switches. Clicking a bar in the arrival-rate or overload chart selects that slot on every page. No plugins or extra
services are needed: Plotly (already a dependency) draws the 3D views in the browser.

---

## 10. System Architecture

```
app.py              entry point: sidebar settings, data loading, pipeline, navigation
 ├── report_pages.py   one function per dashboard page (question title, answer, ≤ 3 tabs)
 ├── ui.py             shared styling, page template (page_header, chart_block, slot_picker) and context
 └── literature.py     literature survey and research gaps (also used in this README)
analysis.py          pipeline orchestration, risk, recommendations, least-crowded slot
 ├── poisson_model.py  λ estimation, PMF/CDF, P(X > C), confidence intervals, Monte Carlo
 └── goodness_of_fit.py  chi-square test with Cochran pooling, dispersion test
forecast.py          regression rates, Poisson–Gamma model, peak-time probability, Bayes update, backtest and tests
visualization.py     Plotly charts, including the 3D surfaces and the animated 3D canteen
utils.py             validation, cleaning, slot-length detection
data_generator.py    reproducible prototype data (5-, 10- or 15-minute slots)
```

```
canteenflow/
├── app.py  report_pages.py  ui.py  literature.py
├── analysis.py  poisson_model.py  goodness_of_fit.py  forecast.py
├── visualization.py  utils.py  data_generator.py
├── data/canteen_arrivals.csv
├── tests/  (test_poisson.py, test_analysis.py, test_data_generator.py, test_quietest.py, test_forecast.py)
├── .streamlit/config.toml
├── requirements.txt  pytest.ini  README.md
```

---

## 11. Installation and Running

Requires Python 3.10 or later (tested on Python 3.14).

```bash
cd canteenflow
python3 -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
pytest -v                          # 83 automated tests
streamlit run app.py               # opens http://localhost:8501
```

Optional: regenerate prototype data with `python data_generator.py --days 90 --seed 7 --interval 10`.

**Research paper.** `paper/canteenflow_ieee.pdf` is the IEEE conference-format write-up of this project. Every figure
and results table in it is generated from the project code, so it can be rebuilt after field data replace the
prototype records:

```bash
pip install -r paper/requirements.txt          # matplotlib, for the figures
python paper/make_figures.py                   # figures, results.json and table rows from the current data
cd paper && pdflatex canteenflow_ieee && bibtex canteenflow_ieee && pdflatex canteenflow_ieee && pdflatex canteenflow_ieee
```

LaTeX needs the IEEEtran class (TeX Live: `texlive-publishers`). Numbers quoted in the paper's text should be
checked against `paper/results.json` after a rebuild.

---

## 12. Results on the Prototype Dataset

90 working days (one semester, two exam weeks), 15-minute slots, capacity C = 30, α = 0.05, thresholds 20% / 50%.

### 12.1 Overload risk per slot (first-review analysis)

| Slot | λ̂ | 95% CI | s² | Var/Mean | P(X > 30) | Risk | χ² p-value |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 12:30–12:45 | 24.47 | 23.46–25.51 | 26.66 | 1.089 | 11.38% | Low | 0.450 |
| 12:45–13:00 | 29.42 | 28.31–30.56 | 39.37 | 1.338 | 40.98% | Medium | 0.552 |
| 13:00–13:15 | 33.37 | 32.18–34.58 | 42.01 | 1.259 | 68.23% | High | 0.309 |
| 13:15–13:30 | 34.36 | 33.16–35.59 | 37.87 | 1.102 | 73.94% | High | 0.701 |
| 13:30–13:45 | 29.04 | 27.94–30.18 | 29.84 | 1.027 | 38.26% | Medium | 0.218 |
| 13:45–14:00 | 24.72 | 23.70–25.77 | 30.93 | 1.251 | 12.43% | Low | 0.310 |
| 14:00–14:15 | 19.12 | 18.23–20.05 | 25.55 | 1.336 | 0.76% | Low | 0.148 |
| 14:15–14:30 | 16.33 | 15.50–17.20 | 20.29 | 1.243 | 0.08% | Low | 0.257 |

- **Model fit.** H₀ rejected in 0 of 8 slots (0.4 expected by chance); no slot-level evidence against the Poisson model.
- **Overload risk.** 13:00–13:30 is high risk; 12:45–13:00 and 13:30–13:45 are medium.
- **Capacity.** Keeping P(overload) below 20% in every slot needs a capacity of 39 per slot at the peak (currently 30).
- **Least-crowded time.** 14:15–14:30 (λ̂ = 16.33), significantly quieter than 14:00–14:15 (λ̂ = 19.12; exact test
  p ≈ 5 × 10⁻⁶). It is the last slot before service ends, so the planner can search within a preferred window.

### 12.2 Peak-time forecast (second review, G7–G9)

- **Regression (R² = 0.579 on the log scale).** Relative to a regular Monday: Friday × 0.906 (t = −3.83,
  p < 0.001), Thursday × 0.946 (p = 0.032), exam day × 0.887 (t = −4.62, p < 0.001); Tuesday and Wednesday are not
  significant. On daily totals the weekday effect is only borderline (ANOVA p = 0.091, Kruskal-Wallis p = 0.089), while
  exam days are clearly quieter (191.9 vs 212.4 arrivals per lunch; Welch t p < 0.001, Mann-Whitney p = 0.005).
- **Predicted peak.** On a regular Monday the most likely peak is 13:15–13:30 (P = 42%) and the peak falls in
  13:00–13:30 with P = 75%. P(at least one slot over capacity) is 98% on a regular Monday, 90% on a regular Friday and
  87% on an exam-week Monday (expected overloaded slots 3.0, 2.0 and 1.8).
- **Same-day update.** If the first two slots of a regular Monday bring 15% more students than forecast (30 and 36),
  the posterior mean of the busyness factor is 1.041 and P(X > 30) at 13:15–13:30 rises from 80.7% to 86.6%.

### 12.3 Validation on unseen days (second review, G10–G11)

60 days forecast one day ahead (training on all earlier days, at least 30), 477 slot forecasts.

| Check | Result | Test | p-value | Supports the model |
| :--- | :--- | :--- | :-: | :-: |
| 95% interval coverage | 93.7% | z-test of a proportion (M4), z = −1.29 | 0.196 | Yes |
| Peak slot predicted correctly | 40.0% vs 13.4% by chance | one-sided z-test (M4), z = 6.04 | < 0.001 | Yes |
| Lower daily error than the per-slot average | MAE 4.57 vs 4.62 | paired t-test (M5), t = 0.56 | 0.290 | No (not significant) |
| Wins on more days | 35 of 60 days | sign test (M5) | 0.123 | No (not significant) |
| PIT uniform | D = 0.067 | Kolmogorov–Smirnov (M5) | 0.028 | No |
| PIT histogram flat | χ² = 16.23, df = 9 | chi-square goodness of fit (M5) | 0.062 | Yes |

- **Probability forecasts.** Brier score 0.1394 (model) vs 0.1393 (per-slot average): no overall skill gain. The
  same-day update after two slots lowers the Brier score of the later slots from 0.1240 to 0.1191 (−4.0%).
- **Where the model helps.** Exam days: MAE 4.53 vs 4.78. Other regular days: 4.51 vs 4.56. Fridays: 4.84 vs 4.70.
- **Correlation (M3).** Predicted and actual counts: Pearson r = 0.727.
- **Overdispersion (G10).** Pearson χ² = 812.4 on 704 df, p = 0.003: counts vary more than Poisson allows once
  weekday and exam effects are removed. v̂ = 0.0057 (day-to-day CV ≈ 7.6%); the Poisson–Gamma P(X > 30) at 13:15–13:30
  is 71.8% against 73.9% for the plain Poisson, and the tails of quiet slots widen (12:30–12:45: 12.8% vs 11.4%).
- **Dependence.** Consecutive slots: Spearman ρ positive and significant for 2 of 7 pairs (12:45→13:00, ρ = 0.235,
  p = 0.026; 14:00→14:15, ρ = 0.222, p = 0.038). Run test on daily residuals: 42 runs vs 45.8 expected, p = 0.42, so
  no evidence that busy days cluster.

**Reading these results honestly.** With 90 days of prototype data and mild weekday effects, the per-slot average is
hard to beat on accuracy, as Ibrahim et al. (2016) report for fixed-effect models. The forecasting layer adds what the
average cannot give: day-type forecasts (better on exam days), the probability that each slot is the peak, a
calibrated 95% interval, and a same-day update that measurably improves the afternoon forecast. The KS test flags that
the model under-forecasts the highest counts somewhat (excess PIT values near 1). Across eight simulated semesters
(seeds 42 and 1–7) the model's MAE was lower than the average's in all eight, significantly so (paired t, p < 0.05)
in four. All figures will be recomputed on field data.

These are model-based estimates on prototype data; they will be recomputed when field counts are available.

---

## 13. Limitations

1. **Prototype data.** Field collection is in progress; current results use synthetic records.
2. **Arrival risk, not queue length.** Queue length also depends on service rate, number of counters and queue
   discipline.
3. **Constant rate within a slot.** The Poisson model assumes λ does not change inside a slot.
4. **Independence.** Group arrivals produce overdispersion.
5. **Threshold choice.** Capacity C (threshold K = C + 1) must be justified for the canteen and slot length.
6. **Project-defined risk cut-offs.** Low / medium / high thresholds are decision rules, not universal constants.
7. **External events.** Exam days are modelled when marked in the data; holidays, weather and timetable changes are not.
8. **Same peak shape every day.** The regression scales a whole day up or down; it cannot move the peak to another
   slot on special days without slot × day-type interaction terms.
9. **Log-scale regression.** Least squares on log counts is a syllabus-friendly stand-in for Poisson regression; means
   are back-transformed with a smearing factor and slightly under-forecast the highest counts (Section 12.3).

---

## 14. Future Scope

1. **Field data.** Replace the prototype records with entry counts collected over m days.
2. **Queueing extension (M/M/c).** Add service times to estimate queue length and waiting time.
3. **Done at the second review:** Poisson–Gamma probabilities for day-to-day variation, Bayesian same-day updating,
   peak-time probabilities and a validation backtest (gaps G7–G11).
4. **Group arrivals.** Compound Poisson models for students arriving together.
5. **Smooth arrival rate.** Estimate λ(t) continuously as a nonhomogeneous Poisson process (Leemis, 1991).
6. **Live data feed.** Run the same-day update automatically from turnstile or camera counts.
7. **Natural-language and voice interface** (proposed at the first review). A usability layer that retrieves the
   verified results and explains them, optionally with text-to-speech. It would not calculate or invent statistical
   values; the statistical model remains the primary contribution.

---

## 15. References

**Literature**
1. Aldor-Noiman, S., Feigin, P. D., & Mandelbaum, A. (2009). Workload forecasting for a call center: Methodology and a case study. *The Annals of Applied Statistics, 3*(4), 1403–1447. https://doi.org/10.1214/09-AOAS255
2. Brown, L., Gans, N., Mandelbaum, A., Sakov, A., Shen, H., Zeltyn, S., & Zhao, L. (2005). Statistical analysis of a telephone call center: A queueing-science perspective. *Journal of the American Statistical Association, 100*(469), 36–50. https://doi.org/10.1198/016214504000001808
3. Cameron, A. C., & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson model. *Journal of Econometrics, 46*(3), 347–364. https://doi.org/10.1016/0304-4076(90)90014-K
4. Chen, J., & Wang, H. (2011). Optimal management model of university canteen based on mixed queuing theory. *Journal of Huangshi Institute of Technology, 27*, 41–44 (CNKI: HSGD201103013); authors, volume and pages as cited by Tang et al. (2019a).
5. Green, L. V., Kolesar, P. J., & Whitt, W. (2007). Coping with time-varying demand when setting staffing requirements for a service system. *Production and Operations Management, 16*(1), 13–39. https://doi.org/10.1111/j.1937-5956.2007.tb00164.x
6. Ibrahim, R., Ye, H., L'Ecuyer, P., & Shen, H. (2016). Modeling and forecasting call center arrivals: A literature survey and a case study. *International Journal of Forecasting, 32*(3), 865–874. https://doi.org/10.1016/j.ijforecast.2015.11.012
7. Jiao, S., Wang, X., Song, D., & Du, S. (2022). Simulation and optimization of university canteen service under the situation of epidemic prevention and control based on queuing theory. *ITM Web of Conferences, 47*, 02052. https://doi.org/10.1051/itmconf/20224702052
8. Kim, S.-H., & Whitt, W. (2014). Are call center and hospital arrivals well modeled by nonhomogeneous Poisson processes? *Manufacturing & Service Operations Management, 16*(3), 464–480. https://doi.org/10.1287/msom.2014.0490
9. Leemis, L. M. (1991). Nonparametric estimation of the cumulative intensity function for a nonhomogeneous Poisson process. *Management Science, 37*(7), 886–900. https://doi.org/10.1287/mnsc.37.7.886
10. Matteson, D. S., McLean, M. W., Woodard, D. B., & Henderson, S. G. (2011). Forecasting emergency medical service call arrival rates. *The Annals of Applied Statistics, 5*(2B), 1379–1406. https://doi.org/10.1214/10-AOAS442
11. McNulty, R. P. (1988). A new measure of forecast skill using the Brier score. *Central Region Technical Attachment 88-26*, National Weather Service, NOAA.
12. Shah, H., Myller, N., & Islas Sedano, C. (2025). Forecasting daily customer flow in restaurants: a multifactor machine learning approach. *Applied Computing and Intelligence, 5*(2), 168–190. https://doi.org/10.3934/aci.2025011
13. Shen, H., & Huang, J. Z. (2008). Forecasting time series of inhomogeneous Poisson processes with application to call center workforce management. *The Annals of Applied Statistics, 2*(2), 601–623. https://doi.org/10.1214/08-AOAS164
14. Tang, T.-Q., Zhang, B.-T., & Xie, C.-Z. (2019b). Modeling and simulation of pedestrian flow in university canteen. *Simulation Modelling Practice and Theory, 95*, 96–111. https://doi.org/10.1016/j.simpat.2019.04.011
15. Tang, T.-Q., Zhang, B.-T., Zhang, J., & Wang, T. (2019a). Statistical analysis and modeling of pedestrian flow in university canteen during peak period. *Physica A, 521*, 29–40. https://doi.org/10.1016/j.physa.2019.01.060
16. Variskhanov, A., & Arreeras, T. (2026). Early peak badges from Wi-Fi telemetry: A field feasibility study of lunchtime crowd management on a smart campus. *Urban Science, 10*(1), 29. https://doi.org/10.3390/urbansci10010029
17. Weinberg, J., Brown, L. D., & Stroud, J. R. (2007). Bayesian forecasting of an inhomogeneous Poisson process with applications to call center data. *Journal of the American Statistical Association, 102*(480), 1185–1199.
18. Yuan, N., Wang, X., & Chen, X. (2023). Analysis on the service capacity of university canteens under the epidemic prevention based on queue theory. *Transactions on Comparative Education, 5*(7). https://doi.org/10.23977/trance.2023.050706

**Statistical methods**
19. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
20. Cochran, W. G. (1954). Some methods for strengthening the common χ² tests. *Biometrics, 10*(4), 417–451. https://doi.org/10.2307/3001616
21. Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. *Journal of the American Statistical Association, 78*(383), 605–610.
22. Garwood, F. (1936). Fiducial limits for the Poisson distribution. *Biometrika, 28*(3/4), 437–442. https://doi.org/10.1093/biomet/28.3-4.437
23. Przyborowski, J., & Wileński, H. (1940). Homogeneity of results in testing samples from Poisson series. *Biometrika, 31*(3/4), 313–323. https://doi.org/10.1093/biomet/31.3-4.313
24. Ross, S. M. (2014). *Introduction to Probability Models* (11th ed.). Academic Press.

**R Markdown report.** `report/canteenflow_report.pdf` is the course submission document in R Markdown format
(title, subtitle, authors with registration numbers, Abstract, Keywords, Introduction with Aim and Objectives,
Preliminaries, Data reference, results with graphs and tables, Comparison, Conclusion, References, and the R code as an
appendix). The analysis is re-implemented in R inside the `.Rmd`, and every number in the text is computed when the
document is knitted. To rebuild it, open `report/canteenflow_report.Rmd` in RStudio and press **Knit**, or run:

```bash
Rscript -e 'install.packages(c("rmarkdown", "knitr", "ggplot2"))'   # once
Rscript -e 'rmarkdown::render("report/canteenflow_report.Rmd")'
```

The data file, capacity and significance level are set in the `params` block of the header. Simulated quantities
(peak-time probabilities, randomised PIT) use R's random number generator, so they can differ slightly from the
dashboard's figures.

**Presentation.** `presentation/canteenflow_presentation.pptx` is the 21-slide course presentation (title and team,
problem, aim and objectives, research gaps, data, method, formulas, results, peak prediction, same-day update,
validation, comparison, dashboard, conclusion, references), with speaker notes on each slide. Its numbers come from
`paper/results.json`.
