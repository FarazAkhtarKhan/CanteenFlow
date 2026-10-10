# Statistical Estimation of Canteen Arrival Overload Probability Using Poisson Distribution

**System name:** CanteenFlow, an interactive statistical analysis dashboard
**Course:** PAMAT501, Probability and Statistics
**Base paper:** Tang, T.-Q., Zhang, B.-T., Zhang, J., & Wang, T. (2019). *Statistical analysis and modeling of pedestrian flow in university canteen during peak period.* Physica A, 521, 29–40. https://doi.org/10.1016/j.physa.2019.01.060

| Presented by | Register No. |
| :--- | :--- |
| Mayur Krishna | 26MCA0191 |
| Faraz Akthar Khan | 26MCA0197 |
| Malhar Bhosle | 26MCA0199 |

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
alternatives. On the 60-day prototype dataset, the riskiest slot is 13:15–13:30 (P = 75.2% at capacity C = 30) and the
least crowded is 14:15–14:30 (λ̂ ≈ 16.0, significantly quieter than the runner-up, p ≈ 1 × 10⁻⁵).

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
```

---

## 2. Objectives

1. **Collect and organise arrival data.** Record arrivals at fixed 5-, 10- or 15-minute intervals across m days.
2. **Estimate time-specific rates.** Calculate λ̂ₜ for every slot and examine Poisson suitability.
3. **Calculate overload probability.** Estimate P(Xₜ ≥ K) = P(Xₜ > C) for a selected threshold, with its uncertainty.
4. **Identify useful time windows.** Classify intervals as relatively low, medium or high risk, and identify the
   least-crowded slot.

---

## 3. Base Paper

Tang et al. (2019a) study pedestrian flow in the second canteen of Beihang University during the lunch peak.

| Data collection | Detailed model | Main outcome |
| :--- | :--- | :--- |
| Camera-based observation; peak period of about 10:30–13:30; short time segments; arrival and departure counts | Arrival and departure rates, service windows and service time, window selection, movement, congestion, meal duration and seating | A complete pedestrian-flow model, calibrated with data and validated against observations |

The base paper is broad and system-level. **This project focuses on the arrival-count component.**

---

## 4. Literature Survey

Ten studies were reviewed: the base paper and nine others covering canteen modelling, canteen queueing,
arrival-process statistics, time-varying demand, model checking and campus crowd sensing. Every reference was checked
against OpenAlex and its DOI; summaries follow each paper's published abstract.

| # | Study | Area | Focus and method | Relevance to this project |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Tang et al. (2019a), *Physica A* (base paper) | Canteen pedestrian flow | Camera data from the Beihang canteen at lunch peak; statistical analysis and a pedestrian-flow model validated against observations | Shows arrivals can be observed in short segments and analysed statistically; we isolate the arrival-count component |
| 2 | Tang, Zhang & Xie (2019b), *Simulation Modelling Practice and Theory* | Canteen pedestrian flow | Extended cellular-automaton model of each pedestrian's motion, calibrated on the same canteen | Shows how detailed full canteen models are; arrival intensity is an input, not a tested quantity |
| 3 | Wang (2011), *J. Huangshi Inst. of Technology* | Canteen queueing | M/M/n/m queueing model to set serving windows, seats and opening times | Typical queueing treatment; Poisson arrivals are assumed rather than tested |
| 4 | Yuan, Wang & Chen (2023), *Transactions on Comparative Education* | Canteen queueing | Survey data, queueing theory and MATLAB simulation of staggered dining times | Confirms timing is the main crowding lever; does not test when crowds are smallest |
| 5 | Brown et al. (2005), *JASA* | Arrival-process statistics | Call-by-call data for a year; a test that a point process is Poisson and models for forecasting Poisson arrival rates | Methodological model: treat arrivals as their own component and test the Poisson assumption |
| 6 | Green, Kolesar & Whitt (2007), *Production and Operations Management* | Time-varying demand | Review of staffing methods when demand varies during the day | Justifies one λ per short slot instead of one rate for the whole peak |
| 7 | Kim & Whitt (2014), *M&SOM* | Arrival-process statistics | KS tests of nonhomogeneous Poisson arrivals with piecewise-constant rates | Poisson holds only with careful analysis; long subintervals and pooled days cause overdispersion |
| 8 | Leemis (1991), *Management Science* | Time-varying demand | Nonparametric estimate of a time-varying Poisson intensity from repeated realisations | Our per-slot λ̂ from m days is its piecewise-constant counterpart; basis for Monte Carlo generation |
| 9 | Cameron & Trivedi (1990), *Journal of Econometrics* | Model checking | Regression-based tests of Poisson equidispersion | Supports checking Var ≈ Mean before using Poisson probabilities |
| 10 | Variskhanov & Arreeras (2026), *Urban Science* | Campus crowd sensing | Wi-Fi telemetry from five canteens; k-means++ demand shapes and an "Early Peak" badge at 11:28 | Shows real demand for simple lunchtime crowd signals; needs Wi-Fi data and gives a day-level badge |

**Positioning.** Previous work models the broader pedestrian-flow or queueing system, or studies Poisson arrivals in
other service settings. It does not directly provide a simple, statistically checked arrival-risk indicator for a
canteen.

---

## 5. Research Gaps

Six gaps were identified. G1 was stated at the first review; G2 to G6 come from the extended survey. Each is a
**focus and application gap**, not a claim that Poisson modelling of arrivals is new.

### G1. Focus and application gap
**What it is.** The reviewed canteen studies model the complete meal process (movement, windows, seating,
congestion) and output simulated flows or queue sizes. They do not turn time-specific arrival rates into a
threshold-exceedance probability with a risk label, so the question "how likely is overload at 13:15?" has no direct
answer.
**Evidence.** Tang et al. (2019a, 2019b); Wang (2011); Yuan et al. (2023).
**Addressed by.** λ̂ₜ per slot → P(Xₜ > C) → Low / Medium / High label (dashboard: *Overload Probability*).

### G2. Validation gap
**What it is.** Queueing models of canteens start from Markovian (Poisson) arrivals, so the assumption is an input,
not a finding. Kim & Whitt (2014) show the assumption must be tested and that careless tests mislead. Testing many
slots also creates a multiple-testing problem, since some rejections occur by chance.
**Evidence.** Wang (2011); Yuan et al. (2023); Kim & Whitt (2014); Brown et al. (2005); Cameron & Trivedi (1990).
**Addressed by.** Per-slot chi-square goodness of fit with Cochran pooling, an index-of-dispersion test, and a
binomial check of the number of rejections (dashboard: *Goodness of Fit*).

### G3. Time-resolution gap
**What it is.** Lunch demand changes within minutes. A single rate for an hour or the whole peak hides the rush and
makes pooled counts overdispersed. Green et al. (2007) show peak-hour and stationary approximations need refinement
under time-varying demand; Kim & Whitt (2014) trace failed Poisson tests to subintervals where the rate varies too
much.
**Evidence.** Green et al. (2007); Kim & Whitt (2014); Leemis (1991).
**Addressed by.** Piecewise-constant rates on 5-, 10- or 15-minute slots, plus an hourly-pooling view that shows the
resulting overdispersion (dashboard: *Arrival Rate*, *Goodness of Fit*).

### G4. Uncertainty gap
**What it is.** λ̂ is estimated from a limited number of days, so it carries sampling error. A slot near a threshold
can move between Medium and High within the plausible range of λ. Carrying the uncertainty in the estimated arrival
rate through to the reported risk is not part of the reviewed canteen studies' output.
**Evidence.** Tang et al. (2019a); Brown et al. (2005).
**Addressed by.** Exact Garwood confidence intervals, P(X > C) evaluated at both ends of the interval, and a Monte
Carlo check (dashboard: *Overload Probability*, *Simulation Check*).

### G5. Decision-support gap
**What it is.** Variskhanov & Arreeras (2026) observe that many smart-campus tools fail to support routine, day-to-day
decisions; their crowd signal needs Wi-Fi telemetry and gives a day-level badge. Queueing studies give window counts
to managers but tell students nothing about when to go. Guidance that needs only arrival counts (when crowding is
least likely, and whether that slot is reliably quieter) is missing.
**Evidence.** Variskhanov & Arreeras (2026); Yuan et al. (2023); Wang (2011).
**Addressed by.** Least-crowded slot with an exact conditional test against the runner-up, a visit planner, and
capacity recommendations (dashboard: *Least-Crowded Time*, *Risk & Recommendations*).

### G6. Data and context gap
**What it is.** The canteen evidence reviewed comes from specific campuses, mainly Beihang University, and arrival
patterns depend on local timetables. We did not find openly available interval-level arrival counts for an Indian
college canteen, so results from elsewhere cannot be reused directly.
**Evidence.** Tang et al. (2019a, 2019b); Variskhanov & Arreeras (2026).
**Addressed by.** A documented collection protocol, a validated CSV import for field counts, and a clearly labelled
synthetic prototype dataset (dashboard: *Data & EDA*).

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

---

## 8. Dataset and Data Collection

**Collection protocol.** Arrivals are counted at the canteen entry point (manual count or permitted video), following
the observation style of Tang et al. (2019a), in fixed 5-, 10- or 15-minute slots during the lunch peak, on m separate
working days. Recorded attributes: date, day, time slot and arrival count.

**Status.** Field data collection is in progress. Until it is complete, the analysis runs on **synthetic prototype
data**, clearly labelled as such in the dashboard. These are not observed results.

**Prototype data.** `data/canteen_arrivals.csv` holds 60 working days × 8 fifteen-minute slots (12:30–14:30), generated
with a fixed seed (42):
- baseline demand: 14 arrivals per 15 minutes plus a Gaussian lunch rush centred at 13:15;
- day-to-day variation: a Gamma(k, 1/k) attendance factor with 5% coefficient of variation;
- weekday effects: Monday 1.02, Tuesday 1.00, Wednesday 1.01, Thursday 1.00, Friday 0.94;
- counts drawn from Poisson(rate × day factor); 0.5% of counts blanked to exercise the cleaning step.

The analysis code never sees the generating profile; it re-estimates every rate from the counts alone.

**Schema.**
```csv
date,day,time_interval,hour,arrivals,period,capacity
2026-01-05,Monday,12:30-12:45,12,32,Lunch,30
```
Only `date`, `time_interval` and `arrivals` are required for field data; other columns are derived.

**Other slot lengths.** `python data_generator.py --interval 10` (or 5) generates 10- or 5-minute slots with rates and
the default capacity scaled to the slot length. The dashboard can generate these directly (sidebar, Dataset).

---

## 9. Dashboard

`streamlit run app.py` opens a multi-page dashboard. The sidebar holds the page navigation, the model parameters
(capacity C, significance level α, risk thresholds), analysis options, date and weekday filters, and the *Slot to
inspect*. Clicking a bar in the overview, arrival-rate or overload charts also selects that slot on every page.

| Section | Page | Contents and interaction |
| :--- | :--- | :--- |
| Project | Overview | Title, abstract, key figures, clickable overload-risk chart with a slot summary, objectives |
| | Literature Review | Filterable table of the ten studies; select a row for focus, method, findings and relevance |
| | Research Gaps | Gap selector (G1–G6) with explanation, evidence and where each gap is addressed |
| | Methodology | Seven-step framework explorer, formulas, interactive Poisson explorer (λ and C sliders) |
| Analysis | Data & EDA | Cleaning report, collection protocol, box plots per slot, weekday profiles, heatmap, records |
| | Arrival Rate (λ̂) | λ̂ with confidence intervals (clickable), worked estimate, summary statistics |
| | Goodness of Fit | Chi-square test per slot, categories, dispersion test, all-slot results |
| | Overload Probability | P(X > C) per slot (clickable), what-if curve, worked calculation, capacity planning slider |
| | Simulation Check | Animated Monte Carlo convergence; one simulated lunch service on demand |
| Results | Risk & Recommendations | Risk distribution, risk table, generated recommendations, CSV download |
| | Least-Crowded Time | Ranking with significance test, visit planner, weekday view |
| Conclusion | Limitations & Future Scope | Limitations and planned extensions |
| | References & Downloads | Full reference list and data downloads |

Every chart has a reading note beside it (toggle *Show reading notes* in the sidebar).

---

## 10. System Architecture

```
app.py              entry point: sidebar settings, data loading, pipeline, navigation
 ├── report_pages.py   one function per dashboard page
 ├── ui.py             shared styling, layout helpers and page context
 └── literature.py     literature survey and research gaps (also used in this README)
analysis.py          pipeline orchestration, risk, recommendations, least-crowded slot
 ├── poisson_model.py  λ estimation, PMF/CDF, P(X > C), confidence intervals, Monte Carlo
 └── goodness_of_fit.py  chi-square test with Cochran pooling, dispersion test
visualization.py     Plotly charts
utils.py             validation, cleaning, slot-length detection
data_generator.py    reproducible prototype data (5-, 10- or 15-minute slots)
```

```
canteenflow/
├── app.py  report_pages.py  ui.py  literature.py
├── analysis.py  poisson_model.py  goodness_of_fit.py
├── visualization.py  utils.py  data_generator.py
├── data/canteen_arrivals.csv
├── tests/  (test_poisson.py, test_analysis.py, test_data_generator.py, test_quietest.py)
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
pytest -v                          # 61 automated tests
streamlit run app.py               # opens http://localhost:8501
```

Optional: regenerate prototype data with `python data_generator.py --days 90 --seed 7 --interval 10`.

---

## 12. Results on the Prototype Dataset

60 working days, 15-minute slots, capacity C = 30, α = 0.05, thresholds 20% / 50%.

| Slot | λ̂ | 95% CI | s² | Var/Mean | P(X > 30) | Risk | χ² p-value |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 12:30–12:45 | 23.71 | 22.49–24.99 | 22.48 | 0.948 | 8.58% | Low | 0.775 |
| 12:45–13:00 | 29.23 | 27.88–30.63 | 40.22 | 1.376 | 39.61% | Medium | 0.717 |
| 13:00–13:15 | 33.38 | 31.94–34.88 | 44.07 | 1.320 | 68.34% | High | 0.571 |
| 13:15–13:30 | 34.58 | 33.11–36.10 | 32.42 | 0.937 | 75.17% | High | 0.673 |
| 13:30–13:45 | 29.25 | 27.89–30.67 | 28.47 | 0.973 | 39.76% | Medium | 0.638 |
| 13:45–14:00 | 24.17 | 22.94–25.44 | 19.77 | 0.818 | 10.20% | Low | 0.403 |
| 14:00–14:15 | 19.25 | 18.16–20.39 | 21.89 | 1.137 | 0.83% | Low | 0.108 |
| 14:15–14:30 | 15.97 | 14.97–17.01 | 15.22 | 0.953 | 0.05% | Low | 0.785 |

- **Model fit.** H₀ rejected in 0 of 8 slots (0.4 expected by chance); no overall evidence against the Poisson model.
- **Overload risk.** 13:00–13:30 is high risk; 12:45–13:00 and 13:30–13:45 are medium.
- **Capacity.** Keeping P(overload) below 20% in every slot needs a capacity of 39 per slot at the peak (currently 30).
- **Least-crowded time.** 14:15–14:30 (λ̂ = 15.97), significantly quieter than 14:00–14:15 (λ̂ = 19.25; exact test
  p ≈ 1 × 10⁻⁵). It is the last slot before service ends, so the planner can search within a preferred window.

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
7. **External events.** Exams, holidays, weather and timetable changes shift arrival patterns.

---

## 14. Future Scope

1. **Field data.** Replace the prototype records with entry counts collected over m days.
2. **Queueing extension (M/M/c).** Add service times to estimate queue length and waiting time.
3. **Overdispersed counts.** Negative binomial or compound Poisson models for group arrivals.
4. **Smooth arrival rate.** Estimate λ(t) continuously as a nonhomogeneous Poisson process (Leemis, 1991).
5. **Live updating.** Bayesian Gamma–Poisson updating of λ from turnstile or camera counts.
6. **Natural-language and voice interface** (proposed at the first review). A usability layer that retrieves the
   verified results and explains them, optionally with text-to-speech. It would not calculate or invent statistical
   values; the statistical model remains the primary contribution.

---

## 15. References

**Literature**
1. Brown, L., Gans, N., Mandelbaum, A., Sakov, A., Shen, H., Zeltyn, S., & Zhao, L. (2005). Statistical analysis of a telephone call center: A queueing-science perspective. *Journal of the American Statistical Association, 100*(469), 36–50. https://doi.org/10.1198/016214504000001808
2. Cameron, A. C., & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson model. *Journal of Econometrics, 46*(3), 347–364. https://doi.org/10.1016/0304-4076(90)90014-K
3. Green, L. V., Kolesar, P. J., & Whitt, W. (2007). Coping with time-varying demand when setting staffing requirements for a service system. *Production and Operations Management, 16*(1), 13–39. https://doi.org/10.1111/j.1937-5956.2007.tb00164.x
4. Kim, S.-H., & Whitt, W. (2014). Are call center and hospital arrivals well modeled by nonhomogeneous Poisson processes? *Manufacturing & Service Operations Management, 16*(3), 464–480. https://doi.org/10.1287/msom.2014.0490
5. Leemis, L. M. (1991). Nonparametric estimation of the cumulative intensity function for a nonhomogeneous Poisson process. *Management Science, 37*(7), 886–900. https://doi.org/10.1287/mnsc.37.7.886
6. Tang, T.-Q., Zhang, B.-T., & Xie, C.-Z. (2019b). Modeling and simulation of pedestrian flow in university canteen. *Simulation Modelling Practice and Theory, 95*, 96–111. https://doi.org/10.1016/j.simpat.2019.04.011
7. Tang, T.-Q., Zhang, B.-T., Zhang, J., & Wang, T. (2019a). Statistical analysis and modeling of pedestrian flow in university canteen during peak period. *Physica A, 521*, 29–40. https://doi.org/10.1016/j.physa.2019.01.060
8. Variskhanov, A., & Arreeras, T. (2026). Early peak badges from Wi-Fi telemetry: A field feasibility study of lunchtime crowd management on a smart campus. *Urban Science, 10*(1), 29. https://doi.org/10.3390/urbansci10010029
9. Wang, H. (2011). Optimal management model of university canteen based on mixed queuing theory. *Journal of Huangshi Institute of Technology* (CNKI: HSGD201103013).
10. Yuan, N., Wang, X., & Chen, X. (2023). Analysis on the service capacity of university canteens under the epidemic prevention based on queue theory. *Transactions on Comparative Education, 5*(7). https://doi.org/10.23977/trance.2023.050706

**Statistical methods**
11. Cochran, W. G. (1954). Some methods for strengthening the common χ² tests. *Biometrics, 10*(4), 417–451. https://doi.org/10.2307/3001616
12. Garwood, F. (1936). Fiducial limits for the Poisson distribution. *Biometrika, 28*(3/4), 437–442. https://doi.org/10.1093/biomet/28.3-4.437
13. Przyborowski, J., & Wileński, H. (1940). Homogeneity of results in testing samples from Poisson series. *Biometrika, 31*(3/4), 313–323. https://doi.org/10.1093/biomet/31.3-4.313
14. Ross, S. M. (2014). *Introduction to Probability Models* (11th ed.). Academic Press.
