# Statistical Estimation of Canteen Arrival Overload Probability Using Poisson Distribution

**System Name:** CanteenFlow — Poisson-Based Canteen Overload Risk Estimation  
**Project Level:** MCA / Postgraduate Academic Research & Prototype  
**Technology Stack:** Python 3, Pandas, NumPy, SciPy, Plotly, Streamlit, Pytest  

---

## 1. Project Title
**Statistical Estimation of Canteen Arrival Overload Probability Using Poisson Distribution**  
*Alternative Short System Name:* **CanteenFlow**

---

## 2. Problem Statement
College canteens experience irregular student traffic throughout operating hours. During specific peak intervals—such as lunch breaks or mid-morning lecture intermissions—the number of arriving students can surge rapidly. When arrival volume exceeds the canteen's serving and queue-handling capacity, several operational bottlenecks emerge:
- Extended queue lengths and excessive student waiting times.
- Congestion at service and payment counters.
- Food depletion and staff stress.

Traditional canteen management relies either on ad-hoc intuition or fixed schedule guesses. To manage service capacity efficiently, institutions require a **rigorous probabilistic framework** that models arrival counts, estimates expected rates from empirical data, tests whether theoretical assumptions hold, and quantifies the probability of capacity overload.

---

## 3. Objectives
1. **Data Generation & Preprocessing:** Synthesize a realistic, reproducible multi-day student arrival dataset across fixed 15-minute operating intervals, clearly labelled as prototype research data.
2. **Rate Parameter Estimation:** Estimate the Poisson intensity parameter $\hat{\lambda}$ for each distinct operating time period.
3. **Statistical Property Verification:** Investigate the relationship between the sample mean and sample variance ($E[X] \approx \mathrm{Var}(X)$) across intervals.
4. **Goodness-of-Fit Validation:** Formulate and execute a Chi-Square Goodness-of-Fit ($\chi^2$) test with systematic category pooling (Cochran's rule) to verify if the Poisson model reasonably fits the empirical data.
5. **Overload Probability Computation:** Calculate the exact probability that student arrivals exceed the configured service capacity ($P(X > C)$) for each period.
6. **Risk Stratification:** Classify operating periods into transparent, user-configurable risk categories (`LOW`, `MEDIUM`, `HIGH`).
7. **Interactive Visual Dashboard:** Provide an explainable Streamlit application with step-by-step mathematical expanders for academic viva demonstration.

---

## 4. Research Methodology & Pipeline
The complete analytical pipeline proceeds in ten disciplined phases:

```
[1. Data Ingestion & Generation]
              ↓
    [2. Data Cleaning]
              ↓
  [3. Exploratory Analysis]
              ↓
     [4. Estimate λ]
              ↓
  [5. Check Mean vs Variance]
              ↓
[6. Poisson Goodness-of-Fit]
              ↓
 [7. Calculate P(X > Capacity)]
              ↓
  [8. Risk Classification]
              ↓
     [9. Visualization]
              ↓
[10. Operational Recommendations]
```

1. **Data Ingestion & Simulation:** Ingests arrival records or simulates realistic student arrivals with time-of-day demand baselines, day-to-day Gamma random factors, weekday multipliers, and Poisson counts.
2. **Data Cleaning:** Validates data integrity, eliminates invalid/negative counts, handles recording gaps without variance-distorting imputation, and removes duplicate stamps.
3. **Exploratory Analysis:** Summarizes arrival frequencies, calculates descriptive metrics, and identifies temporal patterns.
4. **Parameter Estimation:** Computes Maximum Likelihood Estimates $\hat{\lambda} = \frac{\sum x}{n}$ and Garwood exact confidence intervals.
5. **Mean vs. Variance Inspection:** Assesses dispersion indices ($\mathrm{Var} / \mathrm{Mean}$) and runs the two-sided index-of-dispersion test.
6. **Goodness-of-Fit Evaluation:** Evaluates $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ with degrees of freedom $df = k - 2$ after combining sparse tail bins ($E_i \ge 5.0$).
7. **Overload Probability Calculation:** Computes $P(X > C) = 1 - \sum_{x=0}^C \frac{e^{-\lambda}\lambda^x}{x!}$ using both explicit textbook summation and SciPy survival functions.
8. **Risk Stratification:** Maps overload probabilities to operational risk tiers (`LOW`, `MEDIUM`, `HIGH`) using configurable thresholds.
9. **Visualization:** Renders interactive Plotly charts (PMF overlays, rate curves, overload bar charts, risk donuts, mean vs variance scatter plots, and date-time heatmaps).
10. **Operational Recommendations:** Generates automated, plain-language capacity management recommendations.

---

## 5. Mathematical Foundation

### 5.1 The Poisson Probability Mass Function (PMF)
A discrete random variable $X$ representing the number of student arrivals in a fixed interval $[t, t + \Delta t]$ follows a Poisson distribution with parameter $\lambda > 0$ if its probability mass function is:

$$P(X = x) = \frac{e^{-\lambda} \lambda^x}{x!}, \quad x \in \{0, 1, 2, \dots\}$$

where:
- $X$ = Random variable denoting arrival count in the 15-minute window.
- $\lambda$ = Expected arrival rate (mean arrivals per window).
- $x$ = Specific observed arrival count.
- $e$ = Euler's number ($\approx 2.71828$).

### 5.2 Poisson Distribution Properties
For a true Poisson process:
$$\mathbb{E}[X] = \lambda$$
$$\mathrm{Var}(X) = \lambda$$

Hence, the **dispersion index** satisfies:
$$\frac{\mathrm{Var}(X)}{\mathbb{E}[X]} \approx 1$$

> **Crucial Academic Distinction:** While $\mathrm{Mean} \approx \mathrm{Variance}$ is a necessary characteristic of a Poisson distribution, it does **not** solely prove that a dataset follows a Poisson process. Goodness-of-fit testing over the entire frequency distribution is required.

---

## 6. Parameter Estimation ($\hat{\lambda}$)
Given $n$ independent observations $x_1, x_2, \dots, x_n$ for a specific time interval across $n$ operational days, the Maximum Likelihood Estimator (MLE) for $\lambda$ is:

$$\hat{\lambda} = \frac{1}{n} \sum_{i=1}^n x_i$$

### Exact Confidence Intervals (Garwood Formulation)
Because the sum $S = \sum_{i=1}^n x_i$ of $n$ independent Poisson random variables follows $S \sim \mathrm{Poisson}(n\lambda)$, the exact $(1 - \alpha)$ two-sided confidence interval for $\lambda$ is:

$$\left[ \frac{\chi^2_{\alpha/2}(2S)}{2n}, \quad \frac{\chi^2_{1 - \alpha/2}(2S + 2)}{2n} \right]$$

where $\chi^2_p(k)$ is the quantile function of a Chi-Square distribution with $k$ degrees of freedom.

---

## 7. Mean vs. Variance Analysis
For each time interval, sample mean $\bar{x}$ and sample unbiased variance $s^2$ are computed:
$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i, \qquad s^2 = \frac{1}{n - 1}\sum_{i=1}^n (x_i - \bar{x})^2$$

### Index of Dispersion Test
To test whether the sample dispersion deviates from unity more than expected by random variation, we compute the dispersion statistic:
$$D = \frac{(n - 1) s^2}{\bar{x}} \sim \chi^2_{n - 1} \quad \text{under } H_0$$
A two-sided $p$-value determines whether the period exhibits:
- **Equidispersion** ($D \approx n-1$, $p > \alpha$): Consistent with Poisson.
- **Overdispersion** ($s^2 > \bar{x}$, $p \le \alpha$): Arrival clustering, group arrivals, or time-varying rate.
- **Underdispersion** ($s^2 < \bar{x}$, $p \le \alpha$): Regularly scheduled, paced arrivals.

---

## 8. Chi-Square Goodness-of-Fit Test ($\chi^2$)
To scientifically verify whether observed arrivals in an interval follow the fitted Poisson distribution:

### Hypotheses
- **$H_0$:** Student arrival counts follow a Poisson distribution with parameter $\lambda = \hat{\lambda}$.
- **$H_1$:** Student arrival counts do not follow a Poisson distribution.

### Test Statistic
$$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}$$

where:
- $O_i$ = Observed count of days with arrivals falling into category $i$.
- $E_i = n \cdot P(X \in \text{category } i \mid \hat{\lambda})$ = Expected frequency under the fitted model.
- $k$ = Number of categories after pooling.

### Degrees of Freedom
$$df = k - 1 - 1 = k - 2$$
One degree of freedom is lost because the probabilities sum to 1 ($\sum P_i = 1$), and an additional degree of freedom is lost because $\lambda$ is estimated from the sample data.

### Category Pooling (Cochran's Criterion)
To maintain the mathematical validity of the continuous Chi-Square approximation:
1. Expected cell frequencies must satisfy $E_i \ge 5.0$.
2. Starting from discrete arrival values $0, 1, 2, \dots$, adjacent bins are merged sequentially from left to right until each bin satisfies $E_i \ge 5.0$.
3. Any residual right-tail category with $E_i < 5.0$ is combined into the preceding category.
4. If $k < 3$ ($df < 1$), the test is flagged as invalid due to insufficient observations.
5. Bins are established based solely on $\hat{\lambda}$ and $n$, never on observed frequencies.

### Decision Rule
- If $p\text{-value} > \alpha$ (default $\alpha = 0.05$): **Fail to reject $H_0$**. The Poisson model is reasonably consistent with the observed data.
- If $p\text{-value} \le \alpha$: **Reject $H_0$**. Statistically significant evidence that the distribution departs from Poisson.

---

## 9. Overload Probability Calculation
Given a service capacity $C$ (maximum student arrivals the canteen can handle in a 15-minute interval):

$$\text{Overload Event} \iff X > C$$

The cumulative probability of serving all arriving students without overload is:
$$P(X \le C) = \sum_{x=0}^C \frac{e^{-\lambda} \lambda^x}{x!}$$

The probability of overload is:
$$P(X > C) = 1 - P(X \le C) = 1 - \sum_{x=0}^C \frac{e^{-\lambda} \lambda^x}{x!}$$

In CanteenFlow, this calculation is implemented in two independent ways:
1. **Manual Implementation:** Log-gamma loop evaluation for step-by-step academic explanation:
   $$\ln P(X = x) = -\lambda + x \ln \lambda - \ln \Gamma(x + 1)$$
2. **SciPy Implementation:** High-precision survival function $\texttt{scipy.stats.poisson.sf}(C, \lambda)$.

Both methods are cross-validated in automated tests.

---

## 10. Risk Classification System
To translate raw probabilities into operational canteen decisions, each time period is mapped to a risk tier using **project-defined decision thresholds**:

| Risk Level | Decision Rule | Default Threshold | Operational Meaning |
| :---: | :---: | :---: | :--- |
| **LOW** 🟢 | $P(X > C) < \tau_{\mathrm{low}}$ | $< 20\%$ | Standard staffing; negligible queue buildup expected. |
| **MEDIUM** 🟡 | $\tau_{\mathrm{low}} \le P(X > C) \le \tau_{\mathrm{high}}$ | $20\% \le p \le 50\%$ | Moderate rush; prepare standby counter or pre-plated items. |
| **HIGH** 🔴 | $P(X > C) > \tau_{\mathrm{high}}$ | $> 50\%$ | Severe bottleneck likely; open extra service counter and assign staff. |

> **Academic Transparency:** These risk cutoffs are configurable decision criteria rather than universal physical constants. Users can adjust them in the dashboard sidebar.

---

## 11. Operational Dataset & Simulation
The primary dataset simulates real-world canteen lunch-service operations:
- **Operating Hours:** 12:30 PM to 2:30 PM (lunch period).
- **Time Interval:** 15 minutes (8 discrete intervals per day).
- **Duration:** 60 working days (Monday through Friday, 12 academic weeks).
- **Total Records:** 480 raw interval observations.
- **Reproducibility:** Seeded with `numpy.random.default_rng(42)`.

### Simulation Design
To emulate realistic collegiate behavior without hardcoding deterministic arrival peaks:
1. **Baseline Demand Curve $\lambda_0(t)$:** Background rate ($\lambda = 14.0$) plus Gaussian bell curve for the Lunch Rush centered at 13:15 (1:15 PM).
2. **Day-to-Day Variation:** Multiplicative daily shock drawn from $\mathrm{Gamma}(k, 1/k)$ with $5\%$ coefficient of variation.
3. **Weekday Multipliers:** Monday (1.02), Tuesday (1.00), Wednesday (1.01), Thursday (1.00), Friday (0.94).
4. **Stochastic Draws:** Each interval's count is drawn from $\mathrm{Poisson}(\lambda_0(t) \cdot f_{\mathrm{day}})$.
5. **Recording Gaps:** $0.5\%$ of counts are blanked as missing to validate data cleaning.

### Dataset Schema
Saved at `data/canteen_arrivals.csv`:
```csv
date,day,time_interval,hour,arrivals,period,capacity
2026-01-05,Monday,12:30-12:45,12,32,Lunch,30
2026-01-05,Monday,12:45-13:00,12,36,Lunch,30
...
```

---

## 12. System Architecture & Project Structure

### Architecture
```
┌────────────────────────────────────────────────────────┐
│               Streamlit Dashboard (app.py)             │
│   Overview | Arrival Analysis | Goodness of Fit | ...  │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│     analysis.py       │       │   visualization.py    │
│  Pipeline orchestration│      │   Plotly visualizations│
└───────────┬───────────┘       └───────────────────────┘
            │
    ┌───────┴───────────────────────┐
    ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   poisson_model.py    │       │  goodness_of_fit.py   │
│  λ, PMF, P(X>C), Risk │       │  Chi-Square & Pooling │
└───────────┬───────────┘       └───────────────────────┘
            │
    ┌───────┴───────────────────────┐
    ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│       utils.py        │       │   data_generator.py   │
│ Cleaning & Validation │       │ Data Generator/Sim    │
└───────────────────────┘       └───────────────────────┘
```

### Directory Structure
```
canteenflow/
├── app.py                  # Streamlit interactive dashboard
├── analysis.py             # Pipeline orchestration & statistical summarization
├── poisson_model.py        # Poisson PMF, CDF, overload probability, exact CI
├── goodness_of_fit.py      # Chi-Square test with Cochran category merging
├── data_generator.py       # Reproducible arrivals dataset generator
├── visualization.py        # Plotly chart renderers
├── utils.py                # Schema validation, constants, data cleaning
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Python dependency manifest
├── README.md               # Academic documentation & viva guide
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── config.toml         # Theme settings (dark mode, typography)
├── data/
│   └── canteen_arrivals.csv# Canteen arrivals dataset (12:30 PM - 2:30 PM)
└── tests/
    ├── test_poisson.py     # Tests for Poisson PMF, MLE, overload probability
    ├── test_data_generator.py # Tests for data reproducibility & schema
    └── test_analysis.py    # Tests for cleaning, pipeline, and edge cases
```

---

## 13. Installation & How to Run

### Prerequisites
- Python 3.10 or higher (tested on Python 3.14)
- Git

### Step 1: Clone or Navigate to Directory
```bash
cd /home/bad_apple/Probability_courseWork/canteenflow
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Regenerate Synthetic Dataset
```bash
# Default: 60 days, seed 42, capacity 30
python data_generator.py

# Custom options:
python data_generator.py --days 90 --seed 7 --capacity 35
```

### Step 5: Run Automated Tests
```bash
pytest -v
```

### Step 6: Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 14. Example Output & Numerical Results
Based on the 60-day operational dataset ($\mathrm{seed}=42$, capacity $C = 30$, operating hours 12:30 PM – 2:30 PM):

| Time Interval | $\hat{\lambda}$ | 95% Confidence Interval | Variance $s^2$ | $\mathrm{Var}/\mathrm{Mean}$ | $P(X > 30)$ | Risk Tier | $\chi^2$ Decision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12:30–12:45** | 23.71 | $[22.49, 24.99]$ | 22.48 | 0.948 | $8.58\%$ | **LOW** 🟢 | Fail to reject $H_0$ ($p=0.775$) |
| **12:45–13:00** | 29.23 | $[27.88, 30.63]$ | 40.22 | 1.376 | $39.61\%$ | **MEDIUM** 🟡 | Fail to reject $H_0$ ($p=0.717$) |
| **13:00–13:15** | 33.38 | $[31.94, 34.88]$ | 44.07 | 1.320 | $68.34\%$ | **HIGH** 🔴 | Fail to reject $H_0$ ($p=0.571$) |
| **13:15–13:30** | 34.58 | $[33.11, 36.10]$ | 32.42 | 0.937 | $75.17\%$ | **HIGH** 🔴 | Fail to reject $H_0$ ($p=0.673$) |
| **13:30–13:45** | 29.25 | $[27.89, 30.67]$ | 28.47 | 0.973 | $39.76\%$ | **MEDIUM** 🟡 | Fail to reject $H_0$ ($p=0.638$) |
| **13:45–14:00** | 24.17 | $[22.94, 25.44]$ | 19.77 | 0.818 | $10.20\%$ | **LOW** 🟢 | Fail to reject $H_0$ ($p=0.403$) |
| **14:00–14:15** | 19.25 | $[18.16, 20.39]$ | 21.89 | 1.137 | $0.83\%$ | **LOW** 🟢 | Fail to reject $H_0$ ($p=0.108$) |
| **14:15–14:30** | 15.97 | $[14.97, 17.01]$ | 15.22 | 0.953 | $0.05\%$ | **LOW** 🟢 | Fail to reject $H_0$ ($p=0.785$) |

### Overall Goodness-of-Fit Summary
- **Tested Periods:** 8 (all lunch intervals between 12:30 PM and 2:30 PM)
- **Null Hypothesis Rejections ($p \le 0.05$):** 0 out of 8 ($0.0\%$)
- **Expected False Positives under $H_0$:** $8 \times 0.05 = 0.4$
- **Binomial Test ($P(\ge 0 \text{ rejections} \mid H_0)$):** $p = 1.000$
- **Conclusion:** No rejections occur. The Poisson distribution is globally consistent across all 15-minute operating intervals.

---

## 15. Statistical Assumptions & Limitations

### Crucial Academic Limitations (for Viva Defense)
1. **Empirical Distribution Scope:** The operational dataset reflects lunch-hour arrival dynamics (12:30 PM to 2:30 PM) under standard semester conditions.
2. **Poisson Rate Stationarity:** A standard Poisson process assumes a constant arrival rate $\lambda$ throughout the observation window. Over wider windows (e.g., an entire hour), the arrival rate changes, leading to an overdispersed Poisson mixture. CanteenFlow addresses this by using small 15-minute intervals.
3. **Arrival Independence Assumption:** The Poisson model assumes student arrivals occur independently. In reality, college students often arrive in social groups (batches of 2–5). Group arrivals produce batch clustering and overdispersion ($s^2 > \bar{x}$).
4. **Capacity Interpretation:** Capacity $C$ is treated as an arrival count threshold per interval, abstracting away internal queue service dynamics (M/M/c queue transitions, meal preparation times).
5. **External Environmental Factors:** Unmodelled external shocks—such as exam schedules, sudden rainstorms, timetable alterations, or campus festivals—cause systematic shifts in student arrival distributions.
6. **Project-Defined Risk Cutoffs:** Risk categories (`LOW`, `MEDIUM`, `HIGH`) reflect practical operational guidelines rather than universal probabilistic laws.

---

## 16. Future Scope & Extensions
1. **Queuing Theory Integration (M/M/c Model):** Extend the arrival analysis with service-time modeling ($\mu$) to compute queue length $L_q$ and waiting time $W_q$.
2. **Negative Binomial & Compound Poisson Models:** Model overdispersed intervals by introducing a cluster/batch arrival distribution ($\text{Poisson-Logarithmic}$).
3. **Non-Homogeneous Poisson Process (NHPP):** Fit continuous rate functions $\lambda(t)$ using kernel smoothing or splines across the operating day.
4. **Real-Time IoT Integration:** Interface with overhead camera counts or RFID turnstiles to update $\hat{\lambda}$ continuously via Bayesian conjugate updates ($\mathrm{Gamma}(\alpha, \beta)$ priors).
5. **Dynamic Counter Scheduling:** Formulate an integer linear program (ILP) to recommend staff scheduling shifts that minimize counter operating cost subject to $P(\text{Overload}) \le 10\%$.

---

## 17. Academic Defense Q&A (Viva Preparation)

### Q1: Why is the Poisson distribution appropriate for this problem?
> **Answer:** Student arrivals represent count data (discrete non-negative integers $0, 1, 2, \dots$) occurring over a fixed, continuous interval of time ($15\text{ minutes}$). When student arrival events occur with an average rate and individual arrivals are largely independent, the Poisson distribution is the mathematically sound foundational model.

### Q2: How did you estimate $\lambda$?
> **Answer:** We used Maximum Likelihood Estimation (MLE). For a Poisson sample $x_1, \dots, x_n$, the log-likelihood is $\ln L(\lambda) = -n\lambda + \ln(\lambda)\sum x_i - \sum \ln(x_i!)$. Setting $\frac{d}{d\lambda}\ln L(\lambda) = 0$ yields $\hat{\lambda} = \frac{1}{n}\sum_{i=1}^n x_i$.

### Q3: Does $\mathrm{Mean} \approx \mathrm{Variance}$ prove that data is Poisson?
> **Answer:** No. While $\mathbb{E}[X] = \mathrm{Var}(X) = \lambda$ is a distinct mathematical property of the Poisson distribution, observing sample mean close to sample variance is necessary but insufficient. Other distributions can exhibit equidispersion. Stronger statistical proof requires a full goodness-of-fit test comparing observed frequencies against theoretical frequencies across all bins.

### Q4: Why combine categories in the Chi-Square Goodness-of-Fit test?
> **Answer:** The continuous Chi-Square distribution approximates the discrete multinomial statistic only when expected cell frequencies are adequately large ($E_i \ge 5$, Cochran's rule). Without combining sparse tails, very small expected counts in the denominator inflate the $\frac{(O - E)^2}{E}$ statistic artificially, causing high Type I error rates.

### Q5: How do degrees of freedom work in your test?
> **Answer:** For $k$ categories, standard degrees of freedom are $k - 1$ because cell probabilities sum to 1. Since the rate parameter $\lambda$ is not known a priori and is estimated from the sample data via $\hat{\lambda}$, we subtract an additional degree of freedom: $df = k - 1 - 1 = k - 2$.
