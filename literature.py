"""
literature.py
-------------
Literature survey and research gaps used by the dashboard's "Literature Review"
and "Research Gaps" pages and mirrored in README.md.

The first ten entries were checked against OpenAlex / the publisher DOI and
summarised from their published abstracts. The eight added for the second
review (forecasting, gaps G7-G11) were read in full from the PDFs; their
summaries cite specific sections. Where a paper's aim differs from ours, the
"leaves open" field describes what lies outside that aim rather than claiming
the authors made an error.
"""

from __future__ import annotations

PAPERS = [
    {
        "key": "tang2019a", "cite": "Tang et al. (2019a)", "year": 2019, "category": "Canteen pedestrian flow",
        "base": True,
        "reference": "Tang, T.-Q., Zhang, B.-T., Zhang, J., & Wang, T. (2019). Statistical analysis and modeling of "
                     "pedestrian flow in university canteen during peak period. Physica A: Statistical Mechanics and "
                     "its Applications, 521, 29–40.",
        "doi": "10.1016/j.physa.2019.01.060",
        "focus": "Statistical features of pedestrian flow in the second canteen of Beihang University during the "
                 "lunch peak.",
        "method": "Camera-based data collection, statistical analysis of the flow, a pedestrian-flow model, and "
                  "validation of simulations against observations.",
        "finding": "The proposed model reproduces the traffic phenomena observed in the canteen during the peak "
                   "period and can support administrators' planning.",
        "relevance": "Base paper. Establishes that canteen arrivals can be observed in short time segments and "
                     "analysed statistically; our project isolates the arrival-count component.",
        "leaves_open": "A compact per-interval probability of exceeding an arrival threshold, with a low / medium / "
                       "high label, is not the paper's output. The arrival rate is an input to its simulation "
                       "(Section 3), so it is never predicted for a future day; the paper also notes that days "
                       "differ with the timetable, so averaging all days is unsuitable (Section 2), and that "
                       "inflow is Poisson within short periods with a rate that changes every few minutes, "
                       "stated rather than tested.",
    },
    {
        "key": "tang2019b", "cite": "Tang et al. (2019b)", "year": 2019, "category": "Canteen pedestrian flow",
        "reference": "Tang, T.-Q., Zhang, B.-T., & Xie, C.-Z. (2019). Modeling and simulation of pedestrian flow in "
                     "university canteen. Simulation Modelling Practice and Theory, 95, 96–111.",
        "doi": "10.1016/j.simpat.2019.04.011",
        "focus": "Each pedestrian's motion inside a university canteen.",
        "method": "Extended cellular-automaton (CA) model, calibrated with data from the same Beihang canteen.",
        "finding": "The CA model reproduces individual motion with high calibration accuracy and is used to test "
                   "safety and efficiency under different situations.",
        "relevance": "Shows how detailed (and data-hungry) full canteen models are, motivating a lighter "
                     "arrival-only indicator.",
        "leaves_open": "Arrival intensity is an input to the simulation; testing the arrival distribution per "
                       "interval is outside its scope.",
    },
    {
        "key": "wang2011", "cite": "Wang (2011)", "year": 2011, "category": "Canteen queueing",
        "reference": "Wang, H. (2011). Optimal management model of university canteen based on mixed queuing theory. "
                     "Journal of Huangshi Institute of Technology (CNKI: HSGD201103013).",
        "doi": None,
        "focus": "Number of serving windows, seats and opening times for a university canteen.",
        "method": "Multi-server queueing model M/M/n/m built from an on-site investigation.",
        "finding": "Determines the effective number of windows, matching seat count, window opening time and peak "
                   "capacity from customers' maximum tolerance.",
        "relevance": "Typical queueing treatment of a canteen; the 'M' for arrivals means Poisson arrivals are "
                     "assumed.",
        "leaves_open": "The Poisson arrival assumption is a model input rather than something tested slot by slot.",
    },
    {
        "key": "yuan2023", "cite": "Yuan et al. (2023)", "year": 2023, "category": "Canteen queueing",
        "reference": "Yuan, N., Wang, X., & Chen, X. (2023). Analysis on the service capacity of university canteens "
                     "under the epidemic prevention based on queue theory. Transactions on Comparative Education, "
                     "5(7).",
        "doi": "10.23977/trance.2023.050706",
        "focus": "Canteen congestion under staggered (grade-wise) dining times during epidemic control.",
        "method": "Questionnaire survey of service and dining times, queueing theory and MATLAB simulation of two "
                  "operating modes.",
        "finding": "Gives recommendations on how many windows to open under each dining mode.",
        "relevance": "Confirms that canteen crowding is managed by changing timing; our least-crowded-slot "
                     "analysis targets the same lever from the student side.",
        "leaves_open": "Arrival timing comes from surveys and simulation; a statistical test of when crowds are "
                       "smallest is not reported.",
    },
    {
        "key": "brown2005", "cite": "Brown et al. (2005)", "year": 2005, "category": "Arrival-process statistics",
        "reference": "Brown, L., Gans, N., Mandelbaum, A., Sakov, A., Shen, H., Zeltyn, S., & Zhao, L. (2005). "
                     "Statistical analysis of a telephone call center: A queueing-science perspective. Journal of "
                     "the American Statistical Association, 100(469), 36–50.",
        "doi": "10.1198/016214504000001808",
        "focus": "A full year of call-by-call records from a banking call centre.",
        "method": "Decomposes service into arrivals, patience and service time; develops a test that a point "
                  "process is Poisson and models for forecasting Poisson arrival rates.",
        "finding": "Each component needs its own statistical treatment; arrivals are modelled as Poisson with "
                   "rates that vary over the day.",
        "relevance": "Methodological model for our work: treat arrivals as their own component, test the Poisson "
                     "assumption, and estimate time-varying rates.",
        "leaves_open": "Set in a call centre; the approach has not been carried over to campus canteens.",
    },
    {
        "key": "green2007", "cite": "Green et al. (2007)", "year": 2007, "category": "Time-varying demand",
        "reference": "Green, L. V., Kolesar, P. J., & Whitt, W. (2007). Coping with time-varying demand when setting "
                     "staffing requirements for a service system. Production and Operations Management, 16(1), "
                     "13–39.",
        "doi": "10.1111/j.1937-5956.2007.tb00164.x",
        "focus": "Staffing service systems whose demand varies in a predictable daily pattern.",
        "method": "Review of period-by-period, peak-hour, time-lag and infinite-server approximations.",
        "finding": "Stationary peak-hour or period-by-period methods need refinement when demand changes within "
                   "the day; time-dependent behaviour must be captured.",
        "relevance": "Justifies estimating a separate λ for each short slot instead of one rate for the whole "
                     "lunch period.",
        "leaves_open": "Focuses on staffing levels; a student-facing probability of crowding is not its aim.",
    },
    {
        "key": "kim2014", "cite": "Kim & Whitt (2014)", "year": 2014, "category": "Arrival-process statistics",
        "reference": "Kim, S.-H., & Whitt, W. (2014). Are call center and hospital arrivals well modeled by "
                     "nonhomogeneous Poisson processes? Manufacturing & Service Operations Management, 16(3), "
                     "464–480.",
        "doi": "10.1287/msom.2014.0490",
        "focus": "Whether time-varying arrivals in call centres and emergency departments are Poisson.",
        "method": "Kolmogorov–Smirnov tests on piecewise-constant-rate subintervals using the conditional-uniform "
                  "property.",
        "finding": "Data are consistent with a nonhomogeneous Poisson process only when analysed carefully; naive "
                   "tests fail because of data rounding, subintervals where the rate varies too much, and "
                   "overdispersion from pooling days with different rates.",
        "relevance": "Directly supports our design: short slots, a per-slot Poisson test, and a dispersion check; "
                     "it also explains why pooled hourly data can look non-Poisson.",
        "leaves_open": "Studies call centres and hospitals; canteen arrivals are not examined.",
    },
    {
        "key": "leemis1991", "cite": "Leemis (1991)", "year": 1991, "category": "Time-varying demand",
        "reference": "Leemis, L. M. (1991). Nonparametric estimation of the cumulative intensity function for a "
                     "nonhomogeneous Poisson process. Management Science, 37(7), 886–900.",
        "doi": "10.1287/mnsc.37.7.886",
        "focus": "Estimating a time-varying Poisson rate from one or more observed realisations.",
        "method": "Nonparametric, parameter-free estimator of the cumulative intensity; supports Monte Carlo "
                  "generation by inversion.",
        "finding": "A usable intensity estimate can be built from repeated realisations without choosing "
                   "arbitrary parameters.",
        "relevance": "Our per-slot λ̂ from m days is the piecewise-constant version of this idea; it also "
                     "grounds the Monte Carlo check and the smooth-rate future extension.",
        "leaves_open": "Estimates the rate only; turning it into an exceedance risk is left to the application.",
    },
    {
        "key": "cameron1990", "cite": "Cameron & Trivedi (1990)", "year": 1990, "category": "Model checking",
        "reference": "Cameron, A. C., & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the "
                     "Poisson model. Journal of Econometrics, 46(3), 347–364.",
        "doi": "10.1016/0304-4076(90)90014-K",
        "focus": "Testing the Poisson property that variance equals mean.",
        "method": "Simple regression-based tests of equidispersion against overdispersion alternatives.",
        "finding": "Overdispersion can be tested with simple auxiliary regressions; ignoring it misstates "
                   "uncertainty.",
        "relevance": "Supports checking Var ≈ Mean before using Poisson probabilities; our index-of-dispersion "
                     "test is the no-covariate counterpart.",
        "leaves_open": "General count-data econometrics; no application to arrival counts.",
    },
    {
        "key": "variskhanov2026", "cite": "Variskhanov & Arreeras (2026)", "year": 2026,
        "category": "Campus crowd sensing",
        "reference": "Variskhanov, A., & Arreeras, T. (2026). Early peak badges from Wi-Fi telemetry: A field "
                     "feasibility study of lunchtime crowd management on a smart campus. Urban Science, 10(1), 29.",
        "doi": "10.3390/urbansci10010029",
        "focus": "Early warning of lunchtime crowding in five university canteens.",
        "method": "7-minute Wi-Fi access-point profiles (11:00–14:00), k-means++ demand-shape typologies, and a "
                  "cosine-similarity rule that issues an 'Early Peak' badge at 11:28.",
        "finding": "Early Peak days were identified with 0.86 agreement on held-out days, and badge days showed "
                   "about 9% lower peak crowding; presented as a 3-week feasibility study.",
        "relevance": "Shows real demand for simple, interpretable lunchtime crowd signals that support day-to-day "
                     "decisions, the same need our risk labels address.",
        "leaves_open": "Needs campus Wi-Fi telemetry and returns a day-level badge, not a probability per time slot.",
    },
    # --- Added for the second review: forecasting arrivals and checking forecasts (gaps G7-G11) ---
    {
        "key": "jiao2022", "cite": "Jiao et al. (2022)", "year": 2022, "category": "Canteen queueing",
        "reference": "Jiao, S., Wang, X., Song, D., & Du, S. (2022). Simulation and optimization of university "
                     "canteen service under the situation of epidemic prevention and control based on queuing "
                     "theory. ITM Web of Conferences, 47, 02052.",
        "doi": "10.1051/itmconf/20224702052",
        "focus": "Number of serving windows a university canteen should open under epidemic-control limits.",
        "method": "M/M/C queue with exponential inter-arrival and service times, simulated 2000 times from random "
                  "number sequences; one arrival rate of 19.3 students/min and service rate 3.1/min.",
        "finding": "Seven windows balance waiting time and window busy rate (82.3%); four to six windows give "
                   "waits of up to 23 minutes.",
        "relevance": "A second, recent canteen study in which Poisson arrivals are the modelling starting point.",
        "leaves_open": "One constant arrival rate for the whole period, assumed exponential rather than tested; "
                       "the authors themselves list period-specific rates as the next step (Section 4.3).",
    },
    {
        "key": "shah2025", "cite": "Shah et al. (2025)", "year": 2025, "category": "Canteen demand forecasting",
        "reference": "Shah, H., Myller, N., & Islas Sedano, C. (2025). Forecasting daily customer flow in "
                     "restaurants: a multifactor machine learning approach. Applied Computing and Intelligence, "
                     "5(2), 168–190.",
        "doi": "10.3934/aci.2025011",
        "focus": "Daily customer numbers in a university self-service restaurant (Flavoria, University of Turku).",
        "method": "RFID tray data 2019–2024; linear regression, random forest, XGBoost and LSTM over ten feature "
                  "sets (calendar, weather, holidays, menu, previous-day count, smoothing).",
        "finding": "XGBoost has the lowest error (mean MAE 86 customers/day); the previous day's count, "
                   "exponential smoothing and holidays are the strongest predictors (Table 6).",
        "relevance": "The closest campus-dining forecasting study; confirms that calendar and holiday effects matter.",
        "leaves_open": "Forecasts daily totals only, although its own Figure 3 shows 48% of customers arrive in "
                       "one hour; it gives no time-slot forecast, no probability of overload and no Poisson model.",
    },
    {
        "key": "weinberg2007", "cite": "Weinberg et al. (2007)", "year": 2007, "category": "Arrival forecasting",
        "reference": "Weinberg, J., Brown, L. D., & Stroud, J. R. (2007). Bayesian forecasting of an "
                     "inhomogeneous Poisson process with applications to call center data. Journal of the "
                     "American Statistical Association, 102(480), 1185–1199.",
        "doi": None,
        "focus": "One-day-ahead forecasts of 5-minute arrival rates to a bank call centre, updated during the day.",
        "method": "Multiplicative Poisson model: rate = daily level × within-day shape per weekday, AR(1) daily "
                  "level, Bayesian MCMC; within-day learning at 10 am and 12 pm.",
        "finding": "Median RMSE 13–21% below seasonal regressions; 95% intervals cover 94.7%; PIT histograms are "
                   "uniform, so the forecast distributions are calibrated (Section 4.2).",
        "relevance": "The template for forecasting a Poisson rate per slot, updating it during the day, and "
                     "checking calibration with coverage and PIT.",
        "leaves_open": "Relies on a square-root normal approximation valid for large counts (~300 per slot); "
                       "canteen slot counts are about 15–35.",
    },
    {
        "key": "shen2008", "cite": "Shen & Huang (2008)", "year": 2008, "category": "Arrival forecasting",
        "reference": "Shen, H., & Huang, J. Z. (2008). Forecasting time series of inhomogeneous Poisson processes "
                     "with application to call center workforce management. The Annals of Applied Statistics, "
                     "2(2), 601–623.",
        "doi": "10.1214/08-AOAS164",
        "focus": "Forecasting the quarter-hour Poisson rate profile of the next day and updating it within the day.",
        "method": "Poisson factor model fitted by alternating Poisson regressions, AR(1) forecasts of factor "
                  "scores with weekday intercepts, bootstrap distributional forecasts, penalised intraday update.",
        "finding": "Intraday updating improved staffing accuracy by 14.5% and narrowed prediction intervals by "
                   "41.5% (Section 4.4); a simple ratio update (observed / forecast so far) also helps.",
        "relevance": "Shows that the counts already seen today carry information about the rest of the day.",
        "leaves_open": "Built for large call-centre volumes and staffing levels, not for a probability that a "
                       "canteen slot overflows.",
    },
    {
        "key": "aldornoiman2009", "cite": "Aldor-Noiman et al. (2009)", "year": 2009, "category": "Arrival forecasting",
        "reference": "Aldor-Noiman, S., Feigin, P. D., & Mandelbaum, A. (2009). Workload forecasting for a call "
                     "center: Methodology and a case study. The Annals of Applied Statistics, 3(4), 1403–1447.",
        "doi": "10.1214/09-AOAS255",
        "focus": "Half-hourly arrival forecasts for staffing, including special-event effects.",
        "method": "Poisson log-linear model of daily counts with weekday and billing-cycle indicators (Table 1), "
                  "then a mixed model with AR(1) day and period effects; compared with the 'industry model' that "
                  "averages past same-weekday slots.",
        "finding": "Special days are significant predictors; holidays and outliers are left out of fitting but "
                   "kept for evaluation; 15- and 30-minute slots forecast almost equally well (Section 4.4).",
        "relevance": "Directly motivates weekday and exam-day terms; its 'industry model' is the per-slot "
                     "average this project used at the first review.",
        "leaves_open": "Square-root approximation needs about 500 arrivals per period; no overload probability.",
    },
    {
        "key": "matteson2011", "cite": "Matteson et al. (2011)", "year": 2011, "category": "Arrival forecasting",
        "reference": "Matteson, D. S., McLean, M. W., Woodard, D. B., & Henderson, S. G. (2011). Forecasting "
                     "emergency medical service call arrival rates. The Annals of Applied Statistics, 5(2B), "
                     "1379–1406.",
        "doi": "10.1214/10-AOAS442",
        "focus": "Hourly ambulance-call forecasts for Toronto EMS, where counts are small (about 24 per hour).",
        "method": "Poisson counts modelled directly (no normal approximation), latent factor model with "
                  "weekday constraints and smoothing splines, integer-GARCH adjustment for recent deviations.",
        "finding": "Beats the practitioners' 'average of four past periods' on held-out data; Poisson deviance "
                   "and Anscombe residuals used to compare models (Table 1).",
        "relevance": "Supports modelling small counts with Poisson probabilities and testing against a simple "
                     "averaging baseline on held-out days.",
        "leaves_open": "Forecasts counts for staffing; no overload probability or peak-time probability.",
    },
    {
        "key": "ibrahim2016", "cite": "Ibrahim et al. (2016)", "year": 2016, "category": "Arrival forecasting",
        "reference": "Ibrahim, R., Ye, H., L'Ecuyer, P., & Shen, H. (2016). Modeling and forecasting call center "
                     "arrivals: A literature survey and a case study. International Journal of Forecasting, "
                     "32(3), 865–874.",
        "doi": "10.1016/j.ijforecast.2015.11.012",
        "focus": "Survey of arrival-process properties and forecasting models.",
        "method": "Literature survey plus a case study comparing four models on 200 days of Canadian data.",
        "finding": "Arrivals are overdispersed and correlated within and between days; a Gamma 'busyness factor' "
                   "gives Var(X) = λ(1 + (1 + λ)Var W) (Section 3.3); fixed-effect models are hard to beat for "
                   "long horizons (Section 3.2).",
        "relevance": "The basis for the Poisson–Gamma model and the same-day update used here.",
        "leaves_open": "Call-centre focus; the operational value of better forecasts is listed as open.",
    },
    {
        "key": "mcnulty1988", "cite": "McNulty (1988)", "year": 1988, "category": "Forecast verification",
        "reference": "McNulty, R. P. (1988). A new measure of forecast skill using the Brier score. Central Region "
                     "Technical Attachment 88-26, National Weather Service, NOAA.",
        "doi": None,
        "focus": "Scoring probability forecasts of a yes/no event (rain).",
        "method": "Brier score B = (1/N) Σ (pᵢ − oᵢ)² (Brier, 1950) and a skill score S = (R − F) / R against a "
                  "no-skill reference forecast.",
        "finding": "Skill must be judged against a reference forecast; a constant forecast at the observed "
                   "frequency is the natural no-skill reference.",
        "relevance": "Overload in a slot is a yes/no event, so P(X > C) can be scored the same way, with the "
                     "per-slot average as the reference.",
        "leaves_open": "A technical note, not a peer-reviewed paper; it reports the method, not canteen data.",
    },
]

METHOD_REFERENCES = [
    "Garwood, F. (1936). Fiducial limits for the Poisson distribution. Biometrika, 28(3/4), 437–442. "
    "https://doi.org/10.1093/biomet/28.3-4.437",
    "Cochran, W. G. (1954). Some methods for strengthening the common χ² tests. Biometrics, 10(4), 417–451. "
    "https://doi.org/10.2307/3001616",
    "Przyborowski, J., & Wileński, H. (1940). Homogeneity of results in testing samples from Poisson series. "
    "Biometrika, 31(3/4), 313–323. https://doi.org/10.1093/biomet/31.3-4.313",
    "Ross, S. M. (2014). Introduction to Probability Models (11th ed.). Academic Press.",
    "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, "
    "78(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2",
    "Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. Journal of the American "
    "Statistical Association, 78(383), 605–610.",
]

# Research gaps. G1 is the gap stated at the first review; G2-G6 come from the extended survey;
# G7-G11 come from the second-review reading on forecasting (all map to the PAMAT501 syllabus).
GAPS = [
    {
        "id": "G1", "title": "Focus and application gap",
        "summary": "The reviewed canteen studies model the whole system rather than a simple per-slot overload risk.",
        "explain": "Existing canteen work reproduces the complete meal process: movement, service windows, "
                   "seating and congestion. That is valuable but heavy, and its outputs are simulated flows or "
                   "queue sizes. A manager or student asking \"how likely is it to get crowded at 13:15?\" gets "
                   "no direct answer. The gap is not that Poisson modelling is new; it is that the reviewed studies do not "
                   "convert time-specific arrival rates into a threshold-exceedance probability with a risk label.",
        "evidence": ["tang2019a", "tang2019b", "wang2011", "yuan2023"],
        "addressed": "Estimate λ̂ₜ for every short slot, compute P(Xₜ ≥ K), and label each slot Low, Medium or High.",
        "page": "overload",
    },
    {
        "id": "G2", "title": "Validation gap",
        "summary": "The Poisson arrival assumption is usually built in, not tested per time slot.",
        "explain": "Queueing models of canteens start from Markovian (Poisson) arrivals, so the assumption is an "
                   "input rather than a finding. Kim & Whitt (2014) show that whether arrivals are Poisson has to "
                   "be tested, and that careless testing gives misleading answers. Without a test, every "
                   "probability built on the model inherits an unchecked assumption. Testing many slots also "
                   "creates a multiple-testing problem: a few rejections happen by chance alone.",
        "evidence": ["wang2011", "yuan2023", "kim2014", "brown2005", "cameron1990"],
        "addressed": "For every slot: chi-square goodness of fit with pooled categories, an index-of-dispersion "
                     "test, and a binomial check of how many rejections chance alone would produce.",
        "page": "model",
    },
    {
        "id": "G3", "title": "Time-resolution gap",
        "summary": "Hourly or whole-peak averages hide the rush; the rate changes within the lunch period.",
        "explain": "Lunch demand rises and falls within minutes. Averaging over an hour or the whole peak treats "
                   "a changing rate as constant, which understates the busiest minutes and makes pooled counts "
                   "overdispersed. Green et al. (2007) show that peak-hour and stationary approximations need "
                   "refinement under time-varying demand, and Kim & Whitt (2014) trace failed Poisson tests to "
                   "subintervals where the rate varies too much.",
        "evidence": ["green2007", "kim2014", "leemis1991"],
        "addressed": "Piecewise-constant rates on 5-, 10- or 15-minute slots, with an hourly view that "
                     "demonstrates the overdispersion caused by pooling.",
        "page": "arrivals",
    },
    {
        "id": "G4", "title": "Uncertainty gap",
        "summary": "Risk is reported from a point estimate of λ, ignoring sampling error.",
        "explain": "λ̂ is estimated from a limited number of days, so it carries sampling error. A slot whose "
                   "overload probability sits near a threshold can switch between Medium and High within the "
                   "plausible range of λ. The reviewed canteen studies aim to calibrate and reproduce flows; "
                   "carrying the uncertainty in the estimated arrival rate through to a reported risk is not "
                   "part of their output.",
        "evidence": ["tang2019a", "brown2005"],
        "addressed": "Exact Garwood confidence intervals for λ, P(X ≥ K) evaluated at both ends of the interval, "
                     "and a Monte Carlo check of the formula.",
        "page": "overload",
    },
    {
        "id": "G5", "title": "Decision-support gap",
        "summary": "Results are not turned into simple, interpretable guidance for students and managers.",
        "explain": "Variskhanov & Arreeras (2026) note that many smart-campus tools still fail to support routine, "
                   "day-to-day decisions, and they build a simple crowd signal for that reason. Their signal "
                   "needs Wi-Fi telemetry and gives a day-level badge. Queueing studies give window counts for "
                   "managers but tell students nothing about when to go. What is missing is guidance that needs "
                   "only arrival counts: when is crowding least likely, and is that slot reliably quieter?",
        "evidence": ["variskhanov2026", "yuan2023", "wang2011"],
        "addressed": "An interactive dashboard with the least-crowded slot, an exact test that it is genuinely "
                     "quieter than the runner-up, a visit planner, and capacity recommendations.",
        "page": "plan",
    },
    {
        "id": "G6", "title": "Data and context gap",
        "summary": "Evidence comes from specific campuses; local arrival data are not available.",
        "explain": "The canteen studies reviewed use data from particular universities, mainly Beihang University "
                   "in China, and arrival patterns depend on local timetables and break times. We did not find "
                   "openly available interval-level arrival counts for an Indian college canteen, so results "
                   "from elsewhere cannot simply be reused here.",
        "evidence": ["tang2019a", "tang2019b", "variskhanov2026"],
        "addressed": "A documented collection protocol (fixed 5- or 10-minute slots across m days), a validated "
                     "CSV import for field counts, and a clearly labelled synthetic dataset for the prototype.",
        "page": "arrivals",
    },
    {
        "id": "G7", "title": "Prediction gap",
        "summary": "Canteen studies describe past days; none predicts a specific future day's slot rates.",
        "explain": "The first-review method averages each slot over past days, which Aldor-Noiman et al. (2009) "
                   "call the 'industry model'. Tang et al. (2019a) note that days differ with the timetable, so "
                   "a single average is unsuitable, and their arrival rate is an input, never a prediction. "
                   "Shah et al. (2025) forecast campus dining but only daily totals. Weekday and special-day "
                   "effects are standard in arrival forecasting (Weinberg et al., 2007; Aldor-Noiman et al., 2009).",
        "evidence": ["tang2019a", "shah2025", "aldornoiman2009", "weinberg2007"],
        "addressed": "Multiple linear regression of log counts on slot, weekday and exam-day dummies (Module 3), "
                     "with t-tests of each effect and ANOVA / Kruskal-Wallis / t / Mann-Whitney tests of the day "
                     "effects (Module 5), giving a predicted λ and P(X > C) for every slot of a chosen day.",
        "page": "forecast",
    },
    {
        "id": "G8", "title": "Peak-time uncertainty gap",
        "summary": "The busiest slot is reported, but not how likely it is to be the peak on a given day.",
        "explain": "Two neighbouring slots can have similar rates, so on any one day either may turn out busiest. "
                   "Forecasting papers produce full predictive distributions by simulation (Weinberg et al., "
                   "2007; Shen & Huang, 2008), but none turns them into a probability that each slot is the "
                   "day's peak, or that at least one slot overflows.",
        "evidence": ["weinberg2007", "shen2008", "tang2019a"],
        "addressed": "Monte Carlo simulation of whole lunch periods from the predicted Poisson rates (Modules 1-2): "
                     "P(slot t is the peak), the most likely peak window, P(at least one overload).",
        "page": "forecast",
    },
    {
        "id": "G9", "title": "Same-day updating gap",
        "summary": "Forecasts are fixed in the morning, although early counts reveal how busy the day is.",
        "explain": "Weinberg et al. (2007) and Shen & Huang (2008) show that counts observed early in the day "
                   "improve forecasts for the rest of it. Their methods rely on large-count approximations; "
                   "Matteson et al. (2011) warn these fail for small counts like a canteen's. Ibrahim et al. "
                   "(2016) describe the Gamma 'busyness factor' that links the slots of one day.",
        "evidence": ["weinberg2007", "shen2008", "matteson2011", "ibrahim2016"],
        "addressed": "Bayes theorem with a Gamma prior on the day's busyness factor and Poisson counts: the "
                     "posterior is Gamma, and the updated P(X > C) for the remaining slots is exact (Modules 1-2).",
        "page": "forecast",
    },
    {
        "id": "G10", "title": "Overdispersion gap",
        "summary": "Overdispersion is detected but the overload probability still uses the plain Poisson tail.",
        "explain": "Arrival counts are often more variable than Poisson because whole days are busier or quieter "
                   "(Ibrahim et al., 2016; Aldor-Noiman et al., 2009). The first review tested for this but did "
                   "not change P(X > C) when it was present.",
        "evidence": ["ibrahim2016", "aldornoiman2009", "cameron1990"],
        "addressed": "Poisson–Gamma mixture from the theorem of total probability, Var(X) = λ + vλ², with v "
                     "estimated by moments and a Pearson chi-square test of the plain Poisson model "
                     "(Modules 1, 2, 5).",
        "page": "check",
    },
    {
        "id": "G11", "title": "Forecast validation gap",
        "summary": "Canteen results are checked on the same days used to estimate them, never on new days.",
        "explain": "Every forecasting paper read for this review judges a method on held-out days: rolling "
                   "one-day-ahead forecasts, interval coverage and PIT calibration (Weinberg et al., 2007), "
                   "lead times (Aldor-Noiman et al., 2009), train/test years (Matteson et al., 2011). McNulty "
                   "(1988) adds the Brier skill score for probability forecasts against a no-skill reference.",
        "evidence": ["weinberg2007", "aldornoiman2009", "matteson2011", "mcnulty1988", "jiao2022"],
        "addressed": "Rolling one-day-ahead backtest: Brier score and skill score, coverage and peak hit-rate "
                     "z-tests (Module 4), paired t and sign tests, Kolmogorov-Smirnov and chi-square tests of "
                     "the PIT (Module 5), predicted-vs-actual correlation (Module 3).",
        "page": "check",
    },
]


def paper(key: str) -> dict:
    return next(p for p in PAPERS if p["key"] == key)


# ---------------------------------------------------------------------------
# PAMAT501 syllabus map: every statistical component of the project and the module it uses
# ---------------------------------------------------------------------------
SYLLABUS_MODULES = {
    "M1": "Foundations of Probability and Random Variables",
    "M2": "Probability Distributions",
    "M3": "Correlation and Regression Analysis",
    "M4": "Statistical Inference: Large Samples",
    "M5": "Statistical Inference: Small Samples and Non-parametric Tests",
}

SYLLABUS_MAP = [
    # (component, syllabus topics, modules, where in the project, gap)
    ("Poisson model of slot arrivals, PMF and CDF", "Discrete random variables, PMF, CDF; Poisson distribution",
     ["M1", "M2"], "Methodology, Overload risk", "G1"),
    ("Overload probability P(X > C) = 1 − P(X ≤ C)", "Complement rule, CDF; Poisson distribution",
     ["M1", "M2"], "Overload risk", "G1"),
    ("Estimating λ by the sample mean (maximum likelihood)", "Estimation of parameters",
     ["M4"], "Arrival pattern", "G4"),
    ("Exact confidence interval for λ", "Estimation of parameters; chi-square and Gamma distributions",
     ["M2", "M4"], "Arrival pattern", "G4"),
    ("Index-of-dispersion test (Var = Mean)", "Chi-square test for variance", ["M5"], "Model check", "G2"),
    ("Chi-square goodness of fit of the Poisson model", "Chi-square goodness of fit", ["M5"], "Model check", "G2"),
    ("Least-crowded slot vs runner-up", "Conditional distribution; Binomial distribution; hypothesis test",
     ["M1", "M2", "M5"], "Arrival pattern, When to go", "G5"),
    ("Monte Carlo check of P(X > C)", "Random variables; independence; Poisson distribution",
     ["M1", "M2"], "Model check", "G4"),
    ("Day-specific rates: regression on slot, weekday, exam day", "Multiple linear regression", ["M3"],
     "Peak forecast", "G7"),
    ("t-tests of the regression coefficients", "t-test", ["M5"], "Peak forecast", "G7"),
    ("Weekday effect on daily arrivals", "One-way ANOVA; Kruskal-Wallis test", ["M5"], "Peak forecast", "G7"),
    ("Exam-day effect on daily arrivals", "Two-sample t-test; Mann-Whitney U test", ["M5"],
     "Peak forecast", "G7"),
    ("P(slot t is the day's peak), P(any overload)", "Joint distributions; independence; Gamma and Poisson",
     ["M1", "M2"], "Peak forecast", "G8"),
    ("Same-day update of the busyness factor", "Bayes theorem; conditional probability; Gamma and Poisson",
     ["M1", "M2"], "Peak forecast", "G9"),
    ("Poisson–Gamma (negative binomial) overload probability", "Theorem of total probability; moments; "
     "Gamma and Poisson", ["M1", "M2"], "Forecast check", "G10"),
    ("Pearson chi-square test of overdispersion", "Chi-square test", ["M5"], "Forecast check", "G10"),
    ("Coverage of 95% prediction intervals", "Large-sample test of a single proportion", ["M4"],
     "Forecast check", "G11"),
    ("Peak-slot hit rate against chance", "Large-sample test of a single proportion", ["M4"],
     "Forecast check", "G11"),
    ("Model vs per-slot average, day by day", "Paired t-test; sign test", ["M5"], "Forecast check", "G11"),
    ("Calibration of forecast distributions (PIT)", "Kolmogorov-Smirnov test; chi-square goodness of fit", ["M5"],
     "Forecast check", "G11"),
    ("Predicted vs actual counts", "Correlation; rank correlation", ["M3"], "Forecast check", "G11"),
    ("Brier score and skill score of P(X > C)", "Mathematical expectation (mean squared error)", ["M3"],
     "Forecast check", "G11"),
    ("Carry-over between consecutive slots", "Rank correlation (Spearman)", ["M3"], "Forecast check", "G9"),
    ("Day-to-day dependence of busy days", "Run test", ["M5"], "Forecast check", "G11"),
]

