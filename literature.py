"""
literature.py
-------------
Literature survey and research gaps used by the dashboard's "Literature Review"
and "Research Gaps" pages and mirrored in README.md.

Every entry was checked against OpenAlex / the publisher DOI. Summaries are
based on each paper's published abstract; where a paper's aim differs from
ours, the "leaves open" field describes what lies outside that aim rather than
claiming the authors made an error.
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
                       "high label, is not the paper's output.",
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
]

METHOD_REFERENCES = [
    "Garwood, F. (1936). Fiducial limits for the Poisson distribution. Biometrika, 28(3/4), 437–442. "
    "https://doi.org/10.1093/biomet/28.3-4.437",
    "Cochran, W. G. (1954). Some methods for strengthening the common χ² tests. Biometrics, 10(4), 417–451. "
    "https://doi.org/10.2307/3001616",
    "Przyborowski, J., & Wileński, H. (1940). Homogeneity of results in testing samples from Poisson series. "
    "Biometrika, 31(3/4), 313–323. https://doi.org/10.1093/biomet/31.3-4.313",
    "Ross, S. M. (2014). Introduction to Probability Models (11th ed.). Academic Press.",
]

# Research gaps. G1 is the gap stated at the first review; G2-G6 come from the extended survey.
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
        "page": "fit",
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
        "page": "rate",
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
        "page": "quietest",
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
        "page": "data",
    },
]


def paper(key: str) -> dict:
    return next(p for p in PAPERS if p["key"] == key)
