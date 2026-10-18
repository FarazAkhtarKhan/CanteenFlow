"""
ui.py
-----
Shared look and layout helpers for the CanteenFlow dashboard pages.

`app.py` fills `CTX` (data, parameters, pipeline result, selected slot and the
page objects) once per run; every page in report_pages.py reads it through `ctx()`.

Every page follows one template: a question as the title, the answer directly
below it (`page_header`), then charts with a one-line caption and an optional
"About this chart" popover for details (`chart_block`).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import streamlit as st

PLOT_CONFIG = {"displayModeBar": False}
RISK_WORD = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}

CSS = """
<style>
.block-container { max-width: 1200px; padding-top: 2.2rem; padding-bottom: 5rem; }
h1, h2, h3, h4 { color: #1B2A4A; }

/* Page header: question as title, answer underneath */
.page-kicker { color: #00868B; font-weight: 600; font-size: .92rem; margin: 0 0 .2rem; }
.answer-box { background: #1B2A4A; color: #FFFFFF; border-radius: 6px; padding: .9rem 1.2rem; margin: 0 0 1.2rem;
              font-size: 1.12rem; line-height: 1.5; max-width: 900px; }
.answer-box b { color: #9FD4D6; }
.lead { color: #2A3442; font-size: 1rem; line-height: 1.55; margin: -.4rem 0 1.2rem; max-width: 900px; }
.page-title { color: #1B2A4A; font-size: 2.15rem; font-weight: 700; line-height: 1.15; margin: 0 0 .6rem;
              letter-spacing: -.01em; }
.callout { background: #E3F4F4; border-radius: 6px; padding: .85rem 1.1rem; color: #1B2A4A; line-height: 1.55;
           margin: 0 0 1.4rem; max-width: 900px; }
.callout b { color: #1B2A4A; }

/* Cover banner (matches the review deck's title slide) */
.cover { background: #1B2A4A; border-top: 8px solid #00868B; border-radius: 6px; padding: 2rem 2.2rem 1.6rem;
         color: #E8EEF6; margin-bottom: 1.4rem; }
.cover h1 { color: #FFFFFF; font-size: clamp(1.6rem, 3.2vw, 2.4rem); line-height: 1.15; margin: 0 0 .5rem;
            font-weight: 700; padding: 0; }
.cover .sub { color: #9FD4D6; font-size: 1.05rem; margin: 0 0 1.2rem; }
.cover .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .8rem 1.6rem;
               border-top: 1px solid #34497A; padding-top: 1rem; }
.cover .meta div span { display: block; color: #9AA9C2; font-size: .82rem; }
.cover .meta div b { color: #FFFFFF; font-weight: 600; font-size: .98rem; }

/* Key figures */
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; margin: .2rem 0 1.2rem; }
.fig { background: #F2F5F8; border-radius: 6px; padding: .75rem .9rem; border-top: 3px solid #00868B; }
.fig .v { font-size: 1.5rem; font-weight: 700; color: #1B2A4A; line-height: 1.2; white-space: nowrap; }
.fig .l { color: #5B6576; font-size: .85rem; line-height: 1.3; }
.figs.stack { grid-template-columns: 1fr; }

/* Test verdicts */
.verdict { border-left: 4px solid; padding: .6rem .9rem; border-radius: 0 6px 6px 0; margin: .3rem 0 1rem;
           line-height: 1.5; }
.verdict.ok { border-color: #2E8B57; background: #E6F4EC; }
.verdict.bad { border-color: #C0392B; background: #F9E6E4; }
.verdict.na { border-color: #E07B2E; background: #FCEEE2; }

/* Risk pills */
.pill { display: inline-block; padding: .08rem .55rem; border-radius: 999px; font-size: .82rem; font-weight: 600;
        color: #FFFFFF; }
.pill.LOW { background: #2E8B57; } .pill.MEDIUM { background: #E07B2E; } .pill.HIGH { background: #C0392B; }

/* Framework steps and detail panels */
.panel { background: #FFFFFF; border: 1px solid #DDE3EA; border-radius: 6px; padding: 1rem 1.2rem; margin: .4rem 0 1rem; }
.panel h4 { margin: 0 0 .15rem; padding: 0; font-size: .82rem; color: #5B6576; font-weight: 600; }
.panel p { margin: 0 0 .7rem; line-height: 1.5; }
.panel p:last-child { margin-bottom: 0; }
.panel .title { font-size: 1.2rem; font-weight: 700; color: #1B2A4A; margin-bottom: .5rem; }
.panel code { font-size: .8rem; background: #F2F5F8; color: #1B2A4A; padding: .1rem .3rem; border-radius: 3px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: .2rem 1.6rem; }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; margin: .3rem 0 1rem; }
.flow span { background: #1B2A4A; color: #FFFFFF; border-radius: 6px; padding: .45rem .8rem; font-weight: 600;
             font-size: .9rem; }
.flow span:nth-child(4n+3) { background: #00868B; } .flow span:nth-child(4n+5) { background: #3366AA; }
.flow span:nth-child(4n+7) { background: #E07B2E; }
.flow i { color: #5B6576; font-style: normal; }
.answer { font-size: clamp(1.6rem, 3vw, 2.3rem); font-weight: 700; color: #1B2A4A; line-height: 1.15; margin: .2rem 0 .5rem; }
@media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } .cover { padding: 1.4rem 1.2rem; } }
</style>
"""


@dataclass
class Context:
    """Everything a page needs, computed once per run in app.py."""
    raw_label: str = ""
    synthetic: bool = True
    report: dict = field(default_factory=dict)
    clean_df: object = None
    df: object = None
    params: dict = field(default_factory=dict)
    result: object = None
    period: str = ""
    minutes: int = 15
    pages: dict = field(default_factory=dict)


CTX = Context()


def ctx() -> Context:
    return CTX


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def pct(p: float, digits: int = 2) -> str:
    if p != p:  # NaN
        return "n/a"
    return "< 0.01%" if 0 < p < 1e-4 else f"{100 * p:.{digits}f}%"


def slot(period: str) -> str:
    """'13:15-13:30' -> '13:15–13:30' for display."""
    return period.replace("-", "–")


def pill(risk: str) -> str:
    return f'<span class="pill {risk}">{RISK_WORD[risk]}</span>'


# ---------------------------------------------------------------------------
# Layout blocks
# ---------------------------------------------------------------------------
def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def page_header(kicker: str, title: str, answer: str | None = None, lead: str | None = None) -> None:
    """Question as the page title, the short answer in a dark box, an optional one-line lead."""
    block = (f'<p class="page-kicker">{html.escape(kicker)}</p><div class="page-title" role="heading" '
             f'aria-level="1">{html.escape(title)}</div>')
    if answer:
        block += f'<div class="answer-box">{answer}</div>'
    if lead:
        block += f'<p class="lead">{lead}</p>'
    html_block(block)


def chart_block(fig, key: str, caption: str, details: list[tuple[str, str]] | None = None, **kwargs):
    """Chart, a one-line caption underneath, and an 'About this chart' popover with the details."""
    event = chart(fig, key, **kwargs)
    if details:
        c1, c2 = st.columns([5, 1.3], vertical_alignment="center")
        c1.caption(caption)
        with c2.popover("About this chart", icon=":material/info:", width="stretch"):
            for title, body in details:
                st.markdown(f"**{title}**")
                html_block(f'<p style="margin:0 0 .8rem; line-height:1.5">{body}</p>')
    else:
        st.caption(caption)
    return event


def figures(items: list[tuple[str, str]], stacked: bool = False) -> None:
    cells = "".join(f'<div class="fig"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items)
    html_block(f'<div class="figs{" stack" if stacked else ""}">{cells}</div>')


def verdict(kind: str, text: str) -> None:
    html_block(f'<div class="verdict {kind}">{text}</div>')


def chart(fig, key: str, **kwargs):
    return st.plotly_chart(fig, key=key, theme=None, config=PLOT_CONFIG, **kwargs)


SELECTED = "period_sel"  # session key of the slot inspected on the analysis pages


def select_period_from_chart(chart_key: str) -> None:
    """on_select callback: clicking a bar makes that slot the one inspected on every page."""
    points = st.session_state.get(chart_key, {}).get("selection", {}).get("points", [])
    if points and points[0].get("x") in st.session_state.get("_periods", []):
        st.session_state[SELECTED] = points[0]["x"]


def slot_picker(key: str, label: str = "Time slot") -> str:
    """Slot selector shared across pages: changing it here changes it everywhere."""
    periods = st.session_state.get("_periods", [])
    current = st.session_state.get(SELECTED)
    if current not in periods:
        current = periods[0]

    def _store():
        st.session_state[SELECTED] = st.session_state[key]

    st.session_state[key] = current
    return st.selectbox(label, periods, key=key, format_func=slot, on_change=_store)

