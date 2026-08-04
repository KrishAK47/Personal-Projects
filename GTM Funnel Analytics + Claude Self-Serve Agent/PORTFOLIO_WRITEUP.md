# Measuring what actually moved the needle: a GTM funnel + causal inference + Claude agent project

**TL;DR:** I built a small end-to-end project that (1) analyzes acquisition funnel
performance by paid channel, (2) uses difference-in-differences to isolate the
true causal effect of a product change from a naive before/after comparison,
(3) trains a propensity model to prioritize which customers to target next, and
(4) wraps all of it in a Claude-powered agent so a non-technical stakeholder can
just ask a question in plain English and get a grounded, cited answer instead of
waiting on a data pull. Code: [link to your GitHub repo].

## Why I built this

Most of my career has been spent translating messy funnel and experiment data
into decisions — which channels to fund, whether a product change actually
worked, who to prioritize for outreach. The hardest part is rarely the model;
it's making sure the answer is causally sound and getting it in front of the
right person fast enough to matter. This project is a compact, shareable
version of that workflow, built with synthetic data so it's fully public.

## The causal inference part is the point

A naive before/after comparison on the group that got a redesigned upsell flow
showed conversion moving from 8.8% to 13.6% — a huge jump. But the control
group *also* moved slightly over the same window, from 8.7% to 8.9%, just from
seasonality and product-wide changes. Running a difference-in-differences
regression instead of a simple before/after comparison isolates the actual
treatment effect: a **statistically significant +4.5 percentage-point lift**
attributable to the redesign itself (p < 0.001), not to broader trends. That
gap between "what a simple average tells you" and "what actually happened" is
exactly the kind of measurement mistake that leads teams to over- or
under-invest in a change.

## Then: who do we target next?

A gradient-boosted propensity model on tenure, engagement, and support-contact
features ranks customers by upgrade likelihood (test AUC 0.68), producing a
ranked list the growth or lifecycle team could act on directly rather than
guessing who to email next.

## Then: make it self-serve

The last piece is the one I think matters most for how data science teams
actually operate day to day: a Claude agent, built with tool use, that sits on
top of all of the above. Instead of Claude generating numbers from its own
"knowledge," it's given four narrow tools — get the funnel breakdown, get the
DiD result, get the top upgrade candidates, get the model's accuracy — and it
decides which to call based on the question asked. Ask "did the redesign
work?" and it pulls the actual regression result and reports the real
coefficient and p-value, not a hallucinated approximation. That's the same
pattern I've used to give senior stakeholders self-serve access to ad hoc
analysis without a dashboard request queue.

## Stack

Python, pandas, statsmodels (DiD regression), scikit-learn (propensity model),
Claude API with tool use, built with Claude Code.

---

*All data in this project is synthetic, generated for demonstration purposes.
No employer or proprietary data was used.*
