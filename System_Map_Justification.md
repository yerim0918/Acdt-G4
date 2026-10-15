# Three System Maps — Comparative Justification

**Research:** Do movies with higher Metacritic Metascores have higher worldwide box office revenue?

**Dataset:** The group-supplied cleaned 2022–2025 movie dataset. The analysis models 385 films assigned to Action, Comedy, Drama or Horror. The outcome is log worldwide gross (USD). Reported relationships are **observational associations**, not demonstrated causal effects.

## Map A — Statistical Significance-Based

- **Budget:** Strong positive adjusted association with log worldwide revenue (log-budget coefficient approximately 1.160, p < 0.001).
- **Metascore:** Positive adjusted association after controlling for budget, release year and genre (10-point increase approximately +22.9%, p approximately 0.0008). The bivariate association is not significant (approximately -9.1%, p approximately 0.205).
- **Primary genre:** Some differences in estimated revenue levels. Genre also modifies the score–revenue slope in the full interaction model (joint p approximately 0.0493), but genre-moderation significance changes with coding/sample assumptions.
- **Release year:** Included as a control; its 2023–2025 indicator coefficients are not individually statistically significant against reference year 2022 in this sample.

**Read the arrows as associations rather than established causes.** A p-value does not measure the strength of a causal link. A non-significant result does not prove the effect is exactly zero.

## Map B — Theory / Scenario-Based (Including Hidden Features)

This map represents **hypotheses** about how the industry operates rather than observed regression estimates. Potential omitted variables include marketing exposure, franchise recognition, star power, distribution reach, theatrical competition, production quality, word of mouth, streaming release arrangements and international market demand. Some could jointly influence critics' scores and revenue; others could mediate an association or modify how it varies across genres. Feedback is plausible between revenue, attention and audience buzz.

The diagram is a **scenario map, not a verified causal DAG**. Arrow direction, arrow signs and any qualitative strength annotations should not be interpreted as statistically verified. Not all hidden variables have been measured, and some mechanisms may differ across films or time.

### Why non-significant or unmeasured variables may still matter

A small or statistically non-significant coefficient can reflect insufficient information, model misspecification, noisy measurement or correlated predictors, rather than an absence of a meaningful real-world relationship. Unavailable predictors cannot be ruled out just because they are missing from the regression. The study's complete-case selection also excludes many movies with missing budgets and may favor films with wider releases or better public financial reporting.

### Industry decision implications

Assuming critic ratings alone determine commercial success may lead to overinvestment in prestige or review-targeting while neglecting release schedules, international distribution, marketing access, audience reach, platform strategy and franchise effects. Decisions should compare multiple drivers, and future research should collect richer production/distribution data and establish time ordering before attempting causal claims.

## Map C — Machine Learning-Based

Random Forest predicts log worldwide gross from Metascore, log budget, release year and primary genre. In 5-fold cross-validation, Random Forest achieved a mean R² of **0.585**, versus **0.547** for the linear model. Holdout permutation importance ranks log budget highest and primary genre next; Metascore and release year have small or negative measured importance for that single holdout, which should be interpreted as noisy/limited additional prediction contribution rather than evidence of no association.

A **separate** partially linear Double Machine Learning analysis uses 5-fold cross-fitting and Random Forest nuisance models to flexibly adjust for budget, year and genre, yielding a positive estimated Metascore coefficient of approximately 0.02176 log points per score point (about **+24.3%** per 10 points, approximate p = 0.000085). This DML result is an *adjusted association* in this observational study; its causal interpretation would require additional, unverified identifying assumptions.

## Comparison

| System map | Key purpose | What it does **not** establish |
|---|---|---|
| A | Which modeled associations have statistical support | Direct causality; absence of omitted confounding |
| B | Which wider industry mechanisms may matter | Which hidden links are empirically verified |
| C | Which features help prediction and whether flexible adjustment changes the Metascore coefficient | That predictive importance = causal impact; that DML automatically removes all confounding |

### Overall conclusion

Critic Metascore appears positively associated with worldwide box-office revenue **conditional on the measured controls**. Budget is a strong determinant of modeled revenue, and both genre specification and missing industry-level variables limit any simple interpretation of rating effects. The three maps together illustrate why data-driven decisions need predictive tests, theory-informed context and transparent limits.
