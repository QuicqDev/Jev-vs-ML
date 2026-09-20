# Release graphic provenance

Created with the built-in image generation tool. All eight rows, signs, and values were visually checked against `published_results/raw_balanced_accuracy.csv`. The source PNG is preserved without pixel edits in `assets/benchmark-release.png`.

The graphic is an independent QuicqDev benchmark; it does not imply OpenAI or Anthropic authorship or endorsement. The website's quantitative charts are generated directly from the published CSVs.

## Final generation prompt

Use case: infographic-diagram. Create a finished editorial benchmark release graphic for independent research "Jev vs. classical ML". Landscape 3:2, high-resolution, impeccably precise typography. Inspired by the restrained scientific clarity of leading AI research releases: warm ivory #f6f4ed background, near-black type, ample whitespace, fine gray rules, muted terracotta #bd644d for Jev, slate #657578 for classical. No logos or affiliation with OpenAI or Anthropic. No gradients, decorative AI symbols, robots, fake error bars, badges, or 3D.
Top small label: "QUICQDEV / BENCHMARK REPORT 03"
Large title: "Jev vs. classical ML"
Subtitle: "Eight datasets. Eleven classical pipelines. One shared test set per dataset."
Main graphic: immaculate table with eight rows and four columns. Heading "Balanced accuracy (%) · raw decisions". Columns "Dataset", "Jev zero-shot", "Best classical", "Gap (pp)". EXACT rows:
AG News | 87.5 | 88.4 | −0.9
Banking77 | 78.9 | 89.7 | −10.8
SMS Spam | 96.1 | 95.0 | +1.1
IMDb | 96.3 | 88.4 | +7.9
Bank Marketing | 53.4 | 71.8 | −18.4
Online Shoppers | 51.4 | 69.1 | −17.7
Breast Cancer | 61.0 | 100.0 | −39.0
Iris | 97.0 | 100.0 | −3.0
Align numbers precisely. Thin subtle rules. Highlight IMDb row with pale terracotta background; other rows neutral. Use a tiny terracotta swatch by Jev header and slate by classical header, no misleading bar geometry. Table dominates the image.
Below table strong takeaway: "Strong on sentiment. Mixed across tasks."
Small footer in readable type: "Jev 1.13.0 · V3 protocol · Means across 3 seeds; cached zero-shot predictions reused."
Second footer: "Bounded-budget ML comparison. Small tabular holdouts. Banking77 includes API-failure warnings."
Bottom source: "github.com/QuicqDev/Jev-vs-ML"
All exact data and signs must be legible and accurate. This is a designed research figure ready for a public release, not a screenshot of an interface.
