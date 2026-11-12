# Pass 9: matched-phrase activation replay

## Result

The frozen token-only selection produced 167 cross-group exact 8-gram phrase pairs and 24 first-versus-later repeat pairs across 17 streams. All 17 replays exactly matched their saved token prefixes (59,657 replayed tokens); all 17 activation archives were finite. The analysis covered residual layers 0–24 and MLP outputs 0–23.

The mean direction of cross-group activation differences was only weakly consistent across the 25 success-item by failure-item contrasts: mean pairwise cosine ranged from 0.101 to 0.227 across the 48 analyzed features. The strongest value was MLP output layer 3 (0.227). Same-offset background contrasts also showed weak alignment (0.099–0.207; highest MLP output layer 18 at 0.207). Within-failure repeat contrasts varied more (0.035–0.450 across features; highest residual layer 2 at 0.450), but came from only six failure items. These descriptive maxima are selected from 48 features and are not confirmatory.

## Interpretation

This pass found no strong, phrase-specific success/failure activation direction: the cross-group consistency was weak, and similarly weak consistency appeared at the fixed background positions. The more aligned within-failure repeat contrasts are a separate exploratory observation, based on six items and without an independent or causal check. This does not establish a mechanism, identify a useful edit, or reproduce the paper's edited benchmark result.

The comparison is outcome-conditioned and TRAIN-only. The five successful sources came from the earlier observed tranche, and the 12 failures came from pass 8. Exact token replay protects the activation measurement from regenerated text, but it does not make the groups randomized or independent. No weights were edited and no reserved confirmation item was opened.

## Provenance

- Frozen selection and capture request: `0015_matched_thinking_8gram_contexts.json`; request SHA-256 `bfdd6404c3ecae52d2e29e483ea64ceaa26831d6b22575f5c2e164c594a11949`.
- Frozen selection manifest SHA-256: `7846302a226e0b46f62e8e171fa4cb145f2c19fd5507f299b0352b2cbf3cd8`.
- Exact replay and descriptive summary: local ignored artifacts under `results/astra_reproduction_20261111_fresh50/raw/editor_work/`.
- Derived direction-vector archive SHA-256: `ad16d5df193304bb7026452207c0fbd8fdd8944351d33e9cf405a51434a321d2`.
- Raw token traces and activation vectors remain local; this public note contains aggregate results only.

## Next reproduction gate

Do not edit from this observation. A mechanistic claim requires a prespecified intervention and matched control, a dose response, and an untouched confirmation set evaluated with the paper's paired math protocol. The exact author search/orchestration and serving path are still not reproduced by this local exploratory analysis.
