# Follow-up: three full-cap extraction pairs

## Question and scope

The fixed public p50c checkpoint evaluation observed three extraction tasks (x04–x06) that changed from base cap-hit/strict-fail to edited EOS/strict-pass. This follow-up asks whether those paired outcomes recur under fresh sampling seeds. It does not test the fresh Astra editing procedure, since p50c is the already released edit. The four remaining extraction prompts are not added after seeing the outcomes; the follow-up is limited to the three observed wins and is therefore a targeted replication, not an unbiased benchmark estimate.

## Frozen design

- Arms: pinned original Qwen3.5-0.8B base and released p50c patch at layers 15 and 19. Verify the base and edited tensor hashes before generation.
- Items: x04, x05, x06 from the existing frozen task file (same prompt/gold as the first evaluation).
- Replicates: three fresh paired seeds, `20261111`, `20261112`, `20261113`; each seed is applied to both arms for the same item.
- Total: 3 items × 3 seeds × 2 arms = 18 generations. Thinking enabled, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, EOS IDs unchanged, maximum 32,768 generated tokens. BF16 Transformers on Apple MPS, matching the first local checkpoint probe.
- Primary task success: EOS before cap and exact recursive JSON key/type/value match. Report termination, token count, parsed correctness, and repeat8 separately. No scorer discretion is used for structured extraction.
- The edit is not retuned. Report every seed-item pair, including failures. The replication supports these three prompts only; it cannot establish broad extraction or non-math capability.
- Paired seeds are applied to sequential base/edit generation calls, matching the existing runner. This is not token-for-token common random numbers; sampled trajectories diverge after logits differ.

## Interpretation rule

Call the frozen absolute criterion “met” only if p50c strictly succeeds on at least 2/3 fresh seeds for each of x04, x05, and x06, with corresponding base outcomes reported. This threshold alone does not establish that each original paired win recurred or that p50c outperformed base; report all paired wins and losses. The threshold is descriptive and not a significance test. Do not infer internal loop repair from EOS or repeat8 alone.

## Runtime estimate

The earlier 48-generation full-cap suite took 15,556 seconds (about 4 h 19 min). Linear scaling gives roughly 97 minutes for 18 generations; actual time will vary with EOS and cap hits. The raw generation ledger is local/ignored; only sanitized outcomes and aggregate results are candidates for publication.
