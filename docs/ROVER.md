# Word-level combination (ROVER)

## Design (written before any ROVER result was computed, 2026-09-13)

Motivation: the MBR selector keeps one whole transcript per clip (dev 0.1712); a
per-clip oracle is 0.1161. Combining at the word level can exceed the per-clip
oracle, because different models can be right in different parts of one clip.

Fixed in advance, for the same reasons as `docs/SELECTOR.md` (155 clips, one
conversation, no other evaluation set):

1. **Inputs:** the four existing dev runs of set A — turbo, Qwen3-ASR-1.7B,
   Qwen3-ASR-0.6B, Whisper large-v3, all auto language. No references, no
   metadata, no confidence scores (none were saved).
2. **Tokens:** each hypothesis is passed through the official normalizer, then split
   on whitespace; combination works on these case-sensitive tokens and outputs them
   joined by spaces. Scoring normalizes again, so this loses nothing if the
   normalizer is idempotent, which is checked before scoring.
3. **Alignment:** a slot network is built by aligning hypotheses one at a time. The
   first hypothesis seeds one slot per token. Each later hypothesis is aligned by
   dynamic programming: putting a token in a slot costs 0 if that word is already in
   the slot, else 1; leaving a slot empty for this hypothesis costs 0 if the slot
   already holds an empty entry, else 1; opening a new slot costs 1. Equal-cost
   choices prefer, in order: token-in-slot, empty entry, new slot.
4. **Voting:** in each slot, every hypothesis casts one vote for its word or for
   "empty". The option with most votes wins; a tie goes to the option supported by
   the earliest hypothesis in alignment order. An "empty" win emits nothing.
5. **Two variants, both reported:**
   - **R1 (primary), `rover-backbone`:** alignment order starts with the clip's MBR
     selection (the hypothesis `mbr_select` picks), then the remaining runs in the
     order turbo, Qwen 1.7B, Qwen 0.6B, large-v3. Ties go to the MBR choice, so a
     word changes only where another option has strictly more votes.
   - **R2 (secondary), `rover-order`:** alignment order is fixed turbo, Qwen 1.7B,
     Qwen 0.6B, large-v3, with no MBR step.
6. **Decision rule:** R1 counts as an improvement only if the paired-bootstrap 95%
   interval of WER(R1) − WER(MBR selector) lies entirely below zero. R2 is judged the
   same way. Both are also compared with turbo alone.
7. **No tuning:** no weights, thresholds or alternative orders are tried on dev. If
   neither variant passes, the MBR selector stays the best system.

## Results (computed after the design above)

Run: `uv run python scripts/combine_runs.py <four runs> --name <name> --method
{rover-backbone,rover-order} --compare-to mbr-setA-turbo-q17-q06-lv3`. Numbers are
**MEASURED** (file and key given) or **DERIVED** (arithmetic shown).

Pre-scoring check: the official normalizer was idempotent (token level) on all 620
hypotheses of the four input runs, so combining normalized tokens loses nothing at
scoring (session command, VERIFIED).

### Neither variant passes the decision rule

MEASURED: `reports/baselines/rover-R1-backbone-setA.json`,
`reports/baselines/rover-R2-order-setA.json`, `comparisons` and
`versus_incumbent`.

| | MBR selector | R1 `rover-backbone` | R2 `rover-order` |
|---|---:|---:|---:|
| WER | **0.1712** | 0.1722 | 0.1776 |
| 95% CI | 0.1460–0.1972 | 0.1480–0.1978 | 0.1543–0.2034 |
| Paired Δ vs MBR selector (95% CI) | — | +0.0010 (−0.0028 to +0.0051) | +0.0064 (−0.0085 to +0.0234) |
| Paired Δ vs turbo (95% CI) | −0.0521 (−0.0852 to −0.0192) | −0.0511 (−0.0834 to −0.0197) | −0.0457 (−0.0699 to −0.0233) |
| Subs / Del / Ins | 531 / 264 / 60 | 515 / 292 / 53 | 529 / 310 / 48 |
| Hypothesis / reference words | 0.9592 | 0.9521 | 0.9475 |
| Clips whose output differs from the backbone | — | 51 | 93 |
| Official CLI parity | true | true | true |

Neither interval against the MBR selector lies below zero, and both point estimates
are worse, so by the rule fixed in advance **the MBR selector remains the best
system** and no further ROVER variants are tried on dev. Both variants are still
reliably better than turbo alone.

### Where it loses

R1 removes 16 substitutions and 7 insertions relative to the selector but adds 28
deletions: 860 errors against 855 (DERIVED 515 + 292 + 53 and 531 + 264 + 60), which
is the +0.0010 above (DERIVED 5 / 4,994). R2 adds 46 deletions. Outputs also get
shorter. The likely mechanism is split votes: where the systems spell or segment a
word differently, no word gets a majority, "empty" wins, and a correct word is
dropped (EXPECTED; transcripts were not inspected). With four systems from two model
families, 2–2 splits are probably common (EXPECTED; tie frequency was not measured).

### Consequence

The submission keeps MBR selection. Getting closer to the 0.1161 per-clip oracle
would need information the vote does not have, such as per-word or per-clip
confidence; that is the router idea (train on Bangor segments, evaluate on dev).
