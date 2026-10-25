# DESIGN.md - Measured RAG

Project design system. Layered on top of [global-rules.md](global-rules.md); where they conflict, this file wins.

Status: LOCKED for direction, accent, theme, mobile priority and the v4 mockup layout (section 12). Items marked (call) are still proposals (section 13).
Reference mockup: OpenDesign project "Measured RAG redesign", file `measured-rag-main-screen-v4.html` (main screen) and `measured-rag-states.html` (offline, model failed, out of scope). Where this file and the mockup differ, this file wins.

## 1. Intent

Measured RAG is an inspector for retrieval, not a chat app. Its promise is "see every step between a file and an answer", so the interface proves the system works by showing the work.

Feeling: a precision instrument. Calm, dark, dense, exact. Closer to Linear or a debugger than to a chatbot.
References: Linear (restraint, off-white on near-black, slow reveals), openbot.run (one warm accent, big confident type).

Not this: a chat bubble UI, a marketing page, a dashboard of equal cards.

## 2. Layout

The pipeline leads. The answer is its output, always visible.

```
+--------------------------------------------------------------+
| top bar: product / collection / commands / theme             |
+--------------------------------------------------------------+
| source strip: drop, paste, collection (compact, one row)     |
+----------------------------------------+---------------------+
|                                        |                     |
|  PIPELINE (hero, ~65% width)           |  ANSWER (~35%)      |
|  ingest stages -> dense | keyword ->   |  question input     |
|  fusion -> rerank -> generate          |  streamed answer    |
|                                        |  citations          |
+----------------------------------------+---------------------+
| status bar: latency / stage timings / model / embedding      |
+--------------------------------------------------------------+
```

- The source panel shrinks to a strip. It is a setup step, not a destination.
- The pipeline takes the main area and the largest type on the page.
- The answer column is fixed and sticky, never hidden or collapsed on desktop.
- Below 900px: answer first, pipeline below, source behind a button.
- Remove the numbered panel headers (01, 02, 03) as the main structure. Numbers belong on pipeline stages only.

Pipeline area, top to bottom (decided in mockup v4):
1. Hero band, pinned: total latency as a 32px mono figure, a one-line summary at 15px ("6 of 20 candidates were reranked, 3 reached the model and all 3 were cited"), and a thin stage-timing bar (neutral greys, the slowest stage in terracotta).
2. Stage rail, pinned under the hero band (36px): the 9 stages in order, grouped Ingest (1-4), Retrieve (5-6), Fuse and rank (7-8), Generate (9). See section 3. The app also has a Cite card; it sits under the Generate group without a stage number and has no rail item.
3. Scrolling area, internal scroll only. Ingest is one compact card with 4 columns divided by hairlines. Then group labels "Retrieve", "Fuse and rank", "Generate" (11px uppercase, letter-spaced, `--tx2`, 24px space above), each followed by its stage cards. Dense and Keyword are two equal columns. Fusion is the widest card.
- The pipeline is taller than the viewport (about 1,270px in a 680px area at 1440x900). That is accepted: the answer column stays fixed and the rail keeps the trace visible.

## 3. Signature interaction

Hover or focus a citation in the answer: the matching chunk row in the pipeline highlights, and the stages it passed through (dense rank, keyword rank, fusion, rerank) light up in sequence. Release: everything settles back. Keyboard focus does the same as hover.

This link between answer and evidence is the one place we spend motion budget.

Stage rail during a trace: the stages the cited chunk passed through light up in the rail (label to `--tx`, 2px accent underline) with its rank at each stage in mono ("Dense #2", "Keyword #4", "Fusion #1", "Rerank #1", "Generate [1]"). Stages it did not reach stay at rest. Default state shows citation [1] traced. Rail items are buttons: click smooth-scrolls to that stage card (instant under reduced motion), the stage in view gets a `--s3` pill, aria-labels include the rank. Hover never scrolls the page.

## 4. Color

Dark is the primary theme. Light is supported by the same tokens.

Keep the existing surface and text tokens in `web/app/globals.css` (`--bg`, `--s1..s3`, `--bd`, `--bd2`, `--tx`, `--tx2`, `--tx3`). They are already a good neutral ramp.

Changes:
- Accent (decided): replace indigo `#6366F1` with a warm terracotta. Indigo is the default AI-tool color and carries no identity. Contrast checked:
  - Dark: `--ac: #D97757` (6.3:1 on `--bg`, 5.8:1 on `--s2`). Text on an accent button is `--bg` (6.3:1).
  - Light: `--ac: #B4532F` (5.0:1 on white, 4.5:1 on `--bg`). `#D97757` fails on light (2.8:1), so never reuse the dark value there. Text on an accent button is white.
  - Derive `--ac-soft`, `--ac-ring` and `--ac-tx` from these two values.
- Remove the hero glow (`--hero`) and any gradient behind headlines.
- Status hues stay, used only for meaning: `--ok`, `--warn`, `--err`, `--refuse`.
- Text greys: exactly 3 (`--tx`, `--tx2`, `--tx3`). `--tx3` must stay at least 4.5:1 on its surface.
- Accent is used for: primary action, active stage, citation highlight, focus ring. Nothing else.

## 5. Type

- One family: Geist (sans) and Geist Mono (data). Remove Bricolage Grotesque.
- Mono is for numbers, scores, timings, ids, and stage names. Sans is for prose.
- Scale (px): 11, 12, 13, 15, 20, 32. 32 is used once, for the hero latency figure.
- Display (32): weight 500, tracking -0.02em, line-height 1.1. Tighter tracking only at large sizes.
- Floors (decided after mockup review): list rows, scores and body text 13px minimum; secondary metadata (timings, paths, headers) 12px minimum; 11px only for uppercase group labels and tiny captions, under 5% of text elements. Mockup v4 still has about half of text at 12px; if the built page feels small, raise data text before changing layout.
- Numbers use tabular figures so timings and scores do not jitter.

## 6. Space, radius, borders

- Spacing scale: 4, 8, 12, 16, 24 (existing). Add 32 for section gaps (call).
- Page padding 24px. Gap between cards 16px. Card body padding 16px, header padding 12px 16px. Rows 32px. 24px below the hero band.
- Radii: 4 (chips, inputs), 12 (stage cards), full (pills). Drop 6, 8 and 10 unless a popover needs 8.
- Borders (decided after owner feedback "too dense, cards not delimited"): stage cards get a 1px hairline `--bd` (#23232C), radius 12, surface `--s1` (#101015) on `--bg`, and a header strip separated from the body by an inner 1px hairline. The surface step alone was too faint at this density. Borders are also allowed on inputs, popovers and the answer column left edge. Nothing else is boxed: list rows have no borders and no backgrounds except hover and the active citation highlight.
- Shadow: popovers only (`--pop-shadow`). No shadow on cards.

## 7. Density

Dense, not cramped.
- Row height 32px for lists (chunk rows, scores). 36px for controls.
- Dense and Keyword lists show the top 5 rows with a quiet "+N more" footer line. Fusion shows 6 fused rows. Do not repeat in a card what the hero band already says.
- Pipeline stage cards align on a shared grid. No overlapping cards: dense search and keyword search must sit in two equal columns with a defined gap (current build overlaps them).
- Scores render as a thin inline bar plus a mono number, not as large badges.

## 8. Motion

Tokens (call: align to the references):
- `--ease-out: cubic-bezier(0.23, 1, 0.32, 1)` for feedback and entrances.
- `--ease-move: cubic-bezier(0.32, 0.72, 0, 1)` for panels and large moves.
- `--t-fast: 120ms` hover, press, toggle.
- `--t-base: 200ms` state changes, popovers.
- `--t-slow: 500ms` stage reveal, panel moves.

Behaviors:
- Stage running: a thin progress line on the stage card, 1px, accent. No spinners on stages.
- Stage done: the timing counts up to its final value once, then holds. Status dot changes color, no flash.
- Rank changes in fusion and rerank: rows slide to their new position (FLIP, transform only, 500ms `--ease-move`). This is the second place we spend motion budget.
- Answer streaming: tokens appear without per-token animation. Citation chips fade in at the end of their sentence.
- Citation trace: see section 3, 200ms in, 120ms out, 40ms stagger between stages.
- Animate transform and opacity only. Honor `prefers-reduced-motion`: replace slides with instant changes and keep the highlight.

## 9. States to design

Every one of these needs a designed state, not a fallback:
- No collection selected
- Collection loading, API offline
- Indexing a file (per-stage progress)
- Retrieval done, model call failed (keep results, offer Retry; already works, restyle only)
- Out-of-scope question (refusal state, uses `--refuse`)
- Rate limited or daily budget reached
- Long answer, many citations, long chunk titles (truncate with title tooltip)

## 10. Components to restyle first

1. Top bar and collection menu
2. Source strip (from panel to compact strip)
3. Pipeline stage card (the unit everything else repeats)
4. Hit list row with score bar and rank delta
5. Answer column, citation chip, citation popover
6. Status bar

## 11. Do and don't

Do:
- Let numbers be the visual interest: timings, scores, counts in mono.
- Keep one accent and use it sparingly.
- Use real Kubernetes content in every mockup.

Don't:
- Add gradients, glow, glass, or colored shadows.
- Put every region in a bordered box.
- Use emoji or decorative icons. Icons must carry meaning.
- Introduce a second typeface.

## 12. Decisions (owner, locked)

- Accent: warm terracotta (see section 4).
- Light theme: best-effort. Dark is designed and reviewed first. Light must keep the same tokens and pass contrast, but gets no bespoke polish or separate mockups.
- Mobile: desktop-first. Design at 1440 and 1280. Below 900px the layout degrades gracefully (section 2) and must not break, but gets no dedicated design pass.
- Layout: pipeline leads, answer pinned beside it (agreed after reviewing the live app).
- Cards are delimited with hairlines (section 6), reversing the earlier "no borders" preference for stage cards only.
- Stage rail added to keep the evidence trace visible while the pipeline scrolls (section 3). Verified in mockup v4 at 1440x900 and 1280x800.

## 13. Remaining (call) items

Not decided by the owner, so treat as proposals until the first mockup review: the 32px type size, the 32px spacing step, and the exact motion values in section 8.
