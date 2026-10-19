<!-- diagram-design-profile
name: Lattice RAG
slug: lattice-rag
source-url: none
created: 2026-10-19
updated: 2026-10-19
notes: High-voltage cyber/engineering studio design system for LatticeRAG zero-cloud-cost Hybrid GraphRAG
-->

# Style Guide

**The single source of truth for colors, typography, and tokens.** Every diagram draws from this — not from hex values inlined in other reference files. If you want to change the visual skin of Diagram Design, change this file.

Lattice RAG skin is a high-voltage, cyber-engineering studio palette calibrated for in-process GraphRAG architectures, multi-hop Cypher traversals, and low-latency systems. It pairs an obsidian space-navy ground with a hyper-vibrant electric cyan focal accent, crisp titanium ink, and a multi-entity graph series palette (emerald vector, indigo router, amber cache, rose guardrail, and amethyst graph traversal).

---

## Tokens

### Semantic roles

Every token is referred to by **semantic role**, not by its hex value. Type references (`type-*.md`) and SKILL.md say `accent`, not `#0e7490`.

| Role | Purpose | Default (light) | Default (dark) |
|---|---|---|---|
| `paper` | Page background, default node fill | `#f8fafc` (porcelain titanium slate) | `#0b0f19` (obsidian space navy) |
| `paper-2` | Diagram container bg, secondary fill | `#f1f5f9` (soft slate container) | `#111827` (deep graphite slate) |
| `ink` | Primary text, primary stroke | `#090d16` (deep obsidian black) | `#f1f5f9` (crisp titanium white) |
| `ink-strong` | High-contrast text on warm accent fills | `#000000` | `#ffffff` |
| `muted` | Secondary text, default arrow stroke | `#475569` (cool mineral slate) | `#94a3b8` (cool silver slate) |
| `soft` | Sublabels, boundary labels | `#64748b` (refined slate) | `#cbd5e1` (light silver slate) |
| `rule` | Hairline borders | `rgba(9,13,22,0.12)` | `rgba(34,211,238,0.20)` |
| `rule-solid` | Stronger borders, baselines | `#cbd5e1` (slate-300) | `#1f2937` (slate-800) |
| `accent` | Focal / 1–2 max per diagram | `#0e7490` (deep oceanic cyan) | `#22d3ee` (electric neon cyan) |
| `accent-tint` | Fill for accent-bordered boxes | `rgba(14,116,144,0.08)` | `rgba(34,211,238,0.12)` |
| `link` | HTTP/API calls, external arrows | `#4338ca` (deep electric indigo) | `#818cf8` (electric periwinkle-violet) |

> **Brand palette source:** Lattice RAG Studio design system (`frontend/tailwind.config.js`, `frontend/src/index.css`, `frontend/src/components/GraphExplorer.tsx`). In dark mode, `#0b0f19` anchors the background with `#f1f5f9` ink (17.48:1 contrast), `#22d3ee` electric cyan accent (10.60:1 contrast), `#818cf8` neural router link (6.42:1 contrast), `#94a3b8` muted (7.47:1 contrast), and `#cbd5e1` soft (12.90:1 contrast). In light mode, `#f8fafc` porcelain titanium anchors the background with `#090d16` ink (18.57:1 contrast), `#0e7490` deep cyan accent (5.12:1 contrast), `#4338ca` indigo link (7.55:1 contrast), `#475569` muted (7.24:1 contrast), and `#64748b` soft (4.55:1 contrast). Every single token strictly exceeds the WCAG AA 4.5:1 text accessibility threshold.

### Contrast contract

Every token must clear the threshold for each role it plays, against the background it actually sits on, in both skins. `scripts/verify-contrast.py` computes the WCAG 2.x ratio for every row above against the `paper` of its skin (`ink-strong` against the 0.85 `accent` fill it labels) and fails CI when one falls short.

| Use | Minimum vs. its actual background |
|---|---|
| Regular text, including any `soft`, `link`, or `accent` used as text | 4.5:1 |
| Large text (24px+, or 18.7px+ at weight 600+) | 3:1 |
| Essential non-text marks: lines, arrows, borders, focus outlines, data points | 3:1 against the adjacent color |
| Decorative rules and translucent tints (`rule`, `rule-solid`, `paper-2`, `accent-tint`, fills at opacity) | No minimum, but never the only carrier of meaning |

- **A token used in several roles must pass each one.** `accent` is a stroke in one place and a label in another, so it is held to 4.5:1, not the 3:1 its strokes alone would need. Don't assume a token is non-text because it usually is.
- **Translucent colors are composited first.** An `rgba()` fill or stroke is blended over `paper` before its ratio is measured. A value that clears the bar at full strength can fail at 0.5 opacity.
- **Surfaces other than `paper` cost contrast.** Inside an `accent-tint` box or on `paper-2` every ratio drops. Measure text placed there against that surface, not against `paper`.
- **Lattice RAG measurements:** light `ink` 18.57:1, `accent` 5.12:1, `link` 7.55:1, `muted` 7.24:1, `soft` 4.55:1 on `#f8fafc`; dark `ink` 17.48:1, `accent` 10.60:1, `link` 6.42:1, `muted` 7.47:1, `soft` 12.90:1 on `#0b0f19`.

### Inversion rule (light → dark)

Any `rgba(9,13,22, X)` in light becomes `rgba(241,245,249, X)` in dark. Same opacities, RGB flipped. The accent shifts to electric hyper-cyan `#22d3ee` to pop against the dark obsidian paper.

### Series palette (multi-series chart types & graph entities)

A calibrated set of high-contrast, popping tones mapped directly from LatticeDB property graph entity taxonomies and multi-hop execution components. The "1-focal" rule still holds — `accent` is reserved for the focal series; the palette below covers the rest.

| Token | Light | Dark | Knowledge Graph & Architecture Context |
|---|---|---|---|
| `series-1` | `#047857` (emerald) | `#34d399` | FastEmbed HNSW / Vector Cache / Low Latency |
| `series-2` | `#4338ca` (electric indigo) | `#818cf8` | TypeSafe AI Jev / System 1 Choice Router |
| `series-3` | `#b45309` (amber gold) | `#fbbf24` | Tier-2 Redis Fallback / Circuit Breakers |
| `series-4` | `#be123c` (crimson rose) | `#fb7185` | Jev Noul Guardrail / Noise Pruning / Outage |
| `series-5` | `#7e22ce` (amethyst violet) | `#c084fc` | LatticeDB Cypher Relational 2-Hop Traversal |

Fills sit at `0.18` opacity light, `0.22` dark; strokes use the full color. All light series tokens clear 4.80:1 to 7.55:1 WCAG contrast on `#f8fafc`. All dark series tokens clear 6.42:1 to 11.47:1 WCAG contrast on `#0b0f19`.

### Terminal skin (opt-in alternate)

A self-contained palette for the terminal-window primitive (see [primitive-terminal.md](primitive-terminal.md)) — a CLI-chrome register for dev-tool posts, Granian ASGI traces, and server telemetry.

| Token | Hex | Purpose |
|---|---|---|
| `terminal-page` | `#050811` | Page background behind the window |
| `terminal-paper` | `#0b0f19` | Window body, node fill |
| `terminal-bar` | `#111827` | Titlebar strip |
| `terminal-border` | `#1f2937` | Window border, hairlines |
| `terminal-ink` | `#f1f5f9` | Primary text, primary stroke |
| `terminal-muted` | `#94a3b8` | Secondary text, sublabels, ring stroke |
| `terminal-soft` | `#64748b` | Tertiary — inactive dots, spokes |
| `terminal-accent` | `#22d3ee` | The one accent — active command, prompt sign, focal step |
| `terminal-accent-tint` | `rgba(34,211,238,0.14)` | Fill for accent-bordered boxes |

**1-accent rule still holds.** Everything that isn't `terminal-ink` or `terminal-muted`/`terminal-soft` should be `terminal-accent` — never introduce a second hue.

---

## Typography

| Role | Family | Size | Weight | Usage |
|---|---|---|---|---|
| `title` | Instrument Serif | 1.75rem | 400 | Architectural overview titles, Diagram H1 |
| `node-name` | Inter | 12px | 600 | Engine components, server modules, graph nodes |
| `sublabel` | Geist Mono | 9px | 400 | Latencies (<20ms), model names, memory footprints |
| `eyebrow` | Geist Mono | 7–8px | 500, tracked 0.18em, uppercase | Architectural tiers (`TRIAGE`, `GRAPH`, `ISOLATION`) |
| `arrow-label` | Geist Mono | 8px | 400, tracked 0.06em | Protocol calls (`CYPHER_HOP`, `VECTOR_MATCH`) |
| `callout` | Instrument Serif *italic* | 14px | 400 | Zero-cost design takeaways, runtime constraints |

### Font stack

```html
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&family=Noto+Serif:ital@0;1&family=Noto+Sans+KR:wght@400;500;600&family=Noto+Serif+KR:wght@400&family=Noto+Sans+TC:wght@400;500;600&family=Noto+Serif+TC:wght@400&display=swap" rel="stylesheet">
```

### Font source

Font source: `web`

| Value | What ships | Network |
|---|---|---|
| `web` (default) | The Google Fonts `<link>` above; text renders in Instrument Serif, Inter, and Geist Mono | One stylesheet request to `fonts.googleapis.com`, plus the font files |
| `system` | No font `<link>`; every family stack falls through to the platform's own serif, sans, and monospace faces | None |

When generating standalone vector figures for offline technical documentation, use `system` mode if offline compilation is needed. Fallbacks gracefully resolve to Charter/Georgia (serif), Inter/system-ui (sans), and ui-monospace.

### Korean labels

Inter, Geist, and Instrument Serif carry no Hangul. A Korean `<text>` element extends its own family — never swap the skin:

```svg
<text font-family="'Inter', 'Geist', 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif">결제 서비스</text>
```

### Traditional Chinese labels

Inter, Geist, and Instrument Serif carry no Han. A Traditional Chinese `<text>` element extends its own family — never swap the skin:

```svg
<text font-family="'Inter', 'Geist', 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif">請求項比對</text>
```

### Cyrillic labels

Geist and Geist Mono ship Cyrillic (`cyrillic` and `cyrillic-ext` on Google Fonts), so names, sublabels, arrow labels, eyebrows, and legend text in Cyrillic keep the standard treatment.

---

## Stroke, radius, spacing

| Token | Value | Use |
|---|---|---|
| `stroke-thin` | `0.8` | Tag-box outlines, leaf nodes, sub-tiers |
| `stroke-default` | `1` | Most node borders and connectors |
| `stroke-strong` | `1.2` | Primary pipeline paths, focal components |
| `radius-sm` | `4` | Small badges and taxonomy tags |
| `radius-md` | `6` | Core engine node boxes |
| `radius-lg` | `8` | Subgraphs, worker process pools, boundaries |
| `grid` | `4` | Every coord, size, and gap is divisible by 4 (hard rule) |

---

## Node type → treatment

Semantic role combinations — reference these by name in type specs.

| Type | Fill | Stroke |
|---|---|---|
| `focal` (1–2 max) | `accent-tint` | `accent` |
| `backend` | `#111827` (dark) / `#ffffff` (light) | `ink` |
| `store` | `ink @ 0.05` | `muted` |
| `external` | `ink @ 0.03` | `ink @ 0.30` |
| `input` | `muted @ 0.10` | `soft` |
| `optional` | `ink @ 0.02` | `ink @ 0.20` dashed `4,3` |
| `security` | `accent @ 0.05` | `accent @ 0.50` dashed `4,4` |

---

## Customizing the skin

Four options:

1. **Run onboarding** — see [`onboarding.md`](onboarding.md).
2. **Edit by hand** — change the hex values in the tables above.
3. **Brand handoff** — paste your existing design-token JSON into a new section here and map its tokens to the semantic roles above.
4. **Client profiles** — save and switch named skins, or bind one to a project, using [`profiles.md`](profiles.md).

### Constraints (don't break these)

- **Contrast**: every text token (`ink`, `muted`, `soft`, `link`, and `accent`) must clear 4.5:1 on `paper`, in both skins.
- **One accent**: pick one color for `accent`. Two accents erases the focal signal.
- **No rainbow palette**: keep the foundation disciplined (paper, ink, accent, link). The series palette is strictly for multi-category or multi-hop data.
- **Serif + sans + mono**: three families, not more.
- **Paper is clean porcelain slate (#f8fafc) or obsidian navy (#0b0f19)**: eliminates sterile glare while giving an electric, popping modern engineering aesthetic.
- **Container is clean by default**: the diagram sits directly on paper.
