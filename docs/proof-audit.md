# Portfolio Proof Audit

This document is the execution gate for messaging changes.  
Phase work should prefer **softening claims over inventing placeholder proof**.

**Homepage composition:** the live homepage is intentionally minimal (Hero, ProofStrip, FeaturedProjects only). That is a **product choice**, not a deferred structural fix. Optional sections such as “work with me” or “recent writing” are additive polish, not required for narrative coherence.

## Locked data contracts (before implementation)

- `status`: `"in-progress" | "operational" | "shipped" | "archived"`
- `proofLinks`: `Array<{ label: string; href: string; kind?: "repo" | "test" | "ci" | "post" | "artifact" }>`

## Audit row schema

Each claim uses one row with this exact shape:

`surface -> claim -> current proof -> proof gap -> action (keep | soften | add evidence)`

## Current audit (baseline)

| Surface | Claim | Current proof | Proof gap | Action |
| --- | --- | --- | --- | --- |
| `src/components/sections/hero.tsx` | Multi-domain identity (web + DSP + troubleshooting) is a coherent operating model | Name + role badge + Auxillium detail in hero body | No direct method/proof statement on homepage hero itself | add evidence |
| `src/components/sections/hero.tsx` | Site is an active build log ("what I'm learning") | Explicit sentence in hero body | Undercuts senior shipping signal at top of funnel | soften |
| `src/app/about/page.tsx` | About section frames a disciplined engineering method | About page title/description mention broad background | Header currently reads broad identity list ("Developer, Builder, Technical Support") instead of method-led framing | soften |
| `src/components/sections/proof-strip.tsx` | Full Swing support uses repeatable triage methodology | Links to Full Swing case study | Needs one deeper artifact callout visible from strip | add evidence |
| `src/components/sections/proof-strip.tsx` | Site production reliability decisions are documented | Link to contact decision record post | Lacks direct CI/test link from this surface | add evidence |
| `src/components/sections/proof-strip.tsx` | StringFlux has transient-aware and safe oversampling behavior | Link to StringFlux case study | Missing explicit true-now vs not-yet-claimed framing | add evidence |
| `src/app/projects/page.tsx` | Projects page separates strongest evidence from exploratory work | `ProjectGrid` separates featured vs experiments visually | Header copy still bundles professional work + side projects + experiments as one layer | soften |
| `src/content/resume.ts` | Resume summary communicates focused systems diagnosis + constrained building approach | Current summary includes support + web context | Summary remains broad and keyword-heavy | soften |
| `src/content/resume.ts` | Skills communicate current strengths accurately | Flat list contains broad + older items together | No tiering; C++/JUCE/DSP absent from skills list | add evidence |
| `src/app/projects/portfolio-site/page.tsx` | Portfolio case study shows production-ready safeguards | Safeguards list + evidence links + architecture artifact | Evidence links missing CI workflow, env tests, contact action tests | add evidence |
| `src/app/projects/portfolio-site/page.tsx` | "Where it stands" reflects a strong current state | Lists live behaviors | Ends with weak "still iterating" sentence | soften |
| `src/app/projects/full-swing-tech-support/page.tsx` | Full Swing case study proves multi-layer diagnosis under constraints | Workflow, incident pattern, and evidence links exist | Needs denser ruled-out-layers artifact/table for faster credibility scan | add evidence |
| `src/app/projects/stringflux/page.tsx` | StringFlux claims are engineering-grounded and bounded | Architecture, constraints, tradeoffs, validation checks, links | No explicit "true now / being validated / not claimed yet" block | add evidence |
| `src/app/blog/page.tsx` | Blog is outward-facing engineering utility | Current copy says "Notes From the Build" + in-progress framing | Positioning still inward; no formal post type taxonomy yet | soften |

## Phase 2 implementation guardrail

Phase 2 is complete only if both commands pass after `Project` model changes:

- `npm test`
- `npm run build`

## Scope guardrail

Do not rewrite experiment projects during Phases 1-3 unless they block shared components.
