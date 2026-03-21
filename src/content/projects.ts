export type ProjectCategory = "featured" | "experiment";

export interface Project {
  slug: string;
  title: string;
  description: string;
  longDescription?: string;
  problem?: string;
  constraints?: string;
  tradeoff?: string;
  role?: string;
  outcome?: string;
  outcomeType?: "metric" | "proxy" | "technical" | "qualitative";
  image?: string;
  tags: string[];
  github?: string;
  demo?: string;
  caseStudy?: string;
  iframe?: string;
  category: ProjectCategory;
}

/**
 * Homepage keeps one card per major signal area:
 * DSP build, systems troubleshooting, and business-logic-heavy web app.
 */
const HOMEPAGE_FEATURED_SLUGS = [
  "stringflux",
  "full-swing-tech-support",
  "auction-house",
] as const;

export const projects: Project[] = [
  {
    slug: "stringflux",
    title: "StringFlux",
    description:
      "A transient-aware, multiband granular delay and freeze plugin for guitar and other stringed instruments.",
    problem:
      "Generic granular processors often miss instrument-specific transient behavior, which limits expressive control for stringed-instrument performance.",
    constraints:
      "Real-time DSP constraints required safe oversampling transitions, predictable latency behavior, and stable processing under varying host settings.",
    tradeoff:
      "Prioritized deterministic scheduling and engine stability over feature breadth so instrument response remains controllable while the system is still evolving.",
    role:
      "Solo developer responsible for product direction, DSP architecture, and implementation.",
    outcome:
      "In progress. Current implementation includes 3-band crossover routing, transient and density-driven grain scheduling, history/freeze capture, feedback-bus processing, and safe 1x/2x/4x oversampling transitions.",
    outcomeType: "technical",
    tags: [
      "Audio Plugin",
      "DSP",
      "Granular Synthesis",
      "Transient Detection",
      "Oversampling",
    ],
    image: "/images/stringflux/ui-advanced.png",
    github: "https://github.com/mmaitland300/StringFlux.git",
    demo: "/stringflux",
    caseStudy: "/projects/stringflux",
    category: "featured",
  },
  {
    slug: "portfolio-site",
    title: "Portfolio Website",
    description:
      "Production Next.js portfolio with MDX blogging, a protected admin inbox, and an abuse-resistant contact flow. Built with typed content/data structures and deployment-ready environment management.",
    problem:
      "Needed a credible public portfolio that could showcase work, accept contact reliably, and support iterative updates without breaking production.",
    constraints:
      "Needed secure admin access, abuse-resistant contact handling, and deploy-safe content workflows without introducing heavy operational overhead.",
    tradeoff:
      "Chose server actions plus managed services (Resend, Upstash, Neon) over custom infrastructure to improve reliability and iteration speed.",
    role: "Solo developer, system design, UI implementation, data modeling, auth, deployment, and documentation.",
    outcome:
      "Shipped a live site with server-side validation, honeypot plus Redis rate limiting, GitHub OAuth admin gating, and draft-post protection.",
    outcomeType: "technical",
    tags: [
      "Next.js",
      "TypeScript",
      "Tailwind CSS",
      "Prisma",
      "Auth.js",
      "Upstash",
      "MDX",
    ],
    github: "https://github.com/mmaitland300/webproject",
    category: "featured",
  },
  {
    slug: "snake-detector",
    title: "Snake Detector (CNN)",
    description:
      "CV experiment focused on dataset hygiene and evaluation discipline: stratified splits, augmentation policy, and confusion-matrix-driven error review before architecture changes.",
    problem:
      "Snake photo classification fails quietly when class imbalance, label noise, and inconsistent image quality are ignored; naive training runs look fine on paper but generalize poorly.",
    constraints:
      "Limited and noisy source data increased overfitting risk; without explicit val discipline, mistakes read as model issues instead of data issues.",
    tradeoff:
      "Invested in reproducible splits, logging, and error analysis before expanding model capacity so tuning decisions stay tied to measurable failure modes.",
    role: "ML engineer, data curation, preprocessing strategy, model and training setup, and error analysis.",
    outcome:
      "End-to-end pipeline where every training run is comparable (same splits, metrics, artifacts) and poor classes surface in review instead of hiding in aggregate accuracy.",
    outcomeType: "technical",
    tags: ["Python", "Machine Learning", "CNN", "Computer Vision"],
    github: "https://github.com/mmaitland300/Snake-detector",
    caseStudy: "/projects/snake-detector",
    category: "experiment",
  },
  {
    slug: "auction-house",
    title: "Auction House",
    description:
      "Full-stack Django auction platform with account auth, listing lifecycle management, bid validation rules, watchlists, and category browsing.",
    problem:
      "Needed to implement transactional auction behavior with reliable server-side rules for bidding and ownership in a multi-user workflow.",
    constraints:
      "Correctness depended on enforcing bid and listing rules server-side across concurrent user actions and persistent relational data.",
    tradeoff:
      "Used server-rendered Django patterns and strict model-level rules to prioritize correctness and maintainability over richer client-side interactivity.",
    role: "Full-stack developer, data modeling, auth/session flows, bid logic, template UI implementation, and route-level behavior.",
    outcome:
      "Built a complete web app covering core auction flows (create, list, bid, watch, manage) with server-enforced business rules and persistent relational data.",
    outcomeType: "technical",
    tags: ["Django", "Python", "PostgreSQL", "HTML/CSS"],
    github: "https://github.com/mmaitland300/AuctionHouse",
    category: "featured",
  },
  {
    slug: "full-swing-tech-support",
    title: "Full Swing Technical Support Case Study",
    description:
      "Systems-focused support work across simulator hardware/software stacks, including diagnostics for calibration drift, networking/configuration failures, licensing issues, and Windows/peripheral conflicts.",
    problem:
      "Customers needed fast, accurate triage and resolution for multi-layer issues where hardware, software, and environment variables intersected.",
    constraints:
      "Failures crossed calibration, licensing, networking, display, and OS layers, often with incomplete information and high user frustration.",
    tradeoff:
      "Used structured triage and reproducible diagnostic paths instead of quick one-off fixes to reduce repeat incidents and speed future resolution.",
    role: "Technical support specialist, incident triage, remote diagnostics, root-cause isolation, and customer-facing resolution guidance.",
    outcome:
      "Improved issue resolution quality by applying structured troubleshooting playbooks and reproducible diagnostic paths across recurring failure modes.",
    outcomeType: "qualitative",
    tags: [
      "Technical Support",
      "Troubleshooting",
      "Windows",
      "Networking",
      "Hardware/Software Integration",
    ],
    caseStudy: "/projects/full-swing-tech-support",
    category: "featured",
  },
  {
    slug: "sample-organizer",
    title: "Sample Library Organizer",
    description:
      "A file organizer built for musicians and producers to sort and manage large sample libraries. Scans directories, categorizes files, and keeps collections clean.",
    problem:
      "Large sample libraries get messy fast, with thousands of files and inconsistent naming across dozens of folders.",
    constraints:
      "Needed predictable categorization and safe file operations across inconsistent folder structures and naming conventions.",
    tradeoff:
      "Prioritized deterministic rules and repeatable CLI workflows over heuristic automation to keep results understandable and reversible.",
    role: "Solo developer, file system logic, categorization rules, and CLI interface.",
    outcome:
      "Replaced multi-hour manual folder walks with a single deterministic CLI pass (dry-run, explicit rules, reversible moves) for large libraries.",
    outcomeType: "proxy",
    tags: ["Python", "CLI", "File Systems"],
    github: "https://github.com/mmaitland300/organizer_project",
    category: "experiment",
  },
  {
    slug: "turn-based-rpg",
    title: "Turn-Based RPG",
    description:
      "A browser-based turn-based RPG with world exploration, a battle system, and sprite-based graphics.",
    tags: ["JavaScript", "Phaser.js", "Game Dev", "RPG"],
    iframe: "/games/rpg/index.html",
    category: "experiment",
  },
  {
    slug: "atoms-simulation",
    title: "Atoms Simulation",
    description:
      "An interactive particle simulation built with Phaser.js. Watch atoms bounce and interact in real time.",
    tags: ["JavaScript", "Phaser.js", "Canvas", "Physics"],
    iframe: "/games/atoms/index.html",
    category: "experiment",
  },
];

export function getFeaturedProjects() {
  return projects.filter((p) => p.category === "featured");
}

/** Curated subset for the homepage hero grid (strongest proof, least noise). */
export function getHomepageFeaturedProjects(): Project[] {
  const featured = getFeaturedProjects();
  return HOMEPAGE_FEATURED_SLUGS.map((slug) => {
    const p = featured.find((x) => x.slug === slug);
    if (!p) {
      throw new Error(`Homepage featured slug missing from data: ${slug}`);
    }
    return p;
  });
}

export function getExperiments() {
  return projects.filter((p) => p.category === "experiment");
}
