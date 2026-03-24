/**
 * Promotion rule:
 * featured projects should have a case-study proof path (dedicated caseStudy page
 * with explicit artifacts/tradeoffs/current-state/evidence links).
 */
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
  image?: string;
  tags: string[];
  github?: string;
  demo?: string;
  caseStudy?: string;
  iframe?: string;
  category: ProjectCategory;
}

const HOMEPAGE_FEATURED_SLUGS = [
  "stringflux",
  "portfolio-site",
  "full-swing-tech-support",
] as const;

export const projects: Project[] = [
  {
    slug: "stringflux",
    title: "StringFlux",
    description:
      "A multiband granular delay and freeze plugin I'm building for guitar. It listens for transients and uses them to drive grain scheduling, so the texture responds to how you actually play instead of running on a fixed clock.",
    problem:
      "Most granular processors treat every input the same. For stringed instruments, that means pick attacks get smeared and the effect feels disconnected from the performance.",
    constraints:
      "Everything runs in the audio callback, so oversampling changes and grain scheduling have to be real-time safe. I can't rebuild state mid-buffer without risking glitches.",
    tradeoff:
      "I've kept the feature set narrow on purpose. Getting the engine stable and the transient response right matters more than adding controls nobody can trust yet.",
    outcome:
      "Still in progress. The current build has 3-band crossover routing, transient-driven grain scheduling, history/freeze capture, and safe 1x/2x/4x oversampling transitions.",
    tags: [
      "Audio Plugin",
      "DSP",
      "Granular Synthesis",
      "Transient Detection",
      "Oversampling",
    ],
    image: "/images/stringflux/ui-advanced.png",
    github: "https://github.com/mmaitland300/StringFlux",
    demo: "/stringflux",
    caseStudy: "/projects/stringflux",
    category: "featured",
  },
  {
    slug: "portfolio-site",
    title: "Portfolio Website",
    description:
      "This site. Next.js 16 with MDX blogging, a contact form that sends email through Resend and falls back gracefully when the database is down, and a GitHub OAuth admin inbox for managing submissions.",
    problem:
      "I needed somewhere to put my work that wasn't just a GitHub profile. It had to accept contact without getting spammed, and I wanted to be able to iterate on it without worrying about breaking production.",
    constraints:
      "Solo project, so operational overhead had to stay low. No dedicated backend: managed services (Resend for email, Upstash for rate limiting, Neon for Postgres) handle the heavy parts.",
    tradeoff:
      "Server Actions over API routes, managed services over self-hosted infra. More vendor lock-in, but significantly less to maintain and debug alone.",
    outcome:
      "Live at mmaitland.dev with honeypot + Redis rate limiting on contact, GitHub OAuth admin gating, and MDX blog with draft protection.",
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
    image: "/images/projects/portfolio-site-projects.png",
    caseStudy: "/projects/portfolio-site",
    category: "featured",
  },
  {
    slug: "full-swing-tech-support",
    title: "Full Swing Technical Support",
    description:
      "My day job. I work at Auxillium supporting Full Swing simulator customers remotely. Many setups also run Laser Shot or E6 Golf from TruGolf, which I support on the same tickets. This case study documents the triage approach I've built from that work.",
    problem:
      "Simulator issues rarely have one cause. A customer reports \"the ball isn't tracking\" and the root cause could be calibration drift, a licensing timeout, a network config problem, or a Windows update that broke a driver.",
    constraints:
      "Incomplete information is the norm. Customers are frustrated, logs aren't always available, and you're working remotely across all of these layers at once.",
    tradeoff:
      "Slower upfront diagnosis instead of quick one-off fixes. Takes more time per ticket, but the same failure patterns stop coming back.",
    role: "Technical support specialist at Auxillium. Scope is Full Swing simulator deployments plus Laser Shot and E6 Golf from TruGolf when those are part of the install.",
    outcome:
      "Built repeatable triage workflows that I now use across calibration, licensing, display, networking, and OS subsystems. Documented publicly as a case study.",
    tags: [
      "Technical Support",
      "Troubleshooting",
      "Windows",
      "Networking",
      "Hardware/Software Integration",
    ],
    image: "/images/projects/full-swing-triage-artifact.svg",
    caseStudy: "/projects/full-swing-tech-support",
    category: "featured",
  },
  {
    slug: "snake-detector",
    title: "Snake Detector (CNN)",
    description:
      "A CNN image classifier for snake species. The interesting part wasn't the model, it was learning how much dataset quality matters. I spent more time on stratified splits, augmentation, and confusion-matrix-driven error review than on architecture.",
    problem:
      "The raw dataset had class imbalance, noisy labels, and inconsistent image quality. Naive training runs looked fine on aggregate accuracy but generalized poorly.",
    outcome:
      "An end-to-end pipeline where every training run uses the same splits and metrics, and poor-performing classes surface in review instead of hiding in the average.",
    tags: ["Python", "Machine Learning", "CNN", "Computer Vision"],
    github: "https://github.com/mmaitland300/Snake-detector",
    caseStudy: "/projects/snake-detector",
    category: "experiment",
  },
  {
    slug: "auction-house",
    title: "Auction House",
    description:
      "A Django auction platform from CS50 Web. Users can create listings, place bids, manage watchlists, and browse by category. All bid validation and ownership rules are enforced server-side.",
    problem:
      "The main challenge was getting the bid logic right: ensuring server-side rules handle concurrent actions and edge cases like bidding on your own listing or closed auctions.",
    outcome:
      "A working multi-user auction app with auth, listing lifecycle, bid validation, watchlists, and category browsing.",
    tags: ["Django", "Python", "PostgreSQL", "HTML/CSS"],
    github: "https://github.com/mmaitland300/AuctionHouse",
    category: "experiment",
  },
  {
    slug: "sample-organizer",
    title: "Sample Library Organizer",
    description:
      "A Python CLI tool I built to sort my sample libraries. It scans directories, categorizes audio files by deterministic rules, and moves them into a clean folder structure. Supports dry-run so you can preview before committing.",
    image: "/images/projects/sample-organizer-loaded.png",
    tags: ["Python", "CLI", "File Systems"],
    github: "https://github.com/mmaitland300/organizer_project",
    category: "experiment",
  },
  {
    slug: "turn-based-rpg",
    title: "Turn-Based RPG",
    description:
      "A browser RPG prototype I built to learn Phaser's scene system. Turn-based combat, sprite movement, and scene transitions, all running client-side with no backend.",
    tags: ["JavaScript", "Phaser.js", "Game Dev", "RPG"],
    iframe: "/games/rpg/index.html",
    category: "experiment",
  },
  {
    slug: "atoms-simulation",
    title: "Atoms Simulation",
    description:
      "An interactive particle simulation built with Phaser. Mostly a sandbox for playing with collision detection and continuous animation loops in the browser.",
    tags: ["JavaScript", "Phaser.js", "Canvas", "Physics"],
    iframe: "/games/atoms/index.html",
    category: "experiment",
  },
];

export function getFeaturedProjects() {
  return projects.filter((p) => p.category === "featured");
}

/** Curated subset for the homepage hero grid. */
export function getHomepageFeaturedProjects(): Project[] {
  return HOMEPAGE_FEATURED_SLUGS.map((slug) => {
    const p = projects.find((x) => x.slug === slug);
    if (!p) {
      throw new Error(`Homepage featured slug missing from data: ${slug}`);
    }
    return p;
  });
}

export function getExperiments() {
  return projects.filter((p) => p.category === "experiment");
}
