export type ProjectCategory = "featured" | "experiment";

export interface Project {
  slug: string;
  title: string;
  description: string;
  longDescription?: string;
  problem?: string;
  role?: string;
  outcome?: string;
  image?: string;
  tags: string[];
  github?: string;
  demo?: string;
  iframe?: string;
  category: ProjectCategory;
}

export const projects: Project[] = [
  {
    slug: "stringflux",
    title: "StringFlux",
    description:
      "A transient-aware, multiband granular delay and freeze plugin for guitar and other stringed instruments.",
    problem:
      "Generic granular processors often miss instrument-specific transient behavior, which limits expressive control for stringed-instrument performance.",
    role:
      "Solo developer responsible for product direction, DSP architecture, and implementation.",
    outcome:
      "In progress. Current implementation includes 3-band crossover routing, transient and density-driven grain scheduling, history/freeze capture, feedback-bus processing, and safe 1x/2x/4x oversampling transitions.",
    tags: [
      "Audio Plugin",
      "DSP",
      "Granular Synthesis",
      "Transient Detection",
      "Oversampling",
    ],
    github: "https://github.com/mmaitland300/StringFlux.git",
    category: "featured",
  },
  {
    slug: "portfolio-site",
    title: "Portfolio Website",
    description:
      "Production Next.js portfolio with MDX blogging, a protected admin inbox, and an abuse-resistant contact flow. Built with typed content/data structures and deployment-ready environment management.",
    problem:
      "Needed a credible public portfolio that could showcase work, accept contact reliably, and support iterative updates without breaking production.",
    role: "Solo developer, system design, UI implementation, data modeling, auth, deployment, and documentation.",
    outcome:
      "Shipped a live site with server-side validation, honeypot plus Redis rate limiting, GitHub OAuth admin gating, and draft-post protection.",
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
      "Image-classification pipeline for snake species using a custom CNN workflow: dataset preparation, preprocessing, training/evaluation loops, and inference scripts.",
    problem:
      "Needed a repeatable way to classify snake photos under real-world data constraints like inconsistent image quality, class imbalance, and noisy samples.",
    role: "ML engineer, data curation, preprocessing strategy, model and training setup, and error analysis.",
    outcome:
      "Delivered an end-to-end computer vision prototype with reproducible training/evaluation flow and a clear path for future model tuning.",
    tags: ["Python", "Machine Learning", "CNN", "Computer Vision"],
    github: "https://github.com/mmaitland300/Snake-detector",
    category: "featured",
  },
  {
    slug: "auction-house",
    title: "Auction House",
    description:
      "Full-stack Django auction platform with account auth, listing lifecycle management, bid validation rules, watchlists, and category browsing.",
    problem:
      "Needed to implement transactional auction behavior with reliable server-side rules for bidding and ownership in a multi-user workflow.",
    role: "Full-stack developer, data modeling, auth/session flows, bid logic, template UI implementation, and route-level behavior.",
    outcome:
      "Built a complete web app covering core auction flows (create, list, bid, watch, manage) with server-enforced business rules and persistent relational data.",
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
    role: "Technical support specialist, incident triage, remote diagnostics, root-cause isolation, and customer-facing resolution guidance.",
    outcome:
      "Improved issue resolution quality by applying structured troubleshooting playbooks and reproducible diagnostic paths across recurring failure modes.",
    tags: [
      "Technical Support",
      "Troubleshooting",
      "Windows",
      "Networking",
      "Hardware/Software Integration",
    ],
    category: "featured",
  },
  {
    slug: "sample-organizer",
    title: "Sample Library Organizer",
    description:
      "A file organizer built for musicians and producers to sort and manage large sample libraries. Scans directories, categorizes files, and keeps collections clean.",
    problem:
      "Large sample libraries get messy fast, with thousands of files and inconsistent naming across dozens of folders.",
    role: "Solo developer, file system logic, categorization rules, and CLI interface.",
    outcome:
      "Practical tool that saves hours of manual sorting for music producers.",
    tags: ["Python", "CLI", "File Systems"],
    github: "https://github.com/mmaitland300/organizer_project",
    category: "featured",
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

export function getExperiments() {
  return projects.filter((p) => p.category === "experiment");
}
