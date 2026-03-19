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
  // ── Featured / Professional ──────────────────────────────
  {
    slug: "portfolio-site",
    title: "Portfolio Website",
    description:
      "This site — a modern portfolio built from scratch with Next.js, TypeScript, Tailwind CSS, and MDX blogging. Includes an admin inbox, contact form with rate limiting, and GitHub OAuth.",
    problem:
      "Needed a professional web presence that reflects current skills and is easy to maintain and extend.",
    role: "Solo developer — design, implementation, and deployment.",
    outcome:
      "Fully functional portfolio with blog, admin dashboard, and CI-ready codebase.",
    tags: ["Next.js", "TypeScript", "Tailwind CSS", "Prisma", "Auth.js", "MDX"],
    github: "https://github.com/mmaitland300/webproject",
    category: "featured",
  },
  {
    slug: "snake-detector",
    title: "Snake Detector (CNN)",
    description:
      "Built, trained, and tested a Convolutional Neural Network for detecting and classifying snakes in images. Handles data collection, preprocessing, model training, and inference.",
    problem:
      "Needed a reliable way to identify snake species from photos using computer vision.",
    role: "ML engineer — dataset curation, model architecture, training pipeline, and evaluation.",
    outcome:
      "Working CNN model that classifies snake images with practical accuracy.",
    tags: ["Python", "Machine Learning", "CNN", "Computer Vision"],
    github: "https://github.com/mmaitland300/Snake-detector",
    category: "featured",
  },
  {
    slug: "auction-house",
    title: "Auction House",
    description:
      "A full-stack online auction platform built with Django. Users can list items, place bids, manage watchlists, and browse categories.",
    problem:
      "Wanted to build a functional e-commerce-style application with real-time bidding and user accounts.",
    role: "Full-stack developer — database models, authentication, bid logic, and Django templates.",
    outcome:
      "Complete auction platform with user registration, listing management, bidding, and category browsing.",
    tags: ["Django", "Python", "PostgreSQL", "HTML/CSS"],
    github: "https://github.com/mmaitland300/AuctionHouse",
    category: "featured",
  },
  {
    slug: "sample-organizer",
    title: "Sample Library Organizer",
    description:
      "A file organizer built for musicians and producers to sort and manage large sample libraries. Scans directories, categorizes files, and keeps collections clean.",
    problem:
      "Large sample libraries get messy fast — thousands of files with inconsistent naming across dozens of folders.",
    role: "Solo developer — file system logic, categorization rules, and CLI interface.",
    outcome:
      "Practical tool that saves hours of manual sorting for music producers.",
    tags: ["Python", "CLI", "File Systems"],
    github: "https://github.com/mmaitland300/organizer_project",
    category: "featured",
  },

  // ── Experiments ──────────────────────────────────────────
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
  {
    slug: "circles-design",
    title: "Concentric Circles",
    description:
      "A generative art piece using Phaser.js to create concentric circle patterns.",
    tags: ["JavaScript", "Phaser.js", "Generative Art"],
    iframe: "/games/circles/index.html",
    category: "experiment",
  },
];

export function getFeaturedProjects() {
  return projects.filter((p) => p.category === "featured");
}

export function getExperiments() {
  return projects.filter((p) => p.category === "experiment");
}
