export interface Project {
  slug: string;
  title: string;
  description: string;
  longDescription?: string;
  image?: string;
  tags: string[];
  github?: string;
  demo?: string;
  iframe?: string;
  featured: boolean;
}

export const projects: Project[] = [
  {
    slug: "portfolio-site",
    title: "Portfolio Website",
    description:
      "This site! A modern portfolio built with Next.js, TypeScript, Tailwind CSS, and MDX for blogging.",
    tags: ["Next.js", "TypeScript", "Tailwind CSS", "MDX"],
    github: "https://github.com/mmaitland300/webproject",
    featured: true,
  },
  {
    slug: "atoms-simulation",
    title: "Atoms Simulation",
    description:
      "An interactive particle simulation built with Phaser.js. Watch atoms bounce and interact in real time.",
    tags: ["JavaScript", "Phaser.js", "Canvas", "Physics"],
    iframe: "/games/atoms/index.html",
    featured: true,
  },
  {
    slug: "circles-design",
    title: "Concentric Circles",
    description:
      "A mesmerizing generative art piece using Phaser.js to create concentric circle patterns.",
    tags: ["JavaScript", "Phaser.js", "Generative Art"],
    iframe: "/games/circles/index.html",
    featured: true,
  },
  {
    slug: "turn-based-rpg",
    title: "Turn-Based RPG",
    description:
      "A browser-based turn-based RPG with world exploration, battle system, and sprite-based graphics.",
    tags: ["JavaScript", "Phaser.js", "Game Dev", "RPG"],
    iframe: "/games/rpg/index.html",
    featured: true,
  },
];
