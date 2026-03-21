export const contactInfo = {
  name: "Matt Maitland",
  /** Public inbox: set NEXT_PUBLIC_CONTACT_EMAIL for a domain alias; see site-contact helper. */
  location: "Colorado, USA",
  github: "https://github.com/mmaitland300",
};

export type ResumeHighlight = {
  text: string;
  href?: string;
};

export type ResumeExperienceItem = {
  role: string;
  company: string;
  period: string;
  description: string;
  highlights?: ResumeHighlight[];
};

export const resumeSummary =
  "Colorado-based developer and technical support professional focused on practical software, reliable systems, and clear problem solving. Experience includes building full-stack web applications, integrating APIs, deploying production systems, and supporting users through hardware, software, and connectivity issues.";

export const resumeSkills = [
  "TypeScript",
  "JavaScript",
  "React",
  "Next.js",
  "Python",
  "Node.js",
  "Django",
  "Flask",
  "PostgreSQL",
  "MySQL",
  "MongoDB",
  "REST APIs",
  "Tailwind CSS",
  "Apache",
  "Hardware Support",
  "Technical Support",
  "Troubleshooting",
  "TCP/IP",
  "Git",
  "GitHub",
  "HTML5",
  "CSS",
  "SQL",
  "Machine Learning",
];

export const resumeExperience: ResumeExperienceItem[] = [
  {
    role: "Freelance Full-Stack Developer",
    company: "Freelance",
    period: "2018 - Present",
    description:
      "Build custom web applications and software solutions with Django, Flask, React, and Next.js. Handle database design, API integrations, debugging, deployment, and long-term client support from launch through maintenance.",
    highlights: [
      {
        text: "Built and shipped mmaitland.dev (Next.js, Prisma, Auth.js) with validated contact intake, rate limiting, and admin OAuth.",
        href: "https://www.mmaitland.dev",
      },
      {
        text: "StringFlux product page: transient-aware multiband granular delay/freeze plugin currently in active development.",
        href: "https://www.mmaitland.dev/stringflux",
      },
    ],
  },
  {
    role: "Technical Support / Product Support Specialist",
    company: "Auxillium (supporting Full Swing)",
    period: "April 2024 - Present",
    description:
      "Deliver technical support for Full Swing simulator software and hardware, resolving installation, calibration, licensing, display, networking, and performance issues across commercial and residential environments.",
    highlights: [
      {
        text: "Standardized multi-layer triage workflows across calibration, licensing, display, networking, and OS subsystems (documented publicly).",
        href: "/projects/full-swing-tech-support",
      },
      {
        text: "Reduced repeat incidents on recurring failure classes by converting ad-hoc fixes into reproducible diagnostic playbooks.",
      },
    ],
  },
];

export const resumeEducation = [
  {
    degree: "Bachelor's in Biochemistry",
    school: "University of South Florida",
    period: "January 2014 to December 2016",
    description:
      "Completed upper-division science coursework in biochemistry.",
  },
  {
    degree: "Associate in General Studies",
    school: "Florida Southwestern State College",
    period: "January 2008 to December 2011",
    description: "Completed general studies coursework.",
  },
];

export const resumeCertifications = [
  {
    name: "CompTIA A+",
    period: "July 2023 to July 2026",
    description:
      "Validated hands-on skills across hardware, software, networking, troubleshooting, security, mobile devices, and customer support.",
  },
];
