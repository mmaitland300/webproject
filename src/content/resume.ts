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
  "I'm a developer and technical support specialist based in Colorado. I build full-stack web applications and support Full Swing golf simulator systems. My work ranges from Next.js and Django apps to remote hardware/software troubleshooting across production environments.";

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
        text: "Built and shipped this portfolio site (Next.js, Prisma, Auth.js) with contact form, rate limiting, and admin inbox.",
        href: "https://www.mmaitland.dev",
      },
      {
        text: "Building StringFlux, a multiband granular delay/freeze audio plugin for guitar.",
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
        text: "Built triage workflows for recurring simulator issues across calibration, licensing, display, networking, and OS layers.",
        href: "/projects/full-swing-tech-support",
      },
      {
        text: "Turned common one-off fixes into documented playbooks so the same problems stop coming back.",
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
