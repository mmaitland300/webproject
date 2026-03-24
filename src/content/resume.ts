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

export type ResumeEducationItem = {
  degree: string;
  school: string;
  period: string;
  description?: string;
};

export const resumeSummary =
  "Developer and technical support specialist based in Colorado. I work at Auxillium, which provides technical support for Full Swing simulator customers. Many of the same tickets include Laser Shot or E6 Golf from TruGolf on those systems, and I support those as part of the job. I build web and audio software on my own time. Most of my professional experience is multi-layer troubleshooting across hardware, software, and networking, alongside web application development with Next.js and Django.";

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
    role: "Independent Software Projects",
    company: "Self-directed",
    period: "2022 - Present",
    description:
      "Build and maintain web applications and audio software. Current focus: this portfolio site (Next.js, Prisma, Auth.js) and StringFlux, a JUCE/C++ multiband granular delay plugin for guitar.",
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
    company: "Auxillium (technical support for Full Swing)",
    period: "April 2024 - Present",
    description:
      "Remote technical support for Full Swing simulator customers on behalf of Auxillium: installation, calibration, licensing, display, networking, and performance across commercial and residential environments. Also troubleshoot Laser Shot and E6 Golf from TruGolf when those products are part of the same simulator setup.",
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

export const resumeEducation: ResumeEducationItem[] = [
  {
    degree: "Bachelor's in Biochemistry",
    school: "University of South Florida",
    period: "January 2014 to December 2016",
  },
  {
    degree: "Associate in General Studies",
    school: "Florida Southwestern State College",
    period: "January 2008 to December 2011",
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
