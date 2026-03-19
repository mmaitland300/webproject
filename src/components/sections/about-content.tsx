"use client";

import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Code2,
  Briefcase,
  GraduationCap,
  Github,
  FileDown,
  Mail,
  ShieldCheck,
} from "lucide-react";
import { buttonVariants } from "@/components/ui/button";

const skills = [
  "Python",
  "JavaScript",
  "React",
  "Next.js",
  "Node.js",
  "Django",
  "Flask",
  "PostgreSQL",
  "MySQL",
  "MongoDB",
  "REST APIs",
  "Apache",
  "Machine Learning",
  "Technical Support",
  "TCP/IP",
  "Git",
  "GitHub",
  "HTML5",
  "CSS",
  "SQL",
];

const experience = [
  {
    role: "Freelance Web and Software Developer",
    company: "Maitland Web Design",
    period: "January 2018 to Present",
    description:
      "Build custom web and software solutions using Django, Flask, React, and Next.js. Design databases, integrate third-party APIs, train machine learning models for image-based object detection, deploy applications, and support clients from delivery through maintenance.",
  },
  {
    role: "Warehouse Associate",
    company: "Giant Food",
    period: "March 2017 to January 2020",
    description:
      "Handled pallet movement, inventory flow, and shipping deadlines in a large-scale warehouse environment while helping improve team efficiency, stock accuracy, and day-to-day operations.",
  },
  {
    role: "Maintenance",
    company: "Carroll Property Management",
    period: "February 2010 to August 2014",
    description:
      "Maintained hotel, RV park, and marina properties, operated heavy equipment, built utility lines for new RV lots, and handled electrical troubleshooting, demolition, and construction-related repairs.",
  },
  {
    role: "Detailer",
    company: "Fridays Auto Sales",
    period: "April 2006 to September 2009",
    description:
      "Prepared vehicles for customers while supporting mechanics with maintenance and repair work, including brakes, starters, oil changes, inspections, and shop organization.",
  },
];

const education = [
  {
    degree: "Bachelor's in Biochemistry",
    school: "University of South Florida",
    period: "January 2014 to December 2016",
    description:
      "Completed upper-division science coursework while building strong analytical and research habits that continue to inform technical problem solving.",
  },
  {
    degree: "Associate in General Studies",
    school: "Florida Southwestern State College",
    period: "January 2008 to December 2011",
    description:
      "Built a broad academic foundation before continuing into more specialized study.",
  },
];

const certifications = [
  {
    name: "CompTIA A+",
    period: "July 2023 to July 2026",
    description:
      "Validated hands-on skills across hardware, software, networking, troubleshooting, security, mobile devices, and customer support.",
  },
];

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true as const },
  transition: { duration: 0.5 },
};

export function AboutContent() {
  return (
    <div className="space-y-16">
      <motion.section {...fadeUp}>
        <p className="text-lg text-muted-foreground leading-relaxed">
          I&apos;m a Colorado-based developer with experience across web
          development, software support, operations, and hands-on technical
          problem solving. I enjoy learning quickly, building practical
          solutions, and doing the work it takes to get a job done correctly,
          whether that means shipping a client application, troubleshooting a
          system issue, or improving a process.
        </p>
      </motion.section>

      <motion.section {...fadeUp}>
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-purple-500/10">
            <Code2 className="h-5 w-5 text-purple-400" />
          </div>
          <h2 className="text-2xl font-bold">Skills & Technologies</h2>
        </div>
        <div className="flex flex-wrap gap-2">
          {skills.map((skill) => (
            <Badge
              key={skill}
              variant="secondary"
              className="px-3 py-1.5 text-sm"
            >
              {skill}
            </Badge>
          ))}
        </div>
      </motion.section>

      <Separator className="bg-border" />

      <motion.section {...fadeUp}>
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-cyan-500/10">
            <Briefcase className="h-5 w-5 text-cyan-400" />
          </div>
          <h2 className="text-2xl font-bold">Experience</h2>
        </div>
        <div className="space-y-6">
          {experience.map((exp) => (
            <div
              key={`${exp.role}-${exp.company}`}
              className="relative pl-6 border-l-2 border-border"
            >
              <div className="absolute -left-[7px] top-1.5 w-3 h-3 rounded-full bg-purple-500 border-2 border-background" />
              <h3 className="font-semibold text-foreground">{exp.role}</h3>
              <p className="text-sm text-purple-400">{exp.company}</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {exp.period}
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                {exp.description}
              </p>
            </div>
          ))}
        </div>
      </motion.section>

      <motion.section {...fadeUp}>
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-pink-500/10">
            <GraduationCap className="h-5 w-5 text-pink-400" />
          </div>
          <h2 className="text-2xl font-bold">Education</h2>
        </div>
        <div className="space-y-6">
          {education.map((edu) => (
            <div
              key={`${edu.degree}-${edu.school}`}
              className="relative pl-6 border-l-2 border-border"
            >
              <div className="absolute -left-[7px] top-1.5 w-3 h-3 rounded-full bg-pink-500 border-2 border-background" />
              <h3 className="font-semibold text-foreground">{edu.degree}</h3>
              <p className="text-sm text-pink-400">{edu.school}</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {edu.period}
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                {edu.description}
              </p>
            </div>
          ))}
        </div>
      </motion.section>

      <motion.section {...fadeUp}>
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-emerald-500/10">
            <ShieldCheck className="h-5 w-5 text-emerald-400" />
          </div>
          <h2 className="text-2xl font-bold">Certifications</h2>
        </div>
        <div className="space-y-6">
          {certifications.map((cert) => (
            <div
              key={cert.name}
              className="relative pl-6 border-l-2 border-border"
            >
              <div className="absolute -left-[7px] top-1.5 w-3 h-3 rounded-full bg-emerald-500 border-2 border-background" />
              <h3 className="font-semibold text-foreground">{cert.name}</h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                {cert.period}
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                {cert.description}
              </p>
            </div>
          ))}
        </div>
      </motion.section>

      <Separator className="bg-border" />

      <motion.section {...fadeUp} className="flex flex-wrap gap-3">
        <a
          href="https://github.com/mmaitland300"
          target="_blank"
          rel="noopener noreferrer"
          className={buttonVariants({ variant: "outline" })}
        >
          <Github className="mr-2 h-4 w-4" /> GitHub
        </a>
        <a
          href="mailto:mmaitland300@gmail.com"
          className={buttonVariants({ variant: "outline" })}
        >
          <Mail className="mr-2 h-4 w-4" /> Email
        </a>
        <a
          href="/resume.pdf"
          download
          className={buttonVariants({ variant: "outline" })}
        >
          <FileDown className="mr-2 h-4 w-4" /> Download Resume
        </a>
      </motion.section>
    </div>
  );
}
