"use client";

import Link from "next/link";
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
import {
  resumeSkills as skills,
  resumeExperience as experience,
  resumeEducation as education,
  resumeCertifications as certifications,
} from "@/content/resume";
import { HighlightText } from "@/components/ui/highlight-text";

type AboutContentProps = {
  publicEmail: string;
};

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true as const },
  transition: { duration: 0.5 },
};

export function AboutContent({ publicEmail }: AboutContentProps) {
  return (
    <div className="space-y-16">
      <motion.section {...fadeUp}>
        <p className="text-lg font-medium text-foreground leading-relaxed">
          Most of my day is diagnosing problems where the symptom and the root
          cause live in different layers — a customer reports bad ball tracking,
          and the actual issue turns out to be a network switch, a driver
          regression, or a calibration state that drifted after a firmware
          update.
        </p>
        <p className="mt-4 text-lg text-muted-foreground leading-relaxed">
          That kind of structured troubleshooting is what I do full-time at
          Auxillium for Full Swing simulator systems. It has shaped how I
          approach software work too: start with what you can observe, isolate
          variables, and don&apos;t trust a fix you can&apos;t reproduce.
        </p>
        <p className="mt-4 text-lg text-muted-foreground leading-relaxed">
          Music runs as a separate ongoing thread. I play, record, produce, and
          master original material under the name NEUROCHEMICAL ENTROPY. That
          practice sits alongside the software work and directly informs
          projects like StringFlux.
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
              {exp.highlights && exp.highlights.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {exp.highlights.map((highlight, index) => (
                    <li
                      key={`${highlight.text}-${index}`}
                      className="text-sm text-muted-foreground pl-3 relative before:absolute before:left-0 before:top-[0.6em] before:h-1 before:w-1 before:rounded-full before:bg-purple-500/50"
                    >
                      <HighlightText highlight={highlight} />
                    </li>
                  ))}
                </ul>
              )}
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
              {edu.description ? (
                <p className="text-sm text-muted-foreground mt-2">
                  {edu.description}
                </p>
              ) : null}
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
          href={`mailto:${publicEmail}`}
          className={buttonVariants({ variant: "outline" })}
        >
          <Mail className="mr-2 h-4 w-4" /> Email
        </a>
        <Link
          href="/resume"
          className={buttonVariants({ variant: "outline" })}
        >
          <FileDown className="mr-2 h-4 w-4" /> View Resume
        </Link>
      </motion.section>
    </div>
  );
}
