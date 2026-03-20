import type { Metadata } from "next";
import { Github, Mail, MapPin } from "lucide-react";
import {
  contactInfo,
  resumeCertifications,
  resumeEducation,
  resumeExperience,
  resumeSkills,
  resumeSummary,
} from "@/content/resume";
import { getPublicContactEmail } from "@/lib/site-contact";

export const metadata: Metadata = {
  title: "Resume",
  description:
    "Resume for Matt Maitland covering full-stack development and technical support experience.",
};

export default function ResumePage() {
  const publicEmail = getPublicContactEmail();
  return (
    <div className="py-24">
      <div className="mx-auto max-w-4xl px-6">
        <header className="border-b border-border pb-8">
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-muted-foreground">
            Resume
          </p>
          <h1 className="mt-3 text-4xl font-bold tracking-tight sm:text-5xl">
            {contactInfo.name}
          </h1>
          <p className="mt-4 max-w-3xl text-muted-foreground">
            {resumeSummary}
          </p>
          <div className="mt-6 flex flex-wrap gap-4 text-sm text-muted-foreground">
            <a
              href={`mailto:${publicEmail}`}
              className="inline-flex items-center gap-2 transition-colors hover:text-foreground"
            >
              <Mail className="h-4 w-4" />
              {publicEmail}
            </a>
            <span className="inline-flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              {contactInfo.location}
            </span>
            <a
              href={contactInfo.github}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 transition-colors hover:text-foreground"
            >
              <Github className="h-4 w-4" />
              github.com/mmaitland300
            </a>
          </div>
        </header>

        <section className="mt-10">
          <h2 className="text-xl font-semibold">Experience</h2>
          <div className="mt-5 space-y-6">
            {resumeExperience.map((item) => (
              <article
                key={`${item.role}-${item.company}`}
                className="rounded-xl border border-border bg-card/40 p-5"
              >
                <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
                  <div>
                    <h3 className="font-semibold text-foreground">
                      {item.role}
                    </h3>
                    <p className="text-sm text-purple-400">{item.company}</p>
                  </div>
                  <p className="text-sm text-muted-foreground">{item.period}</p>
                </div>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                  {item.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-xl font-semibold">Skills</h2>
          <div className="mt-4 flex flex-wrap gap-2">
            {resumeSkills.map((skill) => (
              <span
                key={skill}
                className="rounded-full border border-border bg-muted px-3 py-1 text-sm text-muted-foreground"
              >
                {skill}
              </span>
            ))}
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-xl font-semibold">Education</h2>
          <div className="mt-5 space-y-5">
            {resumeEducation.map((item) => (
              <article key={`${item.degree}-${item.school}`}>
                <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
                  <div>
                    <h3 className="font-semibold text-foreground">
                      {item.degree}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {item.school}
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground">{item.period}</p>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {item.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-xl font-semibold">Certifications</h2>
          <div className="mt-5 space-y-5">
            {resumeCertifications.map((item) => (
              <article key={item.name}>
                <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
                  <h3 className="font-semibold text-foreground">{item.name}</h3>
                  <p className="text-sm text-muted-foreground">{item.period}</p>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {item.description}
                </p>
              </article>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
