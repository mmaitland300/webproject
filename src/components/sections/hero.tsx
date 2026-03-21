"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Github } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";

export function Hero() {
  return (
    <section className="relative min-h-[calc(100vh-4rem)] flex items-center justify-center overflow-hidden">
      <div className="cyber-frame mx-auto max-w-6xl px-6 py-32 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <div className="cyber-pill mb-6 inline-flex items-center gap-3 rounded-full px-4 py-2 text-xs font-medium uppercase tracking-[0.24em] text-[rgba(196,206,223,0.82)] backdrop-blur-sm">
            <span className="h-1.5 w-1.5 rounded-full bg-[rgba(136,212,255,0.95)] shadow-[0_0_12px_rgba(67,188,255,0.45)]" />
            Web Engineer | DSP Developer | Systems Troubleshooting
          </div>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
          className="text-5xl font-semibold leading-[1.02] tracking-tight sm:text-6xl md:text-7xl lg:text-8xl"
        >
          <span className="hero-lead">Hi, I&apos;m</span>{" "}
          <span className="cyber-title hero-name">Matt Maitland</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
          className="mt-6 mx-auto max-w-2xl text-lg leading-relaxed text-[rgba(194,203,220,0.82)] sm:text-xl"
        >
          I work full-time at Auxillium doing technical support for Full Swing.
          On the side when time permits, I attempt to build web and audio
          software, and write, record, produce, and master original music as
          NEUROCHEMICAL ENTROPY.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3, ease: "easeOut" }}
          className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <Link
            href="/projects"
            className={buttonVariants({
              size: "lg",
              className:
                "bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white border-0 shadow-lg shadow-purple-500/25",
            })}
          >
            View Projects <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
          <a
            href="https://github.com/mmaitland300"
            target="_blank"
            rel="noopener noreferrer"
            className={buttonVariants({ variant: "outline", size: "lg" })}
          >
            <Github className="mr-2 h-4 w-4" /> GitHub
          </a>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 1 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
            className="w-6 h-10 border-2 border-muted-foreground/30 rounded-full flex justify-center pt-2"
          >
            <div className="w-1 h-2 bg-muted-foreground/50 rounded-full" />
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
