"use client";

import { MotionConfig } from "framer-motion";

/**
 * Respects prefers-reduced-motion for all Framer Motion usage under this tree.
 */
export function MotionProvider({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
