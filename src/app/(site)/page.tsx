import { MainContentAnchor } from "@/components/layout/main-content-anchor";
import { Hero } from "@/components/sections/hero";
import { ProofStrip } from "@/components/sections/proof-strip";
import { FeaturedProjects } from "@/components/sections/featured-projects";

export default function HomePage() {
  return (
    <>
      <Hero />
      <MainContentAnchor />
      <ProofStrip />
      <FeaturedProjects />
    </>
  );
}
