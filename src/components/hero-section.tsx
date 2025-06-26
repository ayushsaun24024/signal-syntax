"use client";

import { Button } from "@/components/ui/button";
import { Download, Mouse } from "lucide-react";

const HeroSection = () => {
  const handleScroll = (e: React.MouseEvent<HTMLAnchorElement, MouseEvent>) => {
    e.preventDefault();
    const href = e.currentTarget.href;
    const targetId = href.replace(/.*#/, "");
    const elem = document.getElementById(targetId);
    elem?.scrollIntoView({
      behavior: "smooth",
    });
  };

  return (
    <section id="home" className="relative h-[calc(100vh-4rem)] w-full">
      <div className="container mx-auto flex h-full max-w-5xl flex-col items-center justify-center text-center">
        <h1 className="font-headline text-5xl font-bold tracking-tight text-foreground sm:text-7xl lg:text-8xl">
          Ayush Saun
        </h1>
        <p className="mt-6 text-lg tracking-tight text-muted-foreground sm:text-xl">
          Engineer & Applied Researcher specializing in ETL, MLOps, and Full-Stack Development.
        </p>
        <div className="mt-8 flex flex-col sm:flex-row gap-4">
          <Button size="lg" asChild>
            <a href="/ayush-saun-cv.pdf" download>
              <Download className="mr-2 h-5 w-5" />
              Download CV
            </a>
          </Button>
          <Button size="lg" variant="outline" asChild>
            <a href="#contact" onClick={handleScroll}>
              Contact Me
            </a>
          </Button>
        </div>
      </div>
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2">
        <a href="#about" aria-label="Scroll down" onClick={handleScroll}>
          <div className="flex items-center text-muted-foreground opacity-75 transition-opacity hover:opacity-100">
            <Mouse className="h-8 w-8 animate-bounce" />
          </div>
        </a>
      </div>
    </section>
  );
};

export default HeroSection;
