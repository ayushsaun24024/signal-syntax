"use client";

import { useTheme } from "@/components/theme-provider";
import React, { useRef, useEffect } from "react";

const AnimatedBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let particles: Particle[];

    const options = {
      particleColor: theme === "dark" ? "rgba(255, 255, 255, 0.5)" : "rgba(75, 0, 130, 0.5)",
      lineColor: theme === "dark" ? "rgba(255, 255, 255, 0.1)" : "rgba(75, 0, 130, 0.1)",
      particleAmount: 50,
      defaultRadius: 2,
      variantRadius: 1,
      defaultSpeed: 0.5,
      variantSpeed: 0.5,
      linkRadius: 200,
    };

    let w = (canvas.width = window.innerWidth);
    let h = (canvas.height = window.innerHeight);

    class Particle {
      x: number;
      y: number;
      radius: number;
      speed: number;
      directionAngle: number;
      dx: number;
      dy: number;

      constructor() {
        this.x = Math.random() * w;
        this.y = Math.random() * h;
        this.radius = options.defaultRadius + Math.random() * options.variantRadius;
        this.speed = options.defaultSpeed + Math.random() * options.variantSpeed;
        this.directionAngle = Math.floor(Math.random() * 360);
        this.dx = Math.cos(this.directionAngle) * this.speed;
        this.dy = Math.sin(this.directionAngle) * this.speed;
      }

      update() {
        this.border();
        this.x += this.dx;
        this.y += this.dy;
      }

      border() {
        if (this.x < 0 || this.x > w) this.dx *= -1;
        if (this.y < 0 || this.y > h) this.dy *= -1;
      }

      draw() {
        if (!ctx) return;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fillStyle = options.particleColor;
        ctx.fill();
      }
    }

    const createParticles = () => {
      particles = [];
      for (let i = 0; i < options.particleAmount; i++) {
        particles.push(new Particle());
      }
    };

    const linkParticles = () => {
      if (!ctx) return;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const distance = Math.sqrt(
            Math.pow(particles[i].x - particles[j].x, 2) +
            Math.pow(particles[i].y - particles[j].y, 2)
          );
          if (distance < options.linkRadius) {
            const opacity = 1 - distance / options.linkRadius;
            ctx.strokeStyle = options.lineColor.replace(/,\s*\d+\.\d+\)/, `, ${opacity})`);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
    };
    
    const animate = () => {
      if (!ctx) return;
      ctx.clearRect(0, 0, w, h);
      particles.forEach((particle) => {
        particle.update();
        particle.draw();
      });
      linkParticles();
      animationFrameId = requestAnimationFrame(animate);
    };

    const handleResize = () => {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
      createParticles();
    };

    window.addEventListener("resize", handleResize);
    createParticles();
    animate();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [theme]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full block"
    />
  );
};

export default AnimatedBackground;
