import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";
import AnimatedBackground from "@/components/animated-background";
import PwaInstaller from "@/components/pwa-installer";

export const metadata: Metadata = {
  title: "Signals & Syntax | Ayush Saun",
  description: "Personal portfolio of Ayush Saun, an Engineer & Applied Researcher.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#4B0082" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Playfair+Display:wght@700&display=swap" rel="stylesheet" />
      </head>
      <body className="font-body antialiased">
        <ThemeProvider>
          <AnimatedBackground />
          <div className="relative z-10">
            {children}
          </div>
          <Toaster />
          <PwaInstaller />
        </ThemeProvider>
      </body>
    </html>
  );
}
