"use client";
import { useEffect, useState } from 'react';
import Image from 'next/image';
import { generateAiBio } from '@/ai/flows/ai-powered-bio';
import { Skeleton } from '@/components/ui/skeleton';
import { cvContentForBio } from '@/lib/data';

const AboutSection = () => {
  const [bio, setBio] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchBio = async () => {
      try {
        setLoading(true);
        const result = await generateAiBio({ cvContent: cvContentForBio });
        setBio(result.bio);
      } catch (error) {
        console.error("Failed to generate AI bio:", error);
        // Fallback bio
        setBio("Engineer & Applied Researcher with over 2 years of experience in turning data into production value. Skilled in cloud-native ETL pipelines, big-data speech processing, and robust data-modeling. Proven ability to deliver high-quality software solutions and lead projects from conception to deployment.");
      } finally {
        setLoading(false);
      }
    };
    fetchBio();
  }, []);

  return (
    <section id="about" className="py-20 lg:py-32">
      <div className="container mx-auto grid max-w-6xl grid-cols-1 items-center gap-12 md:grid-cols-3">
        <div className="col-span-1">
          <div className="relative aspect-square rounded-full overflow-hidden shadow-2xl shadow-primary/20">
            <Image
              src="https://placehold.co/400x400.png"
              alt="Ayush Saun"
              fill
              className="object-cover transition-transform duration-500 hover:scale-105"
              data-ai-hint="professional portrait"
            />
          </div>
        </div>
        <div className="md:col-span-2">
          <h2 className="font-headline text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            About Me
          </h2>
          <div className="mt-6 space-y-4 text-muted-foreground">
            {loading ? (
              <>
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-4/5" />
              </>
            ) : (
              <p className="text-lg leading-relaxed">{bio}</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
