
"use client";

import { useState, useMemo } from 'react';
import type { Project } from '@/lib/data';
import { projects } from '@/lib/data';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import Image from 'next/image';
import { Github, ArrowRight } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import MermaidRenderer from './mermaid-renderer';
import ExpandableView from './expandable-view';


const ProjectsSection = () => {
  const [filter, setFilter] = useState('All');
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);

  const allTechnologies = useMemo(() => 
    ['All', ...Array.from(new Set(projects.flatMap(p => p.technologies)))]
  , []);

  const filteredProjects = useMemo(() => 
    filter === 'All'
      ? projects
      : projects.filter(p => p.technologies.includes(filter))
  , [filter]);

  return (
    <section id="projects" className="py-20 lg:py-32 bg-background">
      <div className="container mx-auto max-w-7xl">
        <h2 className="text-center font-headline text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          Projects
        </h2>
        <p className="mt-4 text-center text-lg text-muted-foreground">
          A selection of my work. Filter by technology to learn more.
        </p>
        
        <div className="my-8 flex flex-wrap justify-center gap-2">
          {allTechnologies.map(tech => (
            <Button 
              key={tech} 
              variant={filter === tech ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilter(tech)}
              className="transition-all"
            >
              {tech}
            </Button>
          ))}
        </div>

        <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-2">
          {filteredProjects.map((project) => (
            <Card key={project.title} className="flex flex-col group overflow-hidden hover:shadow-xl transition-shadow duration-300">
              <div className="aspect-video relative overflow-hidden">
                 <Image 
                  src={project.image} 
                  alt={project.title} 
                  fill
                  className="object-cover transition-transform duration-500 group-hover:scale-105"
                  data-ai-hint={project.aiHint}
                 />
              </div>
              <CardHeader>
                <CardTitle className="font-headline text-2xl">{project.title}</CardTitle>
              </CardHeader>
              <CardContent className="flex-grow">
                <p className="text-muted-foreground">{project.description}</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {project.technologies.slice(0, 5).map(tech => (
                    <Badge key={tech} variant="secondary">{tech}</Badge>
                  ))}
                  {project.technologies.length > 5 && <Badge variant="secondary">...</Badge>}
                </div>
              </CardContent>
              <CardFooter className="flex justify-between items-center">
                 <Button variant="outline" onClick={() => setSelectedProject(project)}>
                    Explore
                    <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                 </Button>
                 {project.link && (
                    <Button variant="ghost" size="icon" asChild>
                      <a href={project.link} target="_blank" rel="noopener noreferrer" aria-label="GitHub Repository">
                        <Github />
                      </a>
                    </Button>
                 )}
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>

      <Dialog open={!!selectedProject} onOpenChange={(isOpen) => !isOpen && setSelectedProject(null)}>
        <DialogContent className="max-w-6xl h-[90vh] flex flex-col">
          <DialogHeader>
            <DialogTitle className="font-headline text-3xl">{selectedProject?.title}</DialogTitle>
            <DialogDescription asChild>
                <div className="flex flex-wrap gap-2 mt-2">
                    {selectedProject?.technologies.map(tech => (
                        <Badge key={tech} variant="secondary">{tech}</Badge>
                    ))}
                </div>
            </DialogDescription>
          </DialogHeader>
          <div className="flex-grow overflow-y-auto pr-6 -mr-6 mt-4 flex flex-col gap-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <ExpandableView title="Technical Insights">
                  <p className="text-muted-foreground text-sm leading-relaxed whitespace-pre-wrap">
                    {selectedProject?.details.insights}
                  </p>
                </ExpandableView>

              {selectedProject?.details.pseudocode && (
                <ExpandableView title="Pseudocode">
                  <pre className="text-xs font-code">
                    <code>{selectedProject.details.pseudocode}</code>
                  </pre>
                </ExpandableView>
              )}
            </div>
            
            {selectedProject?.details.flowchart && (
              <div className="pt-4">
                <ExpandableView title="Flowchart">
                  <MermaidRenderer chart={selectedProject.details.flowchart} />
                </ExpandableView>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </section>
  );
};
export default ProjectsSection;
