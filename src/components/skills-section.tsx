import { skills } from '@/lib/data';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Code, BrainCircuit, Cloud, Laptop, Award } from 'lucide-react';

const iconMap: { [key: string]: React.ReactNode } = {
  'Programming Languages & Scripting': <Code className="h-8 w-8 text-primary" />,
  'AI, Machine Learning & Data Science': <BrainCircuit className="h-8 w-8 text-primary" />,
  'Cloud & DevOps': <Cloud className="h-8 w-8 text-primary" />,
  'Web & Software Development': <Laptop className="h-8 w-8 text-primary" />,
  'Professional Skills': <Award className="h-8 w-8 text-primary" />,
};

const SkillsSection = () => {
  return (
    <section id="skills" className="py-20 lg:py-32 bg-secondary/50">
      <div className="container mx-auto max-w-6xl">
        <h2 className="text-center font-headline text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          Technical Skills
        </h2>
        <p className="mt-4 text-center text-lg text-muted-foreground">
          A collection of technologies and tools I am proficient in.
        </p>
        <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-2">
          {Object.entries(skills).map(([category, skillList]) => (
            <Card key={category} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-center gap-4">
                  {iconMap[category] || <Code className="h-8 w-8 text-primary" />}
                  <CardTitle className="font-headline text-xl">{category}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {(skillList as string[]).map((skill) => (
                    <Badge key={skill} variant="secondary" className="text-sm">{skill}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
export default SkillsSection;
