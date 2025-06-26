import { timeline } from '@/lib/data';
import { Briefcase, GraduationCap } from 'lucide-react';

const ExperienceTimeline = () => {
  return (
    <section id="experience" className="py-20 lg:py-32">
      <div className="container mx-auto max-w-4xl">
        <h2 className="text-center font-headline text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          Career & Education
        </h2>
        <p className="mt-4 text-center text-lg text-muted-foreground">
          My professional journey and academic background.
        </p>
        <div className="relative mt-12 pl-6">
          <div className="absolute left-0 top-0 h-full w-0.5 bg-border ml-[11px]"></div>
          {timeline.map((item, index) => (
            <div key={index} className="relative mb-10 pl-10">
              <div className="absolute -left-1.5 top-1 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground ring-8 ring-background">
                {item.type === 'work' ? <Briefcase size={16}/> : <GraduationCap size={16}/>}
              </div>
              <p className="text-sm text-accent">{item.date}</p>
              <h3 className="mt-1 font-headline text-xl font-semibold">{item.title}</h3>
              <h4 className="font-medium text-primary">{item.company}</h4>
              <p className="mt-2 text-muted-foreground">{item.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ExperienceTimeline;
