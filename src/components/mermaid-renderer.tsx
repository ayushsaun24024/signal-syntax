
"use client";

import { useEffect, useState, useId } from 'react';
import mermaid from 'mermaid';
import { useTheme } from '@/components/theme-provider';
import { Skeleton } from './ui/skeleton';
import { ZoomableSVG } from './zoomable-svg';

const MermaidRenderer = ({ chart }: { chart: string }) => {
  const { theme } = useTheme();
  const [svg, setSvg] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const diagramId = `mermaid-${useId().replace(/:/g, "")}`;

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: theme === 'dark' ? 'dark' : 'neutral',
      securityLevel: 'loose',
      fontFamily: 'Inter, sans-serif',
      themeVariables: {
        background: 'transparent',
        primaryColor: theme === 'dark' ? '#E6E6FA' : '#4B0082',
        primaryTextColor: theme === 'dark' ? '#0f172a' : '#ffffff',
        lineColor: theme === 'dark' ? '#94a3b8' : '#475569',
        textColor: theme === 'dark' ? '#f8fafc' : '#0f172a',
      }
    });

    const renderMermaid = async () => {
      setLoading(true);
      setError(null);
      setSvg(null); 
      try {
        await mermaid.parse(chart);
        const { svg: renderedSvg } = await mermaid.render(diagramId, chart);
        setSvg(renderedSvg);
      } catch (e: any) {
        console.error("Mermaid rendering error:", e);
        setError("Could not render the flowchart. Please check the Mermaid syntax.");
        setSvg(null);
      } finally {
        setLoading(false);
      }
    };
    
    // Defer rendering to ensure the container is ready, especially in modals.
    const timerId = setTimeout(renderMermaid, 100);

    return () => clearTimeout(timerId);
  }, [chart, theme, diagramId]);

  if (loading) {
    return <Skeleton className="w-full h-full min-h-64" />;
  }

  if (error) {
    return <div className="text-destructive p-4 bg-destructive/10 rounded-md h-full flex items-center justify-center">{error}</div>;
  }

  return svg ? (
    <ZoomableSVG>
      <div dangerouslySetInnerHTML={{ __html: svg }} />
    </ZoomableSVG>
  ) : <div className="text-muted-foreground flex items-center justify-center h-full">No flowchart available.</div>;
};

export default MermaidRenderer;
