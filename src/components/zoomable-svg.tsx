
"use client";

import { useState, useRef, WheelEvent, MouseEvent, TouchEvent } from 'react';
import { Button } from './ui/button';
import { ZoomIn, ZoomOut, Move } from 'lucide-react';

export const ZoomableSVG = ({ children }: { children: React.ReactNode }) => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleWheel = (e: WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const scaleAmount = -e.deltaY > 0 ? 1.1 : 1 / 1.1;
    const newScale = Math.max(0.1, Math.min(scale * scaleAmount, 5));

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const newX = mouseX - (mouseX - position.x) * (newScale / scale);
    const newY = mouseY - (mouseY - position.y) * (newScale / scale);

    setScale(newScale);
    setPosition({ x: newX, y: newY });
  };

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return; // Only pan with left-click
    e.preventDefault();
    setIsPanning(true);
    setStartPoint({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!isPanning) return;
    e.preventDefault();
    setPosition({
      x: e.clientX - startPoint.x,
      y: e.clientY - startPoint.y,
    });
  };

  const handleMouseUpOrLeave = (e: MouseEvent<HTMLDivElement>) => {
    if (isPanning) {
      e.preventDefault();
      setIsPanning(false);
    }
  };

  const handleTouchStart = (e: TouchEvent<HTMLDivElement>) => {
    if (e.touches.length === 1) {
        e.preventDefault();
        setIsPanning(true);
        setStartPoint({ x: e.touches[0].clientX - position.x, y: e.touches[0].clientY - position.y });
    }
  };

  const handleTouchMove = (e: TouchEvent<HTMLDivElement>) => {
    if (!isPanning || e.touches.length !== 1) return;
    e.preventDefault();
    setPosition({
        x: e.touches[0].clientX - startPoint.x,
        y: e.touches[0].clientY - startPoint.y,
    });
  };

  const handleTouchEnd = (e: TouchEvent<HTMLDivElement>) => {
      setIsPanning(false);
  };

  const zoom = (factor: number) => {
    const newScale = Math.max(0.1, Math.min(scale * factor, 5));
    
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const newX = centerX - (centerX - position.x) * (newScale / scale);
    const newY = centerY - (centerY - position.y) * (newScale / scale);
    
    setScale(newScale);
    setPosition({ x: newX, y: newY });
  }

  const reset = () => {
    setScale(1);
    setPosition({x: 0, y: 0});
  }

  return (
    <div 
        className="relative w-full h-full overflow-hidden bg-background/50 rounded-md touch-none" 
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUpOrLeave}
        onMouseLeave={handleMouseUpOrLeave}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
    >
      <div
        className={`w-full h-full ${isPanning ? 'cursor-grabbing' : 'cursor-grab'}`}
        style={{
          transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
          transition: isPanning ? 'none' : 'transform 0.1s ease-out',
          transformOrigin: 'center center',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        {children}
      </div>
      <div className="absolute bottom-2 right-2 flex gap-1 bg-background/70 backdrop-blur-sm p-1 rounded-md border">
        <Button variant="outline" size="icon" className="h-8 w-8" onClick={() => zoom(1.2)} aria-label="Zoom In"><ZoomIn/></Button>
        <Button variant="outline" size="icon" className="h-8 w-8" onClick={() => zoom(1/1.2)} aria-label="Zoom Out"><ZoomOut/></Button>
        <Button variant="outline" size="icon" className="h-8 w-8" onClick={reset} aria-label="Reset View"><Move/></Button>
      </div>
    </div>
  );
};
