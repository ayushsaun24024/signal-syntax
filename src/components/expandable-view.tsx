
"use client";

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Maximize } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

const ExpandableView = ({ title, children }: { title: string, children: React.ReactNode }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center mb-2">
        <h3 className="font-headline text-xl">{title}</h3>
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
          <DialogTrigger asChild>
            <Button variant="ghost" size="icon">
              <Maximize className="h-5 w-5" />
              <span className="sr-only">Expand</span>
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-[95vw] w-full h-[95vh] flex flex-col">
            <DialogHeader>
              <DialogTitle className="font-headline text-2xl">{title}</DialogTitle>
            </DialogHeader>
            <div className="flex-grow overflow-auto p-4 border rounded-md bg-muted/20" key={String(isOpen)}>
              {children}
            </div>
          </DialogContent>
        </Dialog>
      </div>
      <div className="relative border rounded-md p-4 bg-muted/50 flex-grow h-72 overflow-auto">
        {children}
      </div>
    </div>
  );
};

export default ExpandableView;
