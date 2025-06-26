import { Github, Linkedin } from "lucide-react";

const Footer = () => {
  return (
    <footer className="border-t">
      <div className="container mx-auto flex h-16 items-center justify-between">
        <p className="text-sm text-muted-foreground">
          &copy; {new Date().getFullYear()} Ayush Saun. All rights reserved.
        </p>
        <div className="flex items-center space-x-4">
          <a href="https://github.com/ayushsaun24024" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-muted-foreground hover:text-foreground">
            <Github />
          </a>
          <a href="https://www.linkedin.com/in/ayush-saun-381371180/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className="text-muted-foreground hover:text-foreground">
            <Linkedin />
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
