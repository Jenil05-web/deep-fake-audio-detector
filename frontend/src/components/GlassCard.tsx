import React, { useRef } from 'react';
import type { MouseEvent } from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  glowColor?: 'green' | 'red' | 'blue' | 'none';
  hoverable?: boolean;
  onClick?: () => void;
  id?: string;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = '',
  glowColor = 'none',
  hoverable = true,
  onClick,
  id
}) => {
  const cardRef = useRef<HTMLDivElement | null>(null);

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    const card = cardRef.current;
    if (!card) return;

    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    card.style.setProperty('--x', `${x}px`);
    card.style.setProperty('--y', `${y}px`);
  };

  const getGlowStyles = () => {
    switch (glowColor) {
      case 'green':
        return 'shadow-glass-green border-cyber-neonGreen/20 hover:border-cyber-neonGreen/40';
      case 'red':
        return 'shadow-glass-red border-cyber-neonRed/20 hover:border-cyber-neonRed/40';
      case 'blue':
        return 'shadow-glow-blue/10 border-cyber-neonBlue/20 hover:border-cyber-neonBlue/40';
      default:
        return 'border-cyber-border hover:border-cyber-border/80';
    }
  };

  const getSpotlightClass = () => {
    switch (glowColor) {
      case 'green':
        return 'glow-spotlight-green';
      case 'red':
        return 'glow-spotlight-red';
      default:
        return 'bg-gradient-to-br from-white/[0.01] to-transparent';
    }
  };

  return (
    <div
      id={id}
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onClick={onClick}
      className={`
        relative overflow-hidden rounded-xl border glass-panel transition-all duration-300
        ${getGlowStyles()}
        ${hoverable ? 'hover:translate-y-[-2px] duration-200' : ''}
        ${onClick ? 'cursor-pointer' : ''}
        ${className}
      `}
    >
      {/* Dynamic spotlight hover glow background */}
      <div 
        className={`absolute inset-0 opacity-100 pointer-events-none transition-opacity duration-300 ${getSpotlightClass()}`} 
      />
      
      {/* Corner crosshairs for cyber military HUD vibe */}
      <div className="absolute top-0 left-0 w-1.5 h-1.5 border-t border-l border-slate-500/30 pointer-events-none" />
      <div className="absolute top-0 right-0 w-1.5 h-1.5 border-t border-r border-slate-500/30 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-1.5 h-1.5 border-b border-l border-slate-500/30 pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-1.5 h-1.5 border-b border-r border-slate-500/30 pointer-events-none" />

      {/* Content wrapper */}
      <div className="relative z-10 w-full h-full">
        {children}
      </div>
    </div>
  );
};
