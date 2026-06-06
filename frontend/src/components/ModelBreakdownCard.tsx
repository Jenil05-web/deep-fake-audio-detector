import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { GlassCard } from './GlassCard';

interface ModelBreakdownCardProps {
  name: string;
  score: number; // 0 - 100
  description: string;
  color?: 'green' | 'red' | 'blue';
  delay?: number;
}

export const ModelBreakdownCard: React.FC<ModelBreakdownCardProps> = ({
  name,
  score,
  description,
  color = 'green',
  delay = 0
}) => {
  const [animatedScore, setAnimatedScore] = useState(0);
  
  // Circle Math
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;

  useEffect(() => {
    // Score count up animation
    const duration = 1200; // ms
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing out quadratic
      const easeProgress = progress * (2 - progress);
      setAnimatedScore(easeProgress * score);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    const delayTimeout = setTimeout(() => {
      requestAnimationFrame(animate);
    }, delay * 1000);

    return () => clearTimeout(delayTimeout);
  }, [score, delay]);

  const getColorClasses = () => {
    switch (color) {
      case 'red':
        return {
          stroke: 'stroke-cyber-neonRed',
          text: 'text-cyber-neonRed',
          glow: 'shadow-glow-red/20',
          radial: 'red' as const,
        };
      case 'blue':
        return {
          stroke: 'stroke-cyber-neonBlue',
          text: 'text-cyber-neonBlue',
          glow: 'shadow-glow-blue/20',
          radial: 'blue' as const,
        };
      default:
        return {
          stroke: 'stroke-cyber-neonGreen',
          text: 'text-cyber-neonGreen',
          glow: 'shadow-glow-green/20',
          radial: 'green' as const,
        };
    }
  };

  const style = getColorClasses();

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
    >
      <GlassCard 
        glowColor={style.radial}
        className="p-5 flex items-center justify-between min-h-[110px]"
      >
        <div className="flex-1 pr-4">
          <div className="font-mono text-xs text-cyber-muted tracking-wider mb-1">// {name.toUpperCase()}</div>
          <h4 className="text-md font-syne font-bold text-cyber-text tracking-tight mb-1">{name}</h4>
          <p className="text-xs text-cyber-muted font-mono">{description}</p>
        </div>

        {/* Circular Progress Gauge */}
        <div className="relative flex items-center justify-center w-[72px] h-[72px]">
          <svg className="w-full h-full transform -rotate-90">
            {/* Background Circle */}
            <circle
              cx="36"
              cy="36"
              r={radius}
              className="stroke-[#131620] fill-none"
              strokeWidth="4"
            />
            {/* Foreground Progress Circle */}
            <motion.circle
              cx="36"
              cy="36"
              r={radius}
              className={`fill-none transition-all duration-300 ${style.stroke}`}
              strokeWidth="4"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              style={{
                filter: `drop-shadow(0 0 4px var(--tw-shadow-color, currentColor))`
              }}
            />
          </svg>
          
          {/* Centered Score percentage */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-[12px] font-mono font-bold ${style.text}`}>
              {animatedScore.toFixed(1)}%
            </span>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
};
