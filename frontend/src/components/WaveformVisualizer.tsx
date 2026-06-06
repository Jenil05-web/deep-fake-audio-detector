import React, { useEffect, useRef, useState } from 'react';

interface WaveformVisualizerProps {
  file: File | null;
  isScanning: boolean;
  verdict?: "REAL" | "FAKE" | null;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({ 
  file, 
  isScanning,
  verdict 
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [decoding, setDecoding] = useState(true);
  const [peaks, setPeaks] = useState<number[]>([]);
  const scanPositionRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);

  // Decode audio data using Web Audio API
  useEffect(() => {
    if (!file) return;

    let isCancelled = false;

    const decodeAudio = async () => {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const audioCtx = new AudioContextClass();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0); // Left channel

        // Downsample to 200 points
        const sampleSize = 200;
        const blockSize = Math.floor(channelData.length / sampleSize);
        const computedPeaks: number[] = [];

        for (let i = 0; i < sampleSize; i++) {
          const start = i * blockSize;
          let max = 0;
          for (let j = 0; j < blockSize; j++) {
            const val = Math.abs(channelData[start + j] || 0);
            if (val > max) max = val;
          }
          computedPeaks.push(max);
        }

        // Normalize peaks
        const maxPeak = Math.max(...computedPeaks, 0.01);
        const normalized = computedPeaks.map(p => p / maxPeak);

        if (!isCancelled) {
          setPeaks(normalized);
          setDecoding(false);
        }
      } catch (err) {
        console.warn("Failed to decode audio file, falling back to synthetic waveform", err);
        // Fallback synthetic wave
        const syntheticPeaks: number[] = [];
        for (let i = 0; i < 200; i++) {
          // Combination of sine waves and noise to look like actual speech
          const val = Math.abs(
            Math.sin(i * 0.05) * 0.4 +
            Math.sin(i * 0.2) * 0.3 +
            Math.cos(i * 0.01) * 0.2 +
            (Math.random() - 0.5) * 0.1
          );
          syntheticPeaks.push(val);
        }
        const maxSynthetic = Math.max(...syntheticPeaks, 0.01);
        const normalized = syntheticPeaks.map(p => p / maxSynthetic);

        if (!isCancelled) {
          setPeaks(normalized);
          setDecoding(false);
        }
      }
    };

    decodeAudio();

    return () => {
      isCancelled = true;
    };
  }, [file]);

  // Handle draw loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    let scanProgress = 0;

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw background noise/grid in the waveform card
      ctx.strokeStyle = 'rgba(24, 27, 36, 0.4)';
      ctx.lineWidth = 0.5;
      for (let x = 10; x < width; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 10; y < height; y += 15) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      if (peaks.length === 0) {
        // Drawing load/placeholder state
        ctx.font = '10px "Space Mono"';
        ctx.fillStyle = 'rgba(100, 116, 139, 0.6)';
        ctx.textAlign = 'center';
        ctx.fillText("PRE-ANALYZING SPECTRUM...", width / 2, height / 2 + 3);
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }

      // Draw Waveform bars
      const barWidth = Math.max(1.5, (width / peaks.length) - 1.5);
      const barSpacing = (width / peaks.length);

      peaks.forEach((peak, i) => {
        const x = i * barSpacing;
        const barHeight = Math.max(3, peak * (height - 16));
        const y = (height - barHeight) / 2;

        const isScanned = isScanning && (x / width) < scanProgress;
        
        // Dynamic colors: scanning active vs static display
        const barColor = isScanning
          ? (isScanned ? 'rgba(0, 229, 255, 0.8)' : 'rgba(100, 116, 139, 0.35)')
          : verdict === 'FAKE'
            ? 'rgba(255, 51, 102, 0.7)'
            : verdict === 'REAL'
              ? 'rgba(0, 255, 136, 0.7)'
              : 'rgba(0, 255, 136, 0.55)';

        ctx.fillStyle = barColor;
        ctx.fillRect(x, y, barWidth, barHeight);
      });

      // Draw active scanner bar & laser glow
      if (isScanning) {
        scanProgress += 0.0035; // speed of scanner line
        if (scanProgress > 1) {
          scanProgress = 0; // loop
        }
        scanPositionRef.current = scanProgress;

        const scanX = scanProgress * width;

        // Draw laser scanning line
        ctx.shadowColor = '#00e5ff';
        ctx.shadowBlur = 10;
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(scanX, 0);
        ctx.lineTo(scanX, height);
        ctx.stroke();

        // Scanning gradient trail
        const trailWidth = 40;
        const grad = ctx.createLinearGradient(scanX - trailWidth, 0, scanX, 0);
        grad.addColorStop(0, 'rgba(0, 229, 255, 0)');
        grad.addColorStop(1, 'rgba(0, 229, 255, 0.08)');
        ctx.fillStyle = grad;
        ctx.fillRect(scanX - trailWidth, 0, trailWidth, height);

        ctx.shadowBlur = 0; // reset
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [peaks, isScanning, verdict]);

  return (
    <div className="relative w-full">
      <div className="flex justify-between items-center mb-2 px-1">
        <span className="font-mono text-xs text-cyber-muted tracking-wider">
          {isScanning 
            ? "CRITICAL SCANNING OPERATION IN PROGRESS" 
            : decoding 
              ? "DECODING AUDIO CHANNEL..." 
              : "// AUDIO SPECTRAL SIGNATURE"
          }
        </span>
        <span className="font-mono text-[10px] text-cyber-neonGreen opacity-80 uppercase">
          {decoding ? "Wait" : "Signature Loaded"}
        </span>
      </div>
      <div className="relative w-full h-[96px] bg-[#090b10] border border-[#1d2230] rounded-sm p-3 overflow-hidden glass-panel">
        {/* Radar grids */}
        <div className="absolute top-0 right-0 w-[40px] h-full border-l border-cyber-border border-dashed opacity-20 pointer-events-none" />
        <div className="absolute top-0 left-[20%] w-[1px] h-full border-r border-cyber-border border-dashed opacity-25 pointer-events-none" />
        <div className="absolute top-0 left-[50%] w-[1px] h-full border-r border-cyber-border border-dashed opacity-25 pointer-events-none" />
        <div className="absolute top-0 left-[80%] w-[1px] h-full border-r border-cyber-border border-dashed opacity-25 pointer-events-none" />

        <canvas
          ref={canvasRef}
          className="w-full h-full block"
        />

        {/* Scan effect lines */}
        {isScanning && (
          <div className="absolute inset-0 bg-cyan-500/5 animate-pulse-slow pointer-events-none" />
        )}
      </div>
    </div>
  );
};
