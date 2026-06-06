import React, { useRef, useState, useEffect } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import { UploadCloud, FileAudio, Play, Pause, Trash2 } from 'lucide-react';
import { GlassCard } from './GlassCard';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  onClear: () => void;
  file: File | null;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  onFileSelect,
  onClear,
  file
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [duration, setDuration] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Extract audio metadata (duration) and handle playing states
  useEffect(() => {
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    const audio = new Audio(objectUrl);
    audioRef.current = audio;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
      if (audio.duration) {
        setPlayProgress((audio.currentTime / audio.duration) * 100);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setPlayProgress(0);
      setCurrentTime(0);
    };

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.pause();
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const validateAndProcessFile = (selectedFile: File) => {
    const validExtensions = ['.wav', '.mp3', '.m4a'];
    const extension = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
    
    if (!validExtensions.includes(extension)) {
      alert("Unsupported file format! Please upload WAV, MP3, or M4A files.");
      return;
    }

    const sizeMB = selectedFile.size / (1024 * 1024);
    if (sizeMB > 10) {
      alert("Payload size violation! Audio file cannot exceed 10 MB.");
      return;
    }

    onFileSelect(selectedFile);
  };

  const togglePlayback = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleRemoveFile = () => {
    setDuration(null);
    setIsPlaying(false);
    setPlayProgress(0);
    setCurrentTime(0);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    onClear();
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = 2;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}s`;
  };

  return (
    <div className="w-full">
      {!file ? (
        // Dropzone Layout
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative border-2 border-dashed rounded-xl p-8 md:p-12 text-center cursor-pointer transition-all duration-300
            ${isDragOver 
              ? 'border-cyber-neonGreen bg-cyber-neonGreen/5 shadow-glow-green/10' 
              : 'border-[#1d2230] bg-[#0c0e14]/40 hover:border-[#333e5a] hover:bg-[#0c0e14]/70'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a"
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="relative mx-auto w-16 h-16 rounded-full border border-cyber-border bg-[#0b0c10] flex items-center justify-center mb-6 group-hover:scale-105 transition-transform duration-300">
            <UploadCloud className="w-8 h-8 text-cyber-neonGreen neon-glow-green" />
          </div>

          <h3 className="font-syne font-bold text-lg text-cyber-text tracking-wide mb-2">
            SUBMIT AUDIO PAYLOAD
          </h3>
          <p className="text-sm text-cyber-muted max-w-sm mx-auto mb-6">
            Drag and drop your audio file here, or click to scan directory.
          </p>

          <div className="inline-flex items-center gap-6 px-4 py-2 border border-cyber-border/60 bg-cyber-bg/50 rounded-md font-mono text-[10px] text-cyber-muted">
            <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-cyber-neonGreen" /> WAV</span>
            <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-cyber-neonBlue" /> MP3</span>
            <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-cyber-neonRed" /> M4A</span>
            <span className="text-slate-700">|</span>
            <span>MAX SIZE: 10 MB</span>
          </div>
        </div>
      ) : (
        // Selected File Dashboard Layout
        <GlassCard glowColor="green" hoverable={false} className="p-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-cyber-neonGreen/5 border border-cyber-neonGreen/30 flex items-center justify-center text-cyber-neonGreen shrink-0 neon-glow-green">
                <FileAudio className="w-6 h-6" />
              </div>
              
              <div className="min-w-0">
                <h4 className="font-mono text-sm font-bold text-cyber-text truncate pr-2" title={file.name}>
                  {file.name}
                </h4>
                
                <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1 font-mono text-[10px] text-cyber-muted">
                  <span>SIZE: <strong className="text-slate-400">{formatSize(file.size)}</strong></span>
                  {duration !== null && (
                    <span>DURATION: <strong className="text-slate-400">{formatDuration(duration)}</strong></span>
                  )}
                </div>
              </div>
            </div>

            {/* Custom Audio Player Container */}
            <div className="flex items-center gap-4 bg-[#08090d] border border-cyber-border rounded-lg p-2.5 flex-1 max-w-md">
              <button
                onClick={togglePlayback}
                className="w-8 h-8 rounded-full bg-cyber-neonGreen/10 border border-cyber-neonGreen/30 text-cyber-neonGreen flex items-center justify-center hover:bg-cyber-neonGreen/20 active:scale-95 transition-all duration-200"
                title={isPlaying ? "Pause audio" : "Play audio preview"}
              >
                {isPlaying ? <Pause className="w-4 h-4 fill-cyber-neonGreen" /> : <Play className="w-4 h-4 fill-cyber-neonGreen ml-0.5" />}
              </button>

              <div className="flex-1">
                {/* Track timeline progress */}
                <div className="relative h-1.5 bg-cyber-border rounded-full overflow-hidden">
                  <div 
                    className="absolute left-0 top-0 h-full bg-cyber-neonGreen shadow-glow-green/40 rounded-full transition-all duration-100 ease-linear"
                    style={{ width: `${playProgress}%` }}
                  />
                </div>
                
                <div className="flex justify-between items-center mt-1.5 font-mono text-[9px] text-cyber-muted">
                  <span>
                    {formatDuration(currentTime)}
                  </span>
                  <span>{duration ? formatDuration(duration) : '0:00s'}</span>
                </div>
              </div>

              <button
                onClick={handleRemoveFile}
                className="p-1.5 text-slate-500 hover:text-cyber-neonRed hover:bg-cyber-neonRed/5 rounded transition-all duration-200"
                title="Remove and select another file"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        </GlassCard>
      )}
    </div>
  );
};
