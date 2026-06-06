import React from 'react';
import type { ScanHistoryItem } from '../types';
import { Shield, ShieldAlert, History, Trash2, Clock } from 'lucide-react';

interface RecentScansProps {
  history: ScanHistoryItem[];
  onSelectScan: (jobId: string) => void;
  onClearHistory: () => void;
  activeJobId?: string;
}

export const RecentScans: React.FC<RecentScansProps> = ({
  history,
  onSelectScan,
  onClearHistory,
  activeJobId
}) => {
  const formatTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return '00:00:00';
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0a0a0f]/80 border-r border-cyber-border w-[280px] shrink-0 overflow-y-auto hidden lg:flex glass-panel select-none">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-cyber-border">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-cyber-neonBlue animate-pulse" />
          <span className="font-syne font-bold text-sm tracking-wide text-cyber-text">FORENSIC LOGS</span>
        </div>
        {history.length > 0 && (
          <button
            onClick={onClearHistory}
            className="p-1 text-slate-500 hover:text-cyber-neonRed transition-colors duration-200"
            title="Clear all logs"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* History Items */}
      <div className="flex-1 p-3 space-y-2">
        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[200px] text-center p-4">
            <div className="w-8 h-8 rounded-full border border-dashed border-cyber-border flex items-center justify-center mb-3">
              <Clock className="w-4 h-4 text-cyber-muted" />
            </div>
            <span className="font-mono text-[10px] text-cyber-muted tracking-wider uppercase">NO SCAN RECORDED</span>
            <p className="text-[10px] text-slate-600 mt-1 max-w-[180px]">Upload an audio sample to initialize the local analysis pipeline.</p>
          </div>
        ) : (
          history.map((item) => {
            const isSelected = activeJobId === item.jobId;
            const isFake = item.verdict === 'FAKE';
            
            return (
              <div
                key={item.jobId}
                onClick={() => onSelectScan(item.jobId)}
                className={`
                  p-3 rounded-lg border cursor-pointer transition-all duration-200 group relative overflow-hidden
                  ${isSelected 
                    ? 'bg-cyber-panel/90 border-[#00e5ff]/40 shadow-glow-blue/5' 
                    : 'bg-[#0c0d12]/40 border-cyber-border/80 hover:bg-[#0c0d12]/80 hover:border-slate-700/60'
                  }
                `}
              >
                {/* Accent glow line on active */}
                {isSelected && (
                  <div className="absolute left-0 top-0 bottom-0 w-[3px] bg-cyber-neonBlue" />
                )}

                <div className="flex items-start justify-between gap-1 mb-1">
                  <span className="font-mono text-[11px] font-bold text-cyber-text truncate flex-1 block pr-2" title={item.filename}>
                    {item.filename}
                  </span>
                  
                  {isFake ? (
                    <ShieldAlert className="w-3.5 h-3.5 text-cyber-neonRed shrink-0 neon-glow-red" />
                  ) : (
                    <Shield className="w-3.5 h-3.5 text-cyber-neonGreen shrink-0 neon-glow-green" />
                  )}
                </div>

                <div className="flex justify-between items-center text-[10px] font-mono text-cyber-muted mt-2">
                  <div className="flex items-center gap-1.5">
                    <span className={`
                      font-bold px-1.5 py-0.5 rounded-[2px] text-[8px] tracking-wider
                      ${isFake 
                        ? 'bg-cyber-neonRed/10 text-cyber-neonRed border border-cyber-neonRed/20' 
                        : 'bg-cyber-neonGreen/10 text-cyber-neonGreen border border-cyber-neonGreen/20'
                      }
                    `}>
                      {item.verdict}
                    </span>
                    <span>{item.confidence.toFixed(1)}%</span>
                  </div>
                  <span>{formatTime(item.timestamp)}</span>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Database footer */}
      <div className="p-3 bg-[#07080c] border-t border-cyber-border text-center">
        <span className="font-mono text-[8.5px] text-cyber-muted tracking-widest uppercase">
          SECURE STORAGE ACTIVE
        </span>
      </div>
    </div>
  );
};
