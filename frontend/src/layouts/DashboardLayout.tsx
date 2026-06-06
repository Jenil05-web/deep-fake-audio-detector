import React, { useEffect, useState } from 'react';
import { NeuralNetworkBg } from '../components/NeuralNetworkBg';
import { RecentScans } from '../components/RecentScans';
import type { ScanHistoryItem } from '../types';
import { checkHealth } from '../services/api';
import { Activity, ShieldCheck, Wifi, WifiOff } from 'lucide-react';

interface DashboardLayoutProps {
  children: React.ReactNode;
  history: ScanHistoryItem[];
  onSelectScan: (jobId: string) => void;
  onClearHistory: () => void;
  activeJobId?: string;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  history,
  onSelectScan,
  onClearHistory,
  activeJobId
}) => {
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  
  // Periodically check API health
  useEffect(() => {
    let active = true;

    const check = async () => {
      const health = await checkHealth();
      if (!active) return;
      if (health.online && health.modelLoaded) {
        setApiStatus('online');
      } else {
        setApiStatus('offline');
      }
    };

    check();
    const interval = setInterval(check, 8000); // Check every 8s

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="relative min-h-screen flex flex-col bg-cyber-bg text-cyber-text crt-effect cyber-grid overflow-x-hidden">
      {/* Background canvas elements */}
      <NeuralNetworkBg />

      {/* Header */}
      <header className="relative z-20 border-b border-cyber-border/80 bg-cyber-bg/70 backdrop-blur-md px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Logo with animated pulse */}
          <div className="relative flex items-center justify-center w-8 h-8 rounded border border-cyber-neonGreen/30 bg-cyber-neonGreen/5">
            <ShieldCheck className="w-5 h-5 text-cyber-neonGreen neon-glow-green" />
            <span className="absolute -top-0.5 -right-0.5 flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyber-neonGreen opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyber-neonGreen"></span>
            </span>
          </div>
          <div>
            <h1 className="text-lg font-syne font-extrabold tracking-wider leading-none m-0 flex items-center gap-2">
              SPECTRA
              <span className="font-mono text-[9px] font-normal text-cyber-muted tracking-normal border border-cyber-border px-1 rounded">
                v1.0.4-SECURE
              </span>
            </h1>
            <p className="font-mono text-[9px] text-cyber-muted tracking-widest uppercase mt-0.5">
              AI Forensic Audio Investigation Platform
            </p>
          </div>
        </div>

        {/* API connection indicator status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-cyber-muted hidden sm:inline">// CLOUD PIPELINE:</span>
            {apiStatus === 'checking' && (
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-cyber-border bg-[#0d0e14]/50 font-mono text-[9px] text-cyber-muted">
                <Activity className="w-3 h-3 animate-spin" />
                POLLING NETWORK...
              </span>
            )}
            {apiStatus === 'online' && (
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-cyber-neonGreen/20 bg-cyber-neonGreen/5 font-mono text-[9px] text-cyber-neonGreen shadow-glass-green/10">
                <Wifi className="w-3 h-3 animate-pulse" />
                INFRA ONLINE (Ensemble Ready)
              </span>
            )}
            {apiStatus === 'offline' && (
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-cyber-neonRed/20 bg-cyber-neonRed/5 font-mono text-[9px] text-cyber-neonRed shadow-glass-red/10">
                <WifiOff className="w-3 h-3" />
                INFRA OFFLINE (Mock Mode Active)
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 flex relative z-10 w-full overflow-hidden">
        {/* Sidebar logs */}
        <RecentScans
          history={history}
          onSelectScan={onSelectScan}
          onClearHistory={onClearHistory}
          activeJobId={activeJobId}
        />

        {/* Content Area */}
        <main className="flex-1 flex flex-col p-4 md:p-8 overflow-y-auto max-w-7xl mx-auto w-full relative z-10">
          {children}
        </main>
      </div>

      {/* Footer */}
      <footer className="relative z-20 border-t border-cyber-border/80 bg-cyber-bg/85 backdrop-blur-sm px-6 py-3.5 flex flex-col sm:flex-row justify-between items-center gap-2 text-center text-xs">
        <p className="font-mono text-[10px] text-cyber-muted tracking-wide">
          SPECTRA SECURE ENVELOPE (CNN SPECTRAL PATTERNS + LSTM TIMELINE + VOICE BIOMETRICS)
        </p>
        <p className="font-mono text-[9px] text-slate-500 uppercase">
          Authorized Cybercrime Forensic Agency Access Only
        </p>
      </footer>
    </div>
  );
};
