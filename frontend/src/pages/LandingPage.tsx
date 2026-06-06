import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadZone } from '../components/UploadZone';
import { WaveformVisualizer } from '../components/WaveformVisualizer';
import { analyzeAudio } from '../services/api';
import type { NormalizedResult } from '../types';
import { Activity, Cpu, AlertTriangle, RefreshCw } from 'lucide-react';
import { GlassCard } from '../components/GlassCard';

interface LandingPageProps {
  onAnalysisSuccess: (result: NormalizedResult) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onAnalysisSuccess }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanMessage, setScanMessage] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setErrorMsg(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setIsScanning(false);
    setErrorMsg(null);
  };

  const startAnalysis = async () => {
    if (!selectedFile) return;

    setIsScanning(true);
    setErrorMsg(null);

    try {
      const result = await analyzeAudio(selectedFile, (progressMessage) => {
        setScanMessage(progressMessage);
      });
      onAnalysisSuccess(result);
    } catch (err: unknown) {
      console.error(err);
      const error = err as Error;
      setErrorMsg(error.message || 'An unknown network error occurred during audio processing.');
      setIsScanning(false);
    }
  };

  return (
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto py-4">
      
      {/* Hero Header Section */}
      <div className="text-center mb-8 md:mb-12">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <span className="font-mono text-xs text-cyber-neonGreen tracking-[0.25em] uppercase bg-cyber-neonGreen/5 border border-cyber-neonGreen/10 px-3 py-1 rounded-full shadow-glass-green/5 inline-block mb-4">
            // VOICE AUTHENTICATION PIPELINE
          </span>
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="text-4xl md:text-5xl lg:text-6xl font-syne font-extrabold tracking-tight mb-4 text-cyber-text"
        >
          DeepFake Audio Detection
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.25 }}
          className="text-sm md:text-md text-cyber-muted max-w-xl mx-auto leading-relaxed"
        >
          Analyze speech authenticity using ensemble AI models powered by CNN, LSTM, and Voice Biometrics. Check for cloning anomalies and synthesized speech profiles.
        </motion.p>
      </div>

      {/* Main Upload Area */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.35 }}
        className="w-full space-y-6"
      >
        <UploadZone
          file={selectedFile}
          onFileSelect={handleFileSelect}
          onClear={handleClear}
        />

        {/* Waveform display - loaded when file is selected */}
        {selectedFile && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <WaveformVisualizer 
              key={selectedFile.name}
              file={selectedFile} 
              isScanning={isScanning} 
            />
          </motion.div>
        )}

        {/* Action Controls / Scanners */}
        <AnimatePresence mode="wait">
          {!isScanning ? (
            selectedFile && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex flex-col items-center"
              >
                <span className="font-mono text-[10px] text-cyber-neonGreen tracking-widest uppercase mb-3 animate-pulse">
                  READY FOR ANALYTICAL SCANNING
                </span>
                
                <button
                  onClick={startAnalysis}
                  className="relative group flex items-center gap-3 bg-cyber-neonGreen text-black font-syne font-bold px-8 py-3.5 rounded border border-cyber-neonGreen hover:bg-black hover:text-cyber-neonGreen shadow-glow-green hover:shadow-glow-green/20 transition-all duration-300 transform active:scale-98"
                >
                  <Cpu className="w-5 h-5 group-hover:rotate-45 transition-transform duration-300" />
                  INITIATE CLASSIFIER INFERENCE
                </button>
              </motion.div>
            )
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full"
            >
              {/* Scanning Active Layout */}
              <GlassCard glowColor="blue" hoverable={false} className="p-8 text-center relative overflow-hidden">
                
                {/* Horizontal scanline laser effect */}
                <div className="absolute left-0 right-0 h-0.5 bg-cyber-neonBlue shadow-glow-blue/80 top-0 animate-scanline" />

                {/* Pulsing visual core */}
                <div className="relative mx-auto w-20 h-20 rounded-full border border-cyber-neonBlue/40 bg-cyber-neonBlue/5 flex items-center justify-center mb-6">
                  <Activity className="w-10 h-10 text-cyber-neonBlue animate-pulse" />
                  <div className="absolute inset-0 border border-dashed border-cyber-neonBlue/20 rounded-full animate-[spin_10s_linear_infinite]" />
                  <div className="absolute inset-2 border border-dotted border-cyber-neonBlue/30 rounded-full animate-[spin_15s_linear_infinite]" />
                </div>

                <h3 className="font-syne font-bold text-lg text-cyber-text tracking-wide mb-1">
                  CRITICAL CORE FORENSIC DISPATCH
                </h3>
                
                <p className="font-mono text-sm text-cyber-neonBlue mb-6 tracking-wide uppercase h-6 animate-pulse">
                  {scanMessage || "Preparing Audio Stream..."}
                </p>

                {/* Pseudo status bar */}
                <div className="max-w-md mx-auto bg-cyber-border rounded-full h-1 overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-cyber-neonBlue to-cyber-neonGreen w-[75%] rounded-full animate-[pulse_2s_infinite]" />
                </div>

                <div className="mt-8 font-mono text-[9px] text-cyber-muted space-y-1">
                  <p>CONNECTING CORE SERVER: {import.meta.env.VITE_API_URL || 'http://localhost:8000'}</p>
                  <p>COMPILING MEL-SPECTROGRAM GRADIENTS ... SECURE</p>
                </div>
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Terminal display */}
        {errorMsg && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full"
          >
            <GlassCard glowColor="red" hoverable={false} className="p-5 border-cyber-neonRed/30 bg-cyber-neonRed/5">
              <div className="flex items-start gap-4">
                <div className="p-2 bg-cyber-neonRed/10 border border-cyber-neonRed/30 rounded text-cyber-neonRed">
                  <AlertTriangle className="w-5 h-5 shrink-0" />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-mono text-xs font-bold text-cyber-neonRed tracking-wider uppercase mb-1">
                    PIPELINE DISPATCH FAULT RECORDED
                  </h4>
                  <p className="text-sm text-slate-300 leading-relaxed font-mono">
                    {errorMsg}
                  </p>
                  <p className="text-[10px] text-cyber-muted mt-3 font-mono">
                    The platform will fallback to simulated/mock responses if you choose to retry. Confirm your local API server is active at port 8000.
                  </p>
                  
                  {/* Mock Mode Fallback Button */}
                  <div className="flex gap-3 mt-4">
                    <button
                      onClick={startAnalysis}
                      className="flex items-center gap-1.5 px-3.5 py-1.5 rounded bg-cyber-neonRed/10 border border-cyber-neonRed/20 hover:bg-cyber-neonRed/20 font-mono text-[10px] text-cyber-neonRed font-bold transition-all duration-200"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      RETRY PIPELINE CHECK
                    </button>
                    <button
                      onClick={() => {
                        // Generate mock response for demonstration (National-level AI competition grade support)
                        const mockResult: NormalizedResult = {
                          jobId: `mock-${Math.random().toString(36).substr(2, 9)}`,
                          filename: selectedFile?.name || 'mock_evidence.wav',
                          verdict: Math.random() > 0.4 ? 'FAKE' : 'REAL',
                          confidence: Math.round((70 + Math.random() * 25) * 10) / 10,
                          cnnScore: Math.round((20 + Math.random() * 60) * 10) / 10,
                          lstmScore: Math.round((80 + Math.random() * 20) * 10) / 10,
                          bioScore: Math.round((50 + Math.random() * 50) * 10) / 10,
                          duration: 6.28,
                          inferenceTime: Math.round(5000 + Math.random() * 3000),
                          timestamp: new Date().toISOString(),
                          rawJson: JSON.stringify({
                            mock_pipeline: true,
                            verdict: "FAKE",
                            confidence: 0.731,
                            proba_cnn: 0.206,
                            proba_lstm: 1.00,
                            proba_bio: 1.00,
                            duration_sec: 6.28,
                            inference_ms: 7715.1
                          }, null, 2),
                        };
                        
                        // Fake a brief scan animation for premium UX
                        setIsScanning(true);
                        setScanMessage("Extracting Mock Features...");
                        setTimeout(() => {
                          setScanMessage("Running CNN Analysis...");
                          setTimeout(() => {
                            setScanMessage("Running Voice Biometrics...");
                            setTimeout(() => {
                              onAnalysisSuccess(mockResult);
                            }, 8000); // 800ms
                          }, 800);
                        }, 800);
                      }}
                      className="px-3.5 py-1.5 rounded bg-cyber-border hover:bg-slate-800/80 border border-slate-700/80 font-mono text-[10px] text-cyber-text transition-all duration-200"
                    >
                      FORCE LOCAL EMULATION
                    </button>
                  </div>

                </div>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};
