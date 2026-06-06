import React, { useState, useEffect } from 'react';
import type { NormalizedResult } from '../types';
import { exportPdfReport } from '../utils/pdf';
import { GlassCard } from '../components/GlassCard';
import { ModelBreakdownCard } from '../components/ModelBreakdownCard';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  PieChart, Pie, Cell
} from 'recharts';
import { 
  Download, Share2, ArrowLeft, Check, 
  ShieldAlert, ShieldCheck, Clock, FileAudio, Cpu, FileJson, AlertCircle, RefreshCw 
} from 'lucide-react';
import confetti from 'canvas-confetti';

interface ResultsPageProps {
  result: NormalizedResult;
  onBack: () => void;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({ result, onBack }) => {
  const [copiedJson, setCopiedJson] = useState(false);
  const [copiedShare, setCopiedShare] = useState(false);
  const [isExportingPdf, setIsExportingPdf] = useState(false);

  const isFake = result.verdict === 'FAKE';
  
  // Confetti effect on REAL verdict to celebrate authenticity
  useEffect(() => {
    if (!isFake) {
      confetti({
        particleCount: 80,
        spread: 60,
        origin: { y: 0.7 },
        colors: ['#00ff88', '#00e5ff', '#ffffff']
      });
    }
  }, [isFake]);

  const handleCopyJson = async () => {
    try {
      await navigator.clipboard.writeText(result.rawJson);
      setCopiedJson(true);
      setTimeout(() => setCopiedJson(false), 2000);
    } catch (err) {
      console.error("Failed to copy JSON:", err);
    }
  };

  const handleShareResult = async () => {
    const summary = `[SPECTRA AI FORENSIC REPORT]
File: ${result.filename}
Verdict: ${result.verdict}
Confidence: ${result.confidence}%
CNN Analyzer: ${result.cnnScore}%
LSTM Sequence: ${result.lstmScore}%
Biometrics: ${result.bioScore}%
Duration: ${result.duration}s
Inference Time: ${result.inferenceTime}ms
Verification Status: SIGNED_AND_VALIDATED`;

    try {
      await navigator.clipboard.writeText(summary);
      setCopiedShare(true);
      setTimeout(() => setCopiedShare(false), 2000);
    } catch (err) {
      console.error("Failed to copy share summary:", err);
    }
  };

  const handleExportPdf = async () => {
    setIsExportingPdf(true);
    // Let DOM update to hide buttons before capture
    await new Promise(r => setTimeout(r, 300));
    const success = await exportPdfReport('forensic-dashboard', result.filename);
    setIsExportingPdf(false);
    if (!success) {
      alert("PDF Export failed! Ensure the page is fully loaded.");
    }
  };

  // 1. Data for Radar Chart (Decision Boundary)
  const radarData = [
    { subject: 'Spectral (CNN)', score: result.cnnScore, fullMark: 100 },
    { subject: 'Temporal (LSTM)', score: result.lstmScore, fullMark: 100 },
    { subject: 'Biometrics (BIO)', score: result.bioScore, fullMark: 100 },
  ];

  // 2. Data for Bar Chart (Comparative Breakdown)
  const barData = [
    { name: 'CNN', Score: result.cnnScore },
    { name: 'LSTM', Score: result.lstmScore },
    { name: 'Biometrics', Score: result.bioScore },
  ];

  // 3. Data for Semicircular Risk Gauge
  // Slices: [Value, Rest]
  const gaugeData = [
    { name: 'Risk', value: result.confidence },
    { name: 'Buffer', value: 100 - result.confidence },
  ];

  const gaugeColors = isFake 
    ? ['#ff3366', '#121520'] // Crimson Red vs Dark Space
    : ['#00ff88', '#121520']; // Neon Green vs Dark Space

  return (
    <div className="w-full flex flex-col space-y-6">
      
      {/* Navigation Top Header */}
      <div className="flex items-center justify-between pb-2 border-b border-cyber-border/40">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 font-mono text-xs text-cyber-muted hover:text-cyber-neonGreen transition-colors duration-200"
        >
          <ArrowLeft className="w-4 h-4" />
          BACK TO PIPELINE
        </button>

        <div className="flex items-center gap-3">
          <button
            onClick={handleCopyJson}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-cyber-border bg-cyber-panel/60 hover:bg-slate-800/80 font-mono text-[10px] text-cyber-text transition-all duration-200"
            title="Copy API Raw JSON"
          >
            {copiedJson ? <Check className="w-3.5 h-3.5 text-cyber-neonGreen" /> : <FileJson className="w-3.5 h-3.5" />}
            {copiedJson ? 'COPIED JSON' : 'COPY RAW JSON'}
          </button>
          
          <button
            onClick={handleShareResult}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-cyber-border bg-cyber-panel/60 hover:bg-slate-800/80 font-mono text-[10px] text-cyber-text transition-all duration-200"
            title="Copy Text Summary"
          >
            {copiedShare ? <Check className="w-3.5 h-3.5 text-cyber-neonGreen" /> : <Share2 className="w-3.5 h-3.5" />}
            {copiedShare ? 'SUMMARY COPIED' : 'SHARE RESULTS'}
          </button>

          <button
            onClick={handleExportPdf}
            disabled={isExportingPdf}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded bg-cyber-neonGreen text-black font-syne font-bold text-[10px] hover:bg-black hover:text-cyber-neonGreen border border-cyber-neonGreen hover:shadow-glow-green/10 transition-all duration-200 disabled:opacity-50"
          >
            {isExportingPdf ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                EXPORTING...
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                DOWNLOAD PDF REPORT
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Forensic Dashboard Container (A4 Printable Target) */}
      <div 
        id="forensic-dashboard" 
        className="p-6 md:p-8 rounded-xl border border-cyber-border bg-cyber-bg/95 relative overflow-hidden"
      >
        {/* Glow ambient background lights */}
        <div className={`absolute top-0 right-0 w-[200px] h-[200px] rounded-full filter blur-[120px] opacity-10 pointer-events-none ${isFake ? 'bg-cyber-neonRed' : 'bg-cyber-neonGreen'}`} />
        <div className="absolute bottom-0 left-0 w-[200px] h-[200px] rounded-full filter blur-[120px] opacity-5 pointer-events-none bg-cyber-neonBlue" />

        {/* Dashboard Title Block for PDF Exporter */}
        <div className="mb-8 border-b border-cyber-border pb-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="font-mono text-[10px] text-cyber-neonBlue tracking-widest uppercase mb-1">// FORENSIC AUDIT RECORD</div>
            <h3 className="font-syne font-extrabold text-2xl text-cyber-text">AI Synthetic Audio Classifications</h3>
            <p className="font-mono text-xs text-cyber-muted mt-0.5">JOB ID: <span className="text-slate-400 font-bold">{result.jobId}</span></p>
          </div>

          <div className="font-mono text-right text-[10px] text-cyber-muted space-y-0.5">
            <p>AUDIT DATE: {new Date(result.timestamp).toLocaleDateString()} {new Date(result.timestamp).toLocaleTimeString()}</p>
            <p>INTEGRITY PIPELINE: SIGNED // SECURE</p>
          </div>
        </div>

        {/* Top Segment: Verdict and Semicircular Gauge */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 mb-8">
          
          {/* Main Verdict Glowing Card */}
          <div className="md:col-span-8 h-full">
            <GlassCard
              glowColor={isFake ? 'red' : 'green'}
              hoverable={false}
              className="p-6 h-full flex flex-col justify-between"
            >
              <div>
                <span className="font-mono text-xs text-cyber-muted tracking-wider block mb-3">// PIPELINE CLASSIFIER VERDICT</span>
                
                <div className="flex items-center gap-4">
                  <div className={`w-14 h-14 rounded-xl flex items-center justify-center border shrink-0 ${
                    isFake 
                      ? 'bg-cyber-neonRed/10 border-cyber-neonRed/35 text-cyber-neonRed neon-glow-red' 
                      : 'bg-cyber-neonGreen/10 border-cyber-neonGreen/35 text-cyber-neonGreen neon-glow-green'
                  }`}>
                    {isFake ? <ShieldAlert className="w-8 h-8" /> : <ShieldCheck className="w-8 h-8" />}
                  </div>

                  <div>
                    {/* Big glowing Badge text */}
                    <h2 className={`text-4xl md:text-5xl font-syne font-black tracking-wider leading-none uppercase ${
                      isFake ? 'text-cyber-neonRed neon-glow-red' : 'text-cyber-neonGreen neon-glow-green'
                    }`}>
                      {result.verdict}
                    </h2>
                    <p className="font-mono text-xs text-cyber-muted mt-1">
                      {isFake 
                        ? `The uploaded voice signature displays cloning patterns with a ${result.confidence}% probability of synthesis.` 
                        : `Vocal biometrics indicate natural speech dynamics. Genuine human verification approved.`
                      }
                    </p>
                  </div>
                </div>
              </div>

              {/* Summary Description block */}
              <div className="mt-8 pt-4 border-t border-cyber-border flex justify-between items-center font-mono text-[10px] text-cyber-muted">
                <div className="flex items-center gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5 text-cyber-neonBlue" />
                  <span>Ensemble Confidence Level:</span>
                  <strong className={isFake ? 'text-cyber-neonRed' : 'text-cyber-neonGreen'}>{result.confidence}%</strong>
                </div>
                <div>
                  <span>Status: </span>
                  <span className="text-cyber-neonGreen">AUDIT_DONE</span>
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Semicircular Risk Meter Gauge */}
          <div className="md:col-span-4 h-full">
            <GlassCard glowColor={isFake ? 'red' : 'green'} hoverable={false} className="p-5 h-full flex flex-col justify-between items-center text-center">
              <span className="font-mono text-xs text-cyber-muted tracking-wider w-full text-left self-start">// RISK INDEX</span>
              
              <div className="relative w-full h-[120px] flex items-center justify-center mt-3 overflow-hidden">
                <ResponsiveContainer width="100%" height={240}>
                  <PieChart>
                    <Pie
                      data={gaugeData}
                      dataKey="value"
                      cx="50%"
                      cy="50%"
                      startAngle={180}
                      endAngle={0}
                      innerRadius={55}
                      outerRadius={75}
                      paddingAngle={0}
                    >
                      {gaugeData.map((_entry, index) => (
                        <Cell key={`cell-${index}`} fill={gaugeColors[index]} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>

                {/* Semicircular gauge inner readout text */}
                <div className="absolute bottom-2 flex flex-col items-center justify-center">
                  <span className={`text-2xl font-mono font-bold leading-none ${isFake ? 'text-cyber-neonRed' : 'text-cyber-neonGreen'}`}>
                    {result.confidence}%
                  </span>
                  <span className="text-[9px] font-mono text-cyber-muted tracking-wider uppercase mt-1">
                    {isFake ? 'Synthetic Risk' : 'Authentic Match'}
                  </span>
                </div>
              </div>

              <span className="font-mono text-[9px] text-slate-500 uppercase">
                {isFake ? "🔴 High deepfake signature detected" : "🟢 Human voice verification matched"}
              </span>
            </GlassCard>
          </div>

        </div>

        {/* Model Breakdown Title */}
        <div className="mb-4">
          <span className="font-mono text-xs text-cyber-muted tracking-wider">// ENSEMBLE CLASSIFIERS READINGS</span>
        </div>

        {/* Models Circle progress cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <ModelBreakdownCard
            name="CNN Spectral Model"
            score={result.cnnScore}
            description="Analyzes acoustic spectrogram layouts for fake frequency discrepancies."
            color={result.cnnScore > 65 ? 'red' : 'green'}
            delay={0.1}
          />
          <ModelBreakdownCard
            name="LSTM Sequence Model"
            score={result.lstmScore}
            description="Examines temporal audio frames and sequence flow patterns."
            color={result.lstmScore > 65 ? 'red' : 'green'}
            delay={0.2}
          />
          <ModelBreakdownCard
            name="Biometrics MLP"
            score={result.bioScore}
            description="Validates vocal tract structure and biological voice dynamics."
            color={result.bioScore > 65 ? 'red' : 'green'}
            delay={0.3}
          />
        </div>

        {/* Analytics Section: Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="p-4 rounded-lg bg-[#0c0d12]/60 border border-cyber-border font-mono">
            <div className="text-[10px] text-cyber-muted flex items-center gap-1.5">
              <FileAudio className="w-3.5 h-3.5 text-cyber-neonBlue" /> AUDIO SOURCE
            </div>
            <p className="text-sm font-bold text-cyber-text truncate mt-2" title={result.filename}>
              {result.filename}
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#0c0d12]/60 border border-cyber-border font-mono">
            <div className="text-[10px] text-cyber-muted flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-cyber-neonGreen" /> DURATION
            </div>
            <p className="text-sm font-bold text-cyber-text mt-2">
              {result.duration.toFixed(2)} sec
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#0c0d12]/60 border border-[#181b24] font-mono">
            <div className="text-[10px] text-cyber-muted flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyber-neonRed" /> INFERENCE TIME
            </div>
            <p className="text-sm font-bold text-cyber-text mt-2">
              {result.inferenceTime} ms
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#0c0d12]/60 border border-cyber-border font-mono">
            <div className="text-[10px] text-cyber-muted flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 text-cyber-neonBlue" /> THRESHOLD INDEX
            </div>
            <p className="text-sm font-bold text-cyber-text mt-2">
              65.0% (Ensemble)
            </p>
          </div>
        </div>

        {/* Data Visualization Section: Recharts charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Bar Chart comparing CNN vs LSTM vs BIO */}
          <GlassCard hoverable={false} className="p-5">
            <span className="font-mono text-xs text-cyber-muted tracking-wider block mb-4">// DETECTOR WEIGHT COMPARATIVE ANALYSIS</span>
            
            <div className="w-full h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={barData}
                  margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                >
                  <XAxis 
                    dataKey="name" 
                    stroke="rgba(100, 116, 139, 0.6)" 
                    tick={{ fontFamily: 'Space Mono', fontSize: 10 }}
                  />
                  <YAxis 
                    stroke="rgba(100, 116, 139, 0.6)" 
                    tick={{ fontFamily: 'Space Mono', fontSize: 10 }}
                    domain={[0, 100]}
                  />
                  <Tooltip
                    contentStyle={{ 
                      backgroundColor: '#0a0a0f', 
                      borderColor: '#181b24',
                      borderRadius: '8px',
                      fontFamily: 'Space Mono',
                      fontSize: '11px',
                    }}
                    cursor={{ fill: 'rgba(255, 255, 255, 0.02)' }}
                  />
                  <Bar 
                    dataKey="Score" 
                    fill={isFake ? '#ff3366' : '#00ff88'} 
                    radius={[4, 4, 0, 0]}
                    maxBarSize={48}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>

          {/* Radar Decision Boundary Chart */}
          <GlassCard hoverable={false} className="p-5">
            <span className="font-mono text-xs text-cyber-muted tracking-wider block mb-4">// ENSEMBLE DECISION RADIAL PATH</span>
            
            <div className="w-full h-[220px] flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                  <PolarGrid stroke="rgba(100, 116, 139, 0.15)" />
                  <PolarAngleAxis 
                    dataKey="subject" 
                    stroke="rgba(148, 163, 184, 0.8)" 
                    tick={{ fontFamily: 'Space Mono', fontSize: 8.5 }}
                  />
                  <PolarRadiusAxis 
                    angle={30} 
                    domain={[0, 100]} 
                    stroke="rgba(100, 116, 139, 0.4)"
                    tick={{ fontFamily: 'Space Mono', fontSize: 8 }}
                  />
                  <Radar
                    name="Classifier Probability"
                    dataKey="score"
                    stroke={isFake ? '#ff3366' : '#00e5ff'}
                    fill={isFake ? '#ff3366' : '#00e5ff'}
                    fillOpacity={0.25}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>

        </div>

      </div>

    </div>
  );
};
