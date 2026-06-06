import { useState } from 'react';
import { DashboardLayout } from './layouts/DashboardLayout';
import { LandingPage } from './pages/LandingPage';
import { ResultsPage } from './pages/ResultsPage';
import { useScanHistory } from './hooks/useScanHistory';
import type { NormalizedResult } from './types';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [activeResult, setActiveResult] = useState<NormalizedResult | null>(null);
  const { history, addScan, clearHistory } = useScanHistory();

  const handleAnalysisSuccess = (result: NormalizedResult) => {
    setActiveResult(result);
    addScan(result);
  };

  const handleSelectScan = (jobId: string) => {
    const matched = history.find(item => item.jobId === jobId);
    if (matched) {
      setActiveResult(matched);
    }
  };

  const handleBack = () => {
    setActiveResult(null);
  };

  return (
    <DashboardLayout
      history={history}
      onSelectScan={handleSelectScan}
      onClearHistory={clearHistory}
      activeJobId={activeResult?.jobId}
    >
      <AnimatePresence mode="wait">
        {activeResult ? (
          <motion.div
            key="results"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.4 }}
            className="w-full"
          >
            <ResultsPage 
              result={activeResult} 
              onBack={handleBack} 
            />
          </motion.div>
        ) : (
          <motion.div
            key="landing"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.4 }}
            className="w-full"
          >
            <LandingPage 
              onAnalysisSuccess={handleAnalysisSuccess} 
            />
          </motion.div>
        )}
      </AnimatePresence>
    </DashboardLayout>
  );
}

export default App;
