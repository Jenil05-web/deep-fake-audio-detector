import { useState } from 'react';
import type { ScanHistoryItem, NormalizedResult } from '../types';

const STORAGE_KEY = 'spectra_scan_history';

export function useScanHistory() {
  const [history, setHistory] = useState<ScanHistoryItem[]>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error("Failed to parse scan history:", error);
      return [];
    }
  });

  // Save scan to history
  const addScan = (result: NormalizedResult) => {
    setHistory(prev => {
      // Since ScanHistoryItem is an alias/extends NormalizedResult, we can store the whole object
      const newItem: ScanHistoryItem = { ...result };

      // Filter out if this jobId already exists in history to prevent duplicates
      const filtered = prev.filter(item => item.jobId !== newItem.jobId);
      
      // Place newest item at the top and cap at 10 items
      const updated = [newItem, ...filtered].slice(0, 10);
      
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch (error) {
        console.error("Failed to write scan history:", error);
      }
      
      return updated;
    });
  };

  // Clear history
  const clearHistory = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      setHistory([]);
    } catch (error) {
      console.error("Failed to clear scan history:", error);
    }
  };

  return {
    history,
    addScan,
    clearHistory,
  };
}
