import React from 'react';
import { ChatInput } from '@/components/ChatInput';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { useTaskContext } from '@/context/TaskContext';

// -------------------------------------------------------------------
// Component
// -------------------------------------------------------------------

interface HomePageProps {
  onNavigate?: (page: PageId) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  const {
    backendAvailable,
    miningTask: task,
    startMining,
    stopMining,
  } = useTaskContext();

  return (
    <Layout
      currentPage="home"
      onNavigate={onNavigate || (() => {})}
      showNavigation={!!onNavigate}
    >
        {/* Welcome Screen - leave some space at the bottom to avoid overlapping with fixed input area */}
        <div className="flex flex-col items-center justify-center min-h-[60vh] pb-8 animate-fade-in-up">
          <div className="text-center mb-10">
            <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-primary via-purple-500 to-pink-500 bg-clip-text text-transparent">
              Welcome to QuantaAlpha
            </h2>
            <p className="text-lg text-muted-foreground">
              Describe your goals in natural language, and AI will mine high-quality quantitative factors automatically
            </p>
            {backendAvailable === false && (
              <p className="text-sm text-warning mt-2">
                Backend is offline; using mock data for demo
              </p>
            )}
            {backendAvailable === true && (
              <p className="text-sm text-success mt-2">
                Backend service connected
              </p>
            )}
          </div>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full mb-10">
            <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('home')}>
              <div className="text-4xl mb-3">🤖</div>
              <h3 className="font-semibold mb-2">AI Factor Mining</h3>
              <p className="text-sm text-muted-foreground">
                LLM understands requirements, generates hypotheses, and iteratively evolves them
              </p>
            </div>
            <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('library')}>
              <div className="text-4xl mb-3">📊</div>
              <h3 className="font-semibold mb-2">Factor Library Management</h3>
              <p className="text-sm text-muted-foreground">
                Browse, filter, and analyze all mined factors
              </p>
            </div>
            <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('backtest')}>
              <div className="text-4xl mb-3">🚀</div>
              <h3 className="font-semibold mb-2">Independent Backtest</h3>
              <p className="text-sm text-muted-foreground">
                Select a factor library to run full-period out-of-sample backtests
              </p>
            </div>
          </div>

          {/* System Info Panel */}
          <div className="w-full max-w-4xl glass rounded-2xl p-6 text-sm space-y-3">
            <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
              <span className="text-lg">💡</span> Usage Notes
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 text-muted-foreground">

              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">Default market:</strong>CSI 300 market stock data</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">Mining period:</strong>Training set 2016-2020, validation set 2021 (initial backtests run on validation set)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">Independent backtest:</strong>Test set 2022-01-01 ~ 2025-12-26 (evaluate out-of-sample performance)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">Resource usage:</strong>LLM token/time usage is proportional to <strong className="text-foreground">(evolution rounds x parallel directions)</strong></span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">Base factors:</strong>In the main experiment, new factors are backtested together with 4 base factors (opening return, volume ratio, amplitude return, daily return)</span>
              </div>
            </div>
          </div>
        </div>

      {/* Bottom Chat Input - Always visible on Home Page for starting new tasks */}
      <ChatInput
        onSubmit={startMining}
        onStop={stopMining}
        isRunning={task?.status === 'running'}
      />
    </Layout>
  );
};
