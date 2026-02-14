import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Square, Compass } from 'lucide-react';
import { TaskConfig } from '@/types';

interface ChatInputProps {
  onSubmit: (config: TaskConfig) => void;
  onStop?: () => void;
  isRunning?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSubmit, onStop, isRunning = false }) => {
  const [input, setInput] = useState('');
  const [useCustomMiningDirection, setUseCustomMiningDirection] = useState(false);
  const [config] = useState<Partial<TaskConfig>>({
    librarySuffix: '',
  });
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const examplePrompts = [
    '💹 Mine momentum factors focused on short-term reversal and volume confirmation',
    '💰 Explore value-growth combinations with industry neutrality',
    '📊 Build factors from technical indicators, focusing on RSI and MACD',
  ];

  const handleSubmit = () => {
    if (isRunning) return;
    const suffix = config.librarySuffix?.trim() || undefined;
    onSubmit({
      userInput: input.trim(),
      useCustomMiningDirection,
      ...config,
      librarySuffix: suffix,
    } as TaskConfig);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 pb-6">
      <div className="container mx-auto px-6">
        
        {/* Example Prompts */}
        {!input && !isRunning && (
          <div className="flex flex-wrap justify-center gap-2 mb-3 overflow-x-auto pb-2 scrollbar-hide">
            {examplePrompts.map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => setInput(prompt)}
                className="glass rounded-xl px-4 py-2 text-sm text-muted-foreground hover:text-foreground hover:scale-105 transition-all whitespace-nowrap flex items-center gap-2 card-hover"
              >
                <Sparkles className="h-3 w-3" />
                {prompt}
              </button>
            ))}
          </div>
        )}

        {/* Main Input */}
        <div className="gradient-border">
          <div className="gradient-border-content">
            <div className="glass-strong rounded-xl p-4">
              {/* Icon bar: Custom mining direction etc. */}
              <div className="flex items-center gap-1 mb-3">
                <button
                  type="button"
                  onClick={() => setUseCustomMiningDirection((v) => !v)}
                  title={useCustomMiningDirection ? 'Use mining directions from Settings (enabled)' : 'Use mining directions from Settings (click to enable)'}
                  className={`p-2 rounded-lg transition-all ${
                    useCustomMiningDirection
                      ? 'bg-primary/15 text-primary ring-1 ring-primary/30'
                      : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground'
                  }`}
                >
                  <Compass className="h-4 w-4" />
                </button>
                <span
                  className={`text-xs ml-1 ${
                    useCustomMiningDirection ? 'text-primary font-medium' : 'text-muted-foreground'
                  }`}
                >
                  Custom mining direction
                </span>
              </div>
              <div className="flex items-end gap-3">
                <div className="flex-1">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={
                      isRunning
                        ? 'Experiment is running... you can switch pages without interrupting the task'
                        : useCustomMiningDirection
                        ? 'Custom mining direction is enabled; options from "Settings → Mining Directions" will be used'
                        : 'Describe factor mining requirements, or enable "Custom mining direction" to use settings options (Shift+Enter for new line, Enter to send)'
                    }
                    disabled={isRunning}
                    className="w-full bg-transparent text-base placeholder:text-muted-foreground focus:outline-none resize-none"
                    rows={1}
                    style={{ maxHeight: '120px' }}
                  />
                </div>

                <div className="flex items-center gap-2">
                  {isRunning && onStop ? (
                    <button
                      onClick={onStop}
                      className="p-2.5 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-all hover:scale-105 active:scale-95"
                      title="Stop experiment"
                    >
                      <Square className="h-5 w-5" />
                    </button>
                  ) : (
                    <button
                      onClick={handleSubmit}
                      disabled={isRunning}
                      className="p-2.5 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
                      title="Send (Enter)"
                    >
                      <Send className="h-5 w-5" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
