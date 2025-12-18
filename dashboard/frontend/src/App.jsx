import React, { useState } from 'react';
import ConfigEditor from './components/ConfigEditor';
import ScriptRunner from './components/ScriptRunner';
import { LayoutDashboard, FileCode } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('runner');

  return (
    <div className="h-screen w-screen bg-slate-950 flex flex-col overflow-hidden text-slate-200">
      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center gap-4 shadow-sm z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-900/50">
            <span className="font-bold text-white text-lg">R</span>
          </div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
            RTLSTM Dashboard
          </h1>
        </div>

        <div className="h-6 w-px bg-slate-800 mx-2"></div>

        <nav className="flex gap-2">
          <button
            onClick={() => setActiveTab('runner')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'runner'
                ? 'bg-slate-800 text-blue-400 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
          >
            <div className="flex items-center gap-2">
              <LayoutDashboard size={16} /> Run Scripts
            </div>
          </button>

          <button
            onClick={() => setActiveTab('config')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'config'
                ? 'bg-slate-800 text-blue-400 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
          >
            <div className="flex items-center gap-2">
              <FileCode size={16} /> Configs
            </div>
          </button>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden p-6 relative">
        <div className="absolute inset-0 bg-slate-950/50 pointer-events-none"></div>
        <div className="relative h-full max-w-7xl mx-auto">
          {activeTab === 'runner' ? <ScriptRunner /> : <ConfigEditor />}
        </div>
      </main>
    </div>
  );
}

export default App;
