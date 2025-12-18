import React, { useState, useEffect } from 'react';
import { Play, Terminal, AlertCircle } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function ScriptRunner() {
    const [scripts, setScripts] = useState([]);
    const [selectedScript, setSelectedScript] = useState('');
    const [args, setArgs] = useState('--help');
    const [output, setOutput] = useState('');
    const [isRunning, setIsRunning] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchScripts();
    }, []);

    const fetchScripts = async () => {
        try {
            const res = await fetch(`${API_BASE}/scripts`);
            const data = await res.json();
            setScripts(data);
            if (data.length > 0) setSelectedScript(data[0]);
        } catch (err) {
            console.error("Failed to fetch scripts", err);
        }
    };

    const handleRun = async () => {
        if (!selectedScript) return;

        setIsRunning(true);
        setOutput('');
        setError('');

        try {
            // Simulating a stream or just waiting for response
            const res = await fetch(`${API_BASE}/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script: selectedScript, args: args })
            });

            const data = await res.json();

            if (!res.ok) {
                setError(data.detail || 'Failed to run script');
                setIsRunning(false);
                return;
            }

            const out = `>>> Return Code: ${data.return_code}\n\nSTDOUT:\n${data.stdout}\n\nSTDERR:\n${data.stderr}`;
            setOutput(out);

        } catch (err) {
            setError(err.message);
        } finally {
            setIsRunning(false);
        }
    };

    return (
        <div className="flex flex-col h-full gap-4 text-slate-300">
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                    <Terminal className="text-blue-500" /> Execute Script
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-400">Script</label>
                        <select
                            value={selectedScript}
                            onChange={(e) => setSelectedScript(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                        >
                            {scripts.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-400">Arguments</label>
                        <input
                            type="text"
                            value={args}
                            onChange={(e) => setArgs(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none font-mono text-sm"
                            placeholder="--arg value"
                        />
                    </div>
                </div>

                <div className="mt-6">
                    <button
                        onClick={handleRun}
                        disabled={isRunning || !selectedScript}
                        className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-all ${isRunning
                                ? 'bg-slate-700 cursor-not-allowed text-slate-400'
                                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20'
                            }`}
                    >
                        {isRunning ? (
                            <RefreshCw className="animate-spin" size={18} />
                        ) : (
                            <Play size={18} fill="currentColor" />
                        )}
                        {isRunning ? 'Running...' : 'Run Script'}
                    </button>
                </div>
            </div>

            {/* Output Console */}
            <div className="flex-1 bg-black rounded-lg border border-slate-800 p-4 font-mono text-sm overflow-auto text-slate-300 shadow-inner">
                {error && (
                    <div className="text-red-400 mb-2 flex items-center gap-2">
                        <AlertCircle size={14} /> {error}
                    </div>
                )}
                {output ? (
                    <pre className="whitespace-pre-wrap break-words">{output}</pre>
                ) : (
                    <span className="text-slate-600 italic">Output will appear here...</span>
                )}
            </div>
        </div>
    );
}

// Helper icon since I forgot to import it in the implementation above logic but used it
function RefreshCw({ size, className }) {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={className}
        >
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
            <path d="M8 16H3v5" />
        </svg>
    )
}
