import React, { useState, useEffect } from 'react';
import { Save, FileJson, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function ConfigEditor() {
    const [files, setFiles] = useState([]);
    const [selectedFile, setSelectedFile] = useState(null);
    const [content, setContent] = useState('');
    const [status, setStatus] = useState('');

    useEffect(() => {
        fetchConfigs();
    }, []);

    useEffect(() => {
        if (selectedFile) {
            loadConfig(selectedFile);
        }
    }, [selectedFile]);

    const fetchConfigs = async () => {
        try {
            const res = await fetch(`${API_BASE}/configs`);
            const data = await res.json();
            setFiles(data);
        } catch (err) {
            console.error("Failed to fetch configs", err);
        }
    };

    const loadConfig = async (filename) => {
        try {
            setStatus('Loading...');
            const res = await fetch(`${API_BASE}/configs/${filename}`);
            const data = await res.json();
            setContent(JSON.stringify(data, null, 4));
            setStatus('');
        } catch (err) {
            setStatus('Error loading file');
            console.error(err);
        }
    };

    const handleSave = async () => {
        if (!selectedFile) return;
        try {
            setStatus('Saving...');
            let jsonContent;
            try {
                jsonContent = JSON.parse(content);
            } catch (e) {
                setStatus('Invalid JSON');
                return;
            }

            await fetch(`${API_BASE}/configs/${selectedFile}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(jsonContent)
            });
            setStatus('Saved!');
            setTimeout(() => setStatus(''), 2000);
        } catch (err) {
            setStatus('Error saving');
            console.error(err);
        }
    };

    return (
        <div className="flex h-full gap-4 text-slate-300">
            {/* Sidebar */}
            <div className="w-64 flex-shrink-0 bg-slate-800 rounded-lg p-4 border border-slate-700">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold text-white flex items-center gap-2">
                        <FileJson size={18} /> Configs
                    </h2>
                    <button onClick={fetchConfigs} className="p-1 hover:bg-slate-700 rounded">
                        <RefreshCw size={14} />
                    </button>
                </div>
                <div className="space-y-1">
                    {files.map(file => (
                        <button
                            key={file}
                            onClick={() => setSelectedFile(file)}
                            className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${selectedFile === file
                                    ? 'bg-blue-600 text-white'
                                    : 'hover:bg-slate-700'
                                }`}
                        >
                            {file}
                        </button>
                    ))}
                </div>
            </div>

            {/* Editor */}
            <div className="flex-1 bg-slate-800 rounded-lg border border-slate-700 flex flex-col overflow-hidden">
                {selectedFile ? (
                    <>
                        <div className="p-3 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
                            <span className="font-mono text-sm">{selectedFile}</span>
                            <div className="flex items-center gap-4">
                                <span className="text-xs text-yellow-400">{status}</span>
                                <button
                                    onClick={handleSave}
                                    className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm transition-colors"
                                >
                                    <Save size={16} /> Save
                                </button>
                            </div>
                        </div>
                        <textarea
                            value={content}
                            onChange={(e) => setContent(e.target.value)}
                            className="flex-1 bg-slate-900 p-4 font-mono text-sm resize-none focus:outline-none focus:ring-1 focus:ring-blue-500/50 text-slate-300"
                            spellCheck={false}
                        />
                    </>
                ) : (
                    <div className="flex-1 flex items-center justify-center text-slate-500">
                        Select a file to edit
                    </div>
                )}
            </div>
        </div>
    );
}
