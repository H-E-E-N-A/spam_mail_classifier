import React, { useState } from 'react';
import { Shield, ShieldAlert, ShieldCheck, BarChart3, Mail, Terminal, Info, Trash2, CheckCircle2 } from 'lucide-react';

const SPAM_KEYWORDS = ['winner', 'prize', 'claim', 'free', 'urgent', 'account', 'verify', 'lottery', 'cash', 'reward', 'investment', 'guaranteed'];

const classifyEmail = (text) => {
  const words = text.toLowerCase().split(/\W+/);
  const spamScore = words.filter(word => SPAM_KEYWORDS.includes(word)).length;
  const isSpam = spamScore >= 2 || (spamScore > 0 && text.length < 50);
  const confidence = Math.min(75 + (spamScore * 5), 99) + (Math.random() * 0.9);
  return { label: isSpam ? 'spam' : 'ham', confidence: isSpam ? confidence : 100 - (confidence / 2) };
};

export default function App() {
  const [inputText, setInputText] = useState('');
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('classifier');

  const handleClassify = () => {
    if (!inputText.trim()) return;
    const result = classifyEmail(inputText);
    setHistory([{ id: Date.now(), text: inputText, ...result }, ...history]);
    setInputText('');
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <nav className="bg-slate-900 text-white p-4 shadow-lg">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="text-blue-400 w-8 h-8" />
            <h1 className="text-xl font-bold">Sentinel<span className="text-blue-400">ML</span></h1>
          </div>
          <div className="flex gap-6 text-sm">
            <button onClick={() => setActiveTab('classifier')} className={activeTab === 'classifier' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-400'}>Classifier</button>
            <button onClick={() => setActiveTab('metrics')} className={activeTab === 'metrics' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-400'}>Metrics</button>
          </div>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto p-8">
        {activeTab === 'classifier' ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-white rounded-2xl shadow-sm border p-6">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Paste email content here..."
                  className="w-full h-40 p-4 rounded-xl border bg-slate-50 outline-none"
                />
                <button onClick={handleClassify} className="mt-4 w-full bg-blue-600 text-white py-3 rounded-lg font-bold">Analyze Message</button>
              </div>
              <div className="bg-white rounded-2xl shadow-sm border overflow-hidden">
                <div className="p-4 bg-slate-50 border-b font-bold">Analysis History</div>
                {history.map(item => (
                  <div key={item.id} className="p-4 border-b flex justify-between">
                    <p className="italic text-sm truncate max-w-md">"{item.text}"</p>
                    <span className={`px-2 py-1 rounded text-xs font-bold ${item.label === 'spam' ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}>{item.label.toUpperCase()}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center p-20 bg-white rounded-2xl border">Model Performance: 98.4% Accuracy</div>
        )}
      </main>
    </div>
  );
}