import React, { useState, useEffect, useRef } from 'react';
import { Mic, Send, Brain, MessageSquare, Activity, ShieldCheck, Lock, Sparkles, Calendar, Download, X, Heart } from 'lucide-react';

const App = () => {
  // 1. AUTH
  const [isAuth, setIsAuth] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('auth') === 'success') {
      const user = params.get('user');
      localStorage.setItem("isLoggedIn", "true");
      localStorage.setItem("username", user || "User");
      window.history.replaceState({}, document.title, "/");
      return true;
    }
    return localStorage.getItem("isLoggedIn") === "true";
  });

  // 2. STATE
  const [messages, setMessages] = useState([{
    role: 'assistant',
    text: `Hello ${localStorage.getItem("username") || "there"}! To start, could you share your schedule from today? It helps me understand your day better (e.g., 8-9 Breakfast, 9-5 Work).`,
    emotion: 'neutral'
  }]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState('neutral');
  const [turn, setTurn] = useState(1);
  const [isFinal, setIsFinal] = useState(false);
  const [isExtraPhase, setIsExtraPhase] = useState(false);
  const [activeView, setActiveView] = useState('chat');
  const [sessionConcluded, setSessionConcluded] = useState(false);
  const [sessionId] = useState(`sess_${Date.now()}`);

  const scrollRef = useRef(null);
  const recognition = useRef(null);
  const mediaRecorderRef = useRef(null);
  const transcriptRef = useRef('');
  const audioBlobRef = useRef(null);

  const [downloadModalOpen, setDownloadModalOpen] = useState(false);
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [downloading, setDownloading] = useState(false);

  const toDisplay = (d) => { const p = d.split('-'); return p.length === 3 ? `${p[2]}-${p[1]}-${p[0]}` : d; };

  const fetchHistoryDates = async () => {
    try {
      const user = localStorage.getItem("username") || "Guest";
      const res = await fetch(`https://training-independently-targeted-examining.trycloudflare.com/history/${user}`);
      const data = await res.json();
      const dates = data.dates || [];
      setAvailableDates(dates);
      if (dates.length > 0) setSelectedDate(dates[0]);
    } catch { console.error("Failed to fetch history dates"); }
  };
  useEffect(() => { if (downloadModalOpen) fetchHistoryDates(); }, [downloadModalOpen]);

  const handleDownloadPdf = async () => {
    if (!selectedDate) return;
    setDownloading(true);
    try {
      const user = localStorage.getItem("username") || "Guest";
      const res = await fetch(`https://training-independently-targeted-examining.trycloudflare.com/download-pdf/${user}/${selectedDate}`);
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `Wellness_Report_${selectedDate}.pdf`;
      document.body.appendChild(a); a.click();
      window.URL.revokeObjectURL(url);
    } catch { alert("Failed to download PDF."); }
    setDownloading(false);
  };

  // 3. CORE
  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SR) {
      recognition.current = new SR();
      recognition.current.continuous = true;
      recognition.current.onresult = (e) => {
        let t = ''; for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
        setInputText(t); transcriptRef.current = t;
      };
    }
  }, []);
  useEffect(() => { scrollRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleSend = async (textOverride = null, audioBlob = null) => {
    // If called directly with audio blob or text
    const text = (textOverride || transcriptRef.current || inputText || "").trim();
    const finalBlob = audioBlob || audioBlobRef.current;

    if (!text && !finalBlob) return;

    // Clear the input fields immediately so leftover text doesn't linger in the input bar
    setInputText('');
    transcriptRef.current = '';
    if (audioBlobRef.current) audioBlobRef.current = null;

    setMessages(prev => [...prev, { role: 'user', text: text || "Voice message..." }]);

    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('text', text);
    fd.append('username', localStorage.getItem('username') || "Guest");
    fd.append('is_extra_phase', isExtraPhase ? 'true' : 'false');

    if (finalBlob) {
      fd.append('audio', finalBlob, 'speech.wav');
    }

    try {
      const res = await fetch('https://training-independently-targeted-examining.trycloudflare.com/chat', { method: 'POST', body: fd });
      const data = await res.json();
      setMessages(prev => {
        const u = [...prev]; const li = u.length - 1;
        if (data.transcribed_text) u[li].text = data.transcribed_text;
        return [...u, { role: 'assistant', text: data.response, emotion: data.emotion }];
      });
      setCurrentEmotion(data.emotion); setTurn(data.current_turn);
      setIsFinal(data.is_final);
      if (data.concluded) setSessionConcluded(true);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', text: "Connection error. Ensure backend is at http://127.0.0.1:8000", emotion: 'fear' }]);
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      recognition.current?.stop();
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
      // Transcribed text is already in inputText via onresult — user can review and press Send
    } else {
      transcriptRef.current = '';
      setInputText('');
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);
        const chunks = [];
        mediaRecorderRef.current.ondataavailable = (e) => chunks.push(e.data);
        mediaRecorderRef.current.onstop = () => {
          stream.getTracks().forEach(t => t.stop());
          const blob = new Blob(chunks, { type: 'audio/wav' });
          // Delaying auto-send slightly to allow Speech recognition's final `onresult` event 
          // to populate the `transcriptRef` so it doesn't stay left behind in the input box!
          setTimeout(() => {
            handleSend(null, blob);
          }, 500);
        };
        mediaRecorderRef.current.start();
        recognition.current?.start();
        setIsRecording(true);
      } catch { alert("Microphone access denied."); }
    }
  };

  const handleLogout = () => { localStorage.clear(); setIsAuth(false); window.location.replace("https://dir-triple-volumes-hydraulic.trycloudflare.com"); };

  // 4. UNAUTHORIZED
  if (!isAuth) {
    return (
      <div className="app-root h-screen flex items-center justify-center">
        <div className="card-glass p-12 text-center shadow-2xl max-w-md w-full mx-4 rounded-[2rem]">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-8"
            style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent-border)' }}>
            <Lock style={{ color: 'var(--accent)' }} size={24} />
          </div>
          <h2 className="text-2xl font-bold mb-3" style={{ color: 'var(--text-main)' }}>Access Restricted</h2>
          <p className="mb-10 leading-relaxed text-sm" style={{ color: 'var(--text-soft)' }}>
            Please login via the secure portal to begin your wellness session.
          </p>
          <a href="https://dir-triple-volumes-hydraulic.trycloudflare.com"
            className="block w-full py-4 rounded-full font-bold text-sm uppercase tracking-widest text-center transition-all hover:opacity-90 hover:scale-105"
            style={{ background: 'var(--accent)', color: '#1a0a3b' }}>
            Go to Login
          </a>
        </div>
      </div>
    );
  }

  // 5. MAIN INTERFACE
  return (
    <div className="app-root flex h-screen overflow-hidden">

      {/* ── SIDEBAR ── */}
      <aside className="sidebar-glass w-64 flex flex-col p-5 z-20 shrink-0">

        {/* Logo */}
        <div className="flex items-center gap-3 mb-8 px-2">
          <div className="p-2 rounded-xl" style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent-border)' }}>
            <Brain style={{ color: 'var(--accent)' }} className="w-5 h-5" />
          </div>
          <span className="font-semibold text-lg tracking-tight" style={{ color: 'var(--text-main)' }}>Hridaya</span>
        </div>

        {/* Nav items */}
        <nav className="space-y-1 flex-grow">
          {[
            { id: 'chat', Icon: MessageSquare, label: 'Daily Session' },
            { id: 'daily', Icon: Activity, label: 'Daily Insights' },
            { id: 'weekly', Icon: Calendar, label: 'Weekly Analysis' },
          ].map(({ id, Icon, label }) => (
            <button key={id} onClick={() => setActiveView(id)}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 text-sm font-medium"
              style={activeView === id
                ? { background: 'var(--accent-muted)', color: 'var(--accent)', border: '1px solid var(--accent-border)' }
                : { color: 'var(--text-muted)', border: '1px solid transparent' }}>
              <Icon size={16} /> {label}
            </button>
          ))}

          <div className="pt-3 mt-2" style={{ borderTop: '1px solid var(--border-soft)' }}>
            <button onClick={() => setDownloadModalOpen(true)}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all hover:bg-white/5"
              style={{ color: 'var(--text-muted)', border: '1px solid transparent' }}>
              <Download size={16} /> Export Records
            </button>
          </div>
        </nav>

        {/* Bottom */}
        <div className="space-y-3">
          <button onClick={handleLogout}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition-all hover:bg-red-500/10"
            style={{ color: 'var(--text-muted)' }}>
            <Lock size={16} /> Logout
          </button>

          <div className="px-4 py-4 rounded-2xl text-[10px] uppercase tracking-widest leading-relaxed"
            style={{ background: 'rgba(18,8,38,.50)', border: '1px solid var(--border-soft)' }}>
            <div className="flex items-center gap-2 mb-2 font-bold" style={{ color: 'var(--accent)' }}>
              <ShieldCheck size={12} /> Clinical Guard
            </div>
            <span style={{ color: 'var(--text-muted)' }}>
              User: {localStorage.getItem("username") || "Guest"}<br />
              Session: {turn} / 11
            </span>
          </div>
        </div>
      </aside>

      {/* ── MAIN CONTENT ── */}
      <div className="flex-grow flex flex-col relative" style={{ background: 'transparent' }}>

        {activeView === 'chat' ? (
          <>
            {/* Header */}
            <header className="header-glass h-16 flex items-center justify-between px-8 shrink-0">
              <div className="px-3 py-1.5 rounded-full text-[10px] font-semibold uppercase tracking-widest capitalize"
                style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent-border)', color: 'var(--accent)' }}>
                ✦ {currentEmotion}
              </div>
              <div className="text-[11px] font-medium uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                {turn === 1 ? 'Phase: Schedule' : 'Phase: Diagnostic'}
                <span className="ml-2 font-bold" style={{ color: 'var(--accent)' }}>{turn} / 11</span>
              </div>
            </header>

            {/* Messages */}
            <div className="flex-grow overflow-y-auto px-8 py-6 space-y-5">
              {messages.map((m, i) => (
                <div key={i} className={`msg-bubble flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  style={{ animationDelay: `${Math.min(i * 0.04, 0.3)}s` }}>
                  <div className="max-w-[62%] px-6 py-4 text-[15px] leading-relaxed shadow-lg"
                    style={m.role === 'user'
                      ? {
                        background: 'linear-gradient(135deg, #6d28d9, #9d174d)',
                        border: '1px solid rgba(167,139,250,.3)',
                        borderRadius: '1.5rem 1.5rem 0.25rem 1.5rem',
                        color: '#fce7f3',
                        boxShadow: '0 4px 24px rgba(109,40,217,.25)'
                      }
                      : {
                        background: 'rgba(46,26,96,.65)',
                        border: '1px solid rgba(196,181,253,.14)',
                        borderRadius: '1.5rem 1.5rem 1.5rem 0.25rem',
                        color: 'var(--text-main)',
                        backdropFilter: 'blur(12px)'
                      }}>
                    {m.text}
                  </div>
                </div>
              ))}

              {/* Insights complete */}
              {isFinal && !sessionConcluded && (
                <div className="sparkle-card flex flex-col items-center gap-5 p-8 rounded-3xl text-center"
                  style={{ background: 'rgba(109,40,217,.12)', border: '1px dashed rgba(196,181,253,.25)' }}>
                  <Sparkles style={{ color: 'var(--accent)' }} size={36} />
                  <h3 className="text-lg font-semibold" style={{ color: 'var(--text-main)' }}>Session Complete ✨</h3>
                  <p className="text-sm" style={{ color: 'var(--text-soft)' }}>Your emotional insights have been captured for today.</p>
                  <div className="flex gap-3">
                    <button onClick={() => { setIsExtraPhase(true); setIsFinal(false); setTurn(0); }}
                      className="px-6 py-3 rounded-full font-semibold text-xs uppercase tracking-widest transition-all hover:opacity-90 hover:scale-105"
                      style={{ background: 'var(--accent)', color: '#1a0a3b' }}>
                      Continue Talking
                    </button>
                    <button onClick={() => setActiveView('daily')}
                      className="px-6 py-3 rounded-full font-semibold text-xs uppercase tracking-widest transition-all"
                      style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)', color: 'var(--text-soft)' }}>
                      View Insights
                    </button>
                  </div>
                </div>
              )}
              <div ref={scrollRef} />
            </div>

            {/* Input bar */}
            <div className="px-8 pb-4 pt-2 shrink-0">
              <div className={`input-glass max-w-3xl mx-auto flex items-center gap-3 px-4 py-3 rounded-2xl transition-all duration-500 ${sessionConcluded || (isFinal && !isExtraPhase) ? 'opacity-20 pointer-events-none' : ''
                }`}>
                <button onClick={toggleRecording}
                  className={`p-3 rounded-xl transition-all shrink-0 ${isRecording ? 'recording-pulse' : 'hover:bg-white/10'}`}
                  style={isRecording
                    ? { background: 'rgba(239,68,68,.18)', color: '#f87171' }
                    : { background: 'rgba(196,181,253,.08)', color: 'var(--text-muted)' }}>
                  <Mic size={20} />
                </button>
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => { setInputText(e.target.value); transcriptRef.current = e.target.value; }}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder={isRecording ? "Listening…" : "Share how you feel today…"}
                  className="flex-grow bg-transparent focus:outline-none text-[15px]"
                  style={{ color: 'var(--text-main)', caretColor: 'var(--accent)' }}
                />
                <button onClick={() => handleSend()}
                  className="p-3 rounded-xl transition-all shrink-0 hover:opacity-90 hover:scale-105"
                  style={{ background: 'var(--accent)', color: '#1a0a3b' }}>
                  <Send size={20} />
                </button>
              </div>
            </div>

            {/* Footer */}
            <div className="px-8 pb-4 shrink-0">
              <div className="max-w-3xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 pt-3"
                style={{ borderTop: '1px solid var(--border-soft)' }}>
                <div className="flex items-start gap-2 text-[10px] leading-relaxed max-w-xs" style={{ color: 'var(--text-muted)' }}>
                  <Heart size={10} className="mt-0.5 shrink-0 opacity-50" />
                  <span>
                    <span className="font-semibold opacity-80">AI Disclaimer:</span>{' '}
                    Hridaya is an AI assistant, not a licensed therapist. For real evaluation seek professional support.
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  <span className="font-bold uppercase tracking-widest text-[9px] opacity-60">🆘 Crisis Helplines</span>
                  {[['tel:9152987821', 'iCall · 9152987821'], ['tel:18602662345', 'Vandrevala · 1860-2662345'], ['tel:04424640050', 'SNEHA · 044-24640050']].map(([href, label]) => (
                    <a key={href} href={href} className="hover:text-violet-300 transition-colors">{label}</a>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : (
          /* Dashboard iframe */
          <div className="flex flex-col h-full relative">
            <div className="flex-grow w-full h-full">
              <iframe
                src={`https://above-increase-metallica-multiple.trycloudflare.com/?embed=true&view=${activeView}&username=${localStorage.getItem('username') || 'Guest'}`}
                className="w-full h-full border-none" title="Hridaya Hub" />
            </div>
            <button onClick={() => setActiveView('chat')}
              className="absolute bottom-5 right-5 p-3 rounded-xl card-glass transition-all hover:scale-105 shadow-xl z-20"
              style={{ color: 'var(--accent)' }}>
              <MessageSquare size={20} />
            </button>
          </div>
        )}
      </div>

      {/* ── DOWNLOAD MODAL ── */}
      {downloadModalOpen && (
        <div className="modal-overlay absolute inset-0 z-50 flex items-center justify-center p-8">
          <div className="card-glass max-w-md w-full p-8 shadow-2xl relative rounded-[2rem]">
            <button onClick={() => setDownloadModalOpen(false)}
              className="absolute top-5 right-5 transition-colors hover:text-white"
              style={{ color: 'var(--text-muted)' }}>
              <X size={20} />
            </button>
            <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-5"
              style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent-border)' }}>
              <Download style={{ color: 'var(--accent)' }} size={22} />
            </div>
            <h2 className="text-xl font-semibold mb-1" style={{ color: 'var(--text-main)' }}>Export Wellness Report</h2>
            <p className="text-sm mb-6 italic" style={{ color: 'var(--text-soft)' }}>Download your session summaries as a PDF.</p>
            <div className="space-y-3">
              {availableDates.length === 0 ? (
                <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>No sessions found. Complete a session first.</p>
              ) : (
                <select value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)}
                  className="w-full p-3 focus:outline-none cursor-pointer text-sm rounded-xl"
                  style={{ background: 'rgba(18,8,38,.7)', border: '1px solid var(--border)', color: 'var(--text-main)' }}>
                  {availableDates.map((d) => <option key={d} value={d}>{toDisplay(d)}</option>)}
                </select>
              )}
              <button onClick={handleDownloadPdf}
                disabled={!selectedDate || downloading || availableDates.length === 0}
                className="w-full py-3 rounded-xl font-semibold text-sm uppercase tracking-widest transition-all hover:opacity-90 hover:scale-[1.02] disabled:opacity-40 disabled:hover:scale-100 flex justify-center items-center gap-2"
                style={{ background: 'var(--accent)', color: '#1a0a3b' }}>
                {downloading ? 'Generating PDF…' : 'Download PDF'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;