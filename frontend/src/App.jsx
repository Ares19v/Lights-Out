import React, { useState, useEffect, useRef } from "react";

// ── Formal / Professional Colour themes ───────────────────────────────────
const THEMES = {
  black: { bg: "#0f1115", panel: "#16181d", border: "#252830", accent: "#a3a6ad", text: "#f1f2f4", err: "#ff4d4f", ok: "#10b981", warn: "#faad14" },
  blue:  { bg: "#f0f2f5", panel: "#ffffff", border: "#e5e7eb", accent: "#2563eb", text: "#1f2937", err: "#ef4444", ok: "#10b981", warn: "#f59e0b" },
};

// All supported target keywords
const FACE_TARGETS = [
  "nose", "nose tip", "nose bridge", "nose root", "left nostril", "right nostril",
  "chin", "jaw", "left jaw", "right jaw",
  "forehead", "left temple", "right temple",
  "left eye", "right eye", "left pupil", "right pupil", 
  "left eye inner", "right eye inner", "left eye outer", "right eye outer",
  "left eye top", "left eye bottom", "right eye top", "right eye bottom",
  "left eyebrow", "right eyebrow", "left eyebrow inner", "left eyebrow outer", "right eyebrow inner", "right eyebrow outer",
  "mouth", "lips", "upper lip", "lower lip", "upper lip top", "lower lip bottom", "left mouth corner", "right mouth corner",
  "left cheek", "right cheek"
];
const BODY_TARGETS = [
  "pose nose", "pose left eye", "pose right eye", "left ear", "right ear",
  "left shoulder", "right shoulder", "shoulder",
  "left elbow", "right elbow", "elbow",
  "left wrist", "right wrist", "wrist",
  "left hip", "right hip", "hip",
  "left knee", "right knee", "knee",
  "left ankle", "right ankle", "ankle"
];

function App() {
  const [theme, setTheme]               = useState("black");
  const [isStarted, setIsStarted]       = useState(false);
  const [isConnected, setIsConnected]   = useState(false);
  const [fps, setFps]                   = useState(0);
  const [targetInput, setTargetInput]   = useState("");
  const [currentTarget, setCurrentTarget] = useState("nose");
  const [lockStatus, setLockStatus]     = useState("ACQUIRING");
  const [confidence, setConfidence]     = useState(0);
  const [eventLog, setEventLog]         = useState([]);
  const [missCount, setMissCount]       = useState(0);

  const [faceExpanded, setFaceExpanded] = useState(true);
  const [bodyExpanded, setBodyExpanded] = useState(false);
  
  const [showSettings, setShowSettings] = useState(false);
  const [config, setConfig] = useState({
    depth: false,
    gesture: false,
    crosshairStyle: "military",
    hudColor: "green",
    multiTarget: true,
    voice: false,
    confidence: 0.3,
  });

  const videoRef     = useRef(null);
  const wsRef        = useRef(null);
  const lastFrameTs  = useRef(Date.now());
  const colors       = THEMES[theme];

  const setConfigValue = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ cmd: "set_config", key, value }));
    }
  };

  const toggleConfig = (key) => setConfigValue(key, !config[key]);

  // ── WebSocket connection ──────────────────────────────────────────────────
  useEffect(() => {
    if (!isStarted) return;

    const ws = new WebSocket("ws://127.0.0.1:8765");
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      addLog("System connection established.", "info");
    };

    ws.onclose = () => {
      setIsConnected(false);
      setFps(0);
      addLog("Connection lost. Retrying...", "err");
    };

    ws.onerror = () => {
      addLog("Connection error. Ensure backend is running.", "err");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (videoRef.current && data.image) {
        videoRef.current.src = "data:image/jpeg;base64," + data.image;
      }

      if (data.status   !== undefined) setLockStatus(data.status);
      if (data.conf     !== undefined) setConfidence(data.conf);
      if (data.miss     !== undefined) setMissCount(data.miss);
      if (data.target   !== undefined) setCurrentTarget(data.target);

      const now = Date.now();
      setFps(Math.round(1000 / (now - lastFrameTs.current)));
      lastFrameTs.current = now;

      if (data.status === "LOCKED" && data.target) {
        const ts = new Date().toLocaleTimeString();
        addLog(`Locked onto: ${data.target}`, "ok");
      }
    };

    return () => ws.close();
  }, [isStarted]);

  const addLog = (msg, type="info") => {
    const ts = new Date().toLocaleTimeString([], {hour: "2-digit", minute:"2-digit", second:"2-digit"});
    setEventLog((prev) => [{time: ts, msg, type}, ...prev].slice(0, 40));
  };

  const sendTarget = (kw) => {
    const trimmed = kw.trim().toLowerCase();
    if (!trimmed) return;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ cmd: "set_target", target: trimmed }));
      addLog(`Target updated to "${trimmed}"`, "info");
      setCurrentTarget(trimmed);
    }
    setTargetInput("");
  };

  const sendShutdown = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ cmd: "shutdown" }));
    }
  };

  const statusColor = {
    LOCKED:    colors.ok,
    ACQUIRING: colors.warn,
    LOST:      colors.err,
    MANUAL:    colors.accent,
  }[lockStatus] ?? colors.accent;

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div style={{
      backgroundColor: colors.bg, minHeight: "100vh", color: colors.text,
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", 
      padding: "32px", display: "flex", flexDirection: "column", transition: "all 0.3s ease",
    }}>
      <style dangerouslySetInnerHTML={{ __html: `
        body { margin: 0; padding: 0; overflow-x: hidden; }
        @keyframes spin { to { transform: rotate(360deg); } }
        input:focus { outline: none; border-color: ${colors.accent}; box-shadow: 0 0 0 2px ${colors.accent}33; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${colors.border}; border-radius: 4px; }
        .dashboard-card { background: ${colors.panel}; border: 1px solid ${colors.border}; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
      `}} />

      {/* ── Header ── */}
      <header style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        paddingBottom: "24px", marginBottom: "24px", borderBottom: `1px solid ${colors.border}`
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <h1 style={{ margin: 0, fontSize: "20px", fontWeight: 600, letterSpacing: "-0.5px" }}>
            Lights-Out <span style={{ color: colors.accent, fontWeight: 400 }}>Professional</span>
          </h1>
          <div style={{ display: "flex", gap: "6px" }}>
            {Object.keys(THEMES).map(t => (
              <div key={t} onClick={() => setTheme(t)} style={{
                width: "12px", height: "12px", borderRadius: "50%",
                backgroundColor: THEMES[t].bg, border: `1px solid ${THEMES[t].border}`,
                cursor: "pointer", opacity: theme === t ? 1 : 0.4,
              }} title={`${t} theme`} />
            ))}
          </div>
        </div>

        <div style={{ display: "flex", gap: "24px", fontSize: "14px", fontWeight: 500 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: isConnected ? colors.ok : colors.err }} />
            {isConnected ? "Connected" : "Disconnected"}
          </div>
          <div style={{ color: colors.accent }}>Engine: MediaPipe + YOLOv8</div>
        </div>
      </header>

      {/* ── Main Layout ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "24px", flex: 1 }}>

        {/* ── Left Column: Video & Main Metrics ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* Main Viewport */}
          <div className="dashboard-card" style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ padding: "12px 16px", borderBottom: `1px solid ${colors.border}`, display: "flex", justifyContent: "space-between", fontSize: "13px", fontWeight: 500, color: colors.accent }}>
              <span>Live Camera Feed</span>
              <span>Input: 1280x720</span>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", position: "relative", width: "100%", aspectRatio: "16/9", backgroundColor: colors.panel }}>
              {!isStarted ? (
                <button
                  onClick={() => setIsStarted(true)}
                  style={{
                    padding: "12px 24px", backgroundColor: colors.accent, color: theme === 'black' ? '#000' : '#fff',
                    border: "none", borderRadius: "6px", fontSize: "15px", fontWeight: 500, cursor: "pointer", transition: "opacity 0.2s"
                  }}
                  onMouseOver={e => e.target.style.opacity = 0.9}
                  onMouseOut={e  => e.target.style.opacity = 1}
                >
                  Start Tracking Engine
                </button>
              ) : (
                <img ref={videoRef} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} alt="Feed" />
              )}
            </div>
          </div>

          {/* Telemetry (FPS & Confidence Outside Video) */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
            <div className="dashboard-card" style={{ padding: "20px" }}>
              <div style={{ fontSize: "13px", color: colors.accent, fontWeight: 500, marginBottom: "8px" }}>System Performance</div>
              <div style={{ display: "flex", alignItems: "baseline", gap: "8px" }}>
                <span style={{ fontSize: "32px", fontWeight: 600 }}>{fps}</span>
                <span style={{ fontSize: "14px", color: colors.accent }}>frames per second</span>
              </div>
            </div>
            
            <div className="dashboard-card" style={{ padding: "20px" }}>
              <div style={{ fontSize: "13px", color: colors.accent, fontWeight: 500, marginBottom: "12px" }}>Tracking Confidence</div>
              <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                <div style={{ flex: 1, height: "8px", backgroundColor: colors.border, borderRadius: "4px", overflow: "hidden" }}>
                  <div style={{ width: `${confidence * 100}%`, height: "100%", backgroundColor: statusColor, transition: "width 0.3s ease" }} />
                </div>
                <div style={{ fontSize: "18px", fontWeight: 600, width: "48px", textAlign: "right" }}>
                  {Math.round(confidence * 100)}%
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Right Column: Controls ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>

          {/* Target Status */}
          <div className="dashboard-card" style={{ padding: "20px", borderTop: `4px solid ${statusColor}` }}>
            <div style={{ fontSize: "13px", color: colors.accent, fontWeight: 500, marginBottom: "4px" }}>Current Target</div>
            <div style={{ fontSize: "24px", fontWeight: 600, textTransform: "capitalize", marginBottom: "8px" }}>
              {currentTarget}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", color: statusColor, fontWeight: 500, fontSize: "14px" }}>
              <span style={{ display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", backgroundColor: statusColor }} />
              {lockStatus}
            </div>
          </div>

          {/* Manual Input */}
          <div className="dashboard-card" style={{ padding: "20px" }}>
            <div style={{ fontSize: "13px", color: colors.accent, fontWeight: 500, marginBottom: "12px" }}>Assign Target</div>
            <div style={{ display: "flex", gap: "8px" }}>
              <input
                value={targetInput}
                onChange={e => setTargetInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && sendTarget(targetInput)}
                placeholder="e.g. nose, left elbow"
                style={{
                  flex: 1, padding: "10px 12px", backgroundColor: theme === 'black' ? '#0f1115' : '#f9fafb',
                  border: `1px solid ${colors.border}`, color: colors.text, borderRadius: "6px", fontSize: "14px",
                }}
              />
              <button
                onClick={() => sendTarget(targetInput)}
                style={{
                  padding: "0 16px", backgroundColor: colors.accent, color: theme === 'black' ? '#000' : '#fff',
                  border: "none", borderRadius: "6px", fontWeight: 500, cursor: "pointer"
                }}
              >
                Set
              </button>
            </div>
          </div>

          {/* Quick Selection */}
          <div className="dashboard-card" style={{ padding: "0", display: "flex", flexDirection: "column", flex: 1, maxHeight: "400px", overflowY: "auto" }}>
            
            {/* Facial Landmarks Accordion */}
            <div style={{ borderBottom: `1px solid ${colors.border}` }}>
              <div 
                onClick={() => setFaceExpanded(!faceExpanded)}
                style={{ 
                  padding: "16px 20px", fontSize: "13px", color: colors.accent, fontWeight: 500, 
                  cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center",
                  backgroundColor: faceExpanded ? colors.panel : "transparent", transition: "background-color 0.2s"
                }}
              >
                <span>Facial Landmarks</span>
                <span style={{ transform: faceExpanded ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.3s" }}>▼</span>
              </div>
              
              <div style={{ 
                padding: faceExpanded ? "0 20px 20px 20px" : "0", 
                maxHeight: faceExpanded ? "500px" : "0", 
                overflow: "hidden", transition: "all 0.3s ease",
                opacity: faceExpanded ? 1 : 0
              }}>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {FACE_TARGETS.map(kw => (
                    <button key={kw} onClick={() => sendTarget(kw)} style={{
                      padding: "6px 12px", backgroundColor: currentTarget === kw ? colors.accent : "transparent",
                      color: currentTarget === kw ? (theme === 'black' ? '#000' : '#fff') : colors.text,
                      border: `1px solid ${currentTarget === kw ? colors.accent : colors.border}`, 
                      borderRadius: "20px", cursor: "pointer", fontSize: "12px", transition: "all 0.2s"
                    }}>
                      {kw}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Body Keypoints Accordion */}
            <div>
              <div 
                onClick={() => setBodyExpanded(!bodyExpanded)}
                style={{ 
                  padding: "16px 20px", fontSize: "13px", color: colors.accent, fontWeight: 500, 
                  cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center",
                  backgroundColor: bodyExpanded ? colors.panel : "transparent", transition: "background-color 0.2s"
                }}
              >
                <span>Body Keypoints</span>
                <span style={{ transform: bodyExpanded ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.3s" }}>▼</span>
              </div>
              
              <div style={{ 
                padding: bodyExpanded ? "0 20px 20px 20px" : "0", 
                maxHeight: bodyExpanded ? "500px" : "0", 
                overflow: "hidden", transition: "all 0.3s ease",
                opacity: bodyExpanded ? 1 : 0
              }}>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {BODY_TARGETS.map(kw => (
                    <button key={kw} onClick={() => sendTarget(kw)} style={{
                      padding: "6px 12px", backgroundColor: currentTarget === kw ? colors.accent : "transparent",
                      color: currentTarget === kw ? (theme === 'black' ? '#000' : '#fff') : colors.text,
                      border: `1px solid ${currentTarget === kw ? colors.accent : colors.border}`, 
                      borderRadius: "20px", cursor: "pointer", fontSize: "12px", transition: "all 0.2s"
                    }}>
                      {kw}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Logs */}
          <div className="dashboard-card" style={{ padding: "0", display: "flex", flexDirection: "column", minHeight: "150px", maxHeight: "250px" }}>
            <div style={{ padding: "12px 16px", borderBottom: `1px solid ${colors.border}`, display: "flex", justifyContent: "space-between", fontSize: "13px", fontWeight: 500, color: colors.accent }}>
              <span>Event History</span>
              <span style={{ cursor: "pointer" }} onClick={() => setEventLog([])}>Clear</span>
            </div>
            <div style={{ padding: "12px 16px", overflowY: "auto", flex: 1, display: "flex", flexDirection: "column", gap: "10px" }}>
              {eventLog.map((e, i) => (
                <div key={i} style={{ display: "flex", gap: "12px", fontSize: "13px" }}>
                  <span style={{ color: colors.accent, whiteSpace: "nowrap" }}>{e.time}</span>
                  <span style={{ color: e.type === "err" ? colors.err : (e.type === "ok" ? colors.ok : colors.text) }}>{e.msg}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Footer Controls */}
          <div style={{ display: "flex", justifyContent: "flex-end", gap: "16px", marginTop: "auto", paddingTop: "12px" }}>
            <span 
              onClick={() => setShowSettings(true)}
              style={{ 
                cursor: "pointer", fontSize: "16px", color: colors.text, opacity: 0.3, 
                transition: "opacity 0.2s"
              }}
              onMouseEnter={(e) => e.currentTarget.style.opacity = 1}
              onMouseLeave={(e) => e.currentTarget.style.opacity = 0.3}
              title="Engine Settings"
            >
              ⚙️
            </span>
            <span 
              onClick={sendShutdown}
              style={{ 
                cursor: "pointer", fontSize: "16px", color: colors.text, opacity: 0.3, 
                transition: "opacity 0.2s"
              }}
              onMouseEnter={(e) => e.currentTarget.style.opacity = 1}
              onMouseLeave={(e) => e.currentTarget.style.opacity = 0.3}
              title="Shutdown Engine"
            >
              ⏻
            </span>
          </div>

        </div>
      </div>

      {/* Settings Modal Overlay */}
      {showSettings && (
        <div style={{
          position: "fixed", top: 0, left: 0, width: "100%", height: "100%",
          backgroundColor: "rgba(0,0,0,0.6)", zIndex: 1000,
          display: "flex", justifyContent: "center", alignItems: "center"
        }}>
          <div style={{
            backgroundColor: colors.panel, border: `1px solid ${colors.border}`,
            borderRadius: "12px", padding: "24px", width: "400px", maxWidth: "90%",
            boxShadow: "0 10px 30px rgba(0,0,0,0.3)"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
              <h2 style={{ margin: 0, fontSize: "18px", color: colors.accent, fontWeight: 600 }}>Engine Configuration</h2>
              <span onClick={() => setShowSettings(false)} style={{ cursor: "pointer", fontSize: "18px", opacity: 0.6 }}>✕</span>
            </div>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: "14px", fontWeight: 600 }}>Threat Assessment (Multi-Lock)</div>
                  <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Use MiDaS Z-Axis Depth to snap to the closest person.</div>
                </div>
                <label style={{ display: "flex", alignItems: "center", cursor: "pointer" }}>
                  <input type="checkbox" checked={config.depth} onChange={() => toggleConfig("depth")} style={{ display: "none" }} />
                  <div style={{
                    width: "40px", height: "20px", borderRadius: "10px", 
                    backgroundColor: config.depth ? colors.accent : colors.border,
                    position: "relative", transition: "background-color 0.2s"
                  }}>
                    <div style={{
                      width: "16px", height: "16px", borderRadius: "50%", backgroundColor: "#fff",
                      position: "absolute", top: "2px", left: config.depth ? "22px" : "2px",
                      transition: "left 0.2s"
                    }} />
                  </div>
                </label>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: "14px", fontWeight: 600 }}>Gesture Authorization</div>
                  <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Only track subjects who raise their hand.</div>
                </div>
                <label style={{ display: "flex", alignItems: "center", cursor: "pointer" }}>
                  <input type="checkbox" checked={config.gesture} onChange={() => toggleConfig("gesture")} style={{ display: "none" }} />
                  <div style={{
                    width: "40px", height: "20px", borderRadius: "10px", 
                    backgroundColor: config.gesture ? colors.accent : colors.border,
                    position: "relative", transition: "background-color 0.2s"
                  }}>
                    <div style={{
                      width: "16px", height: "16px", borderRadius: "50%", backgroundColor: "#fff",
                      position: "absolute", top: "2px", left: config.gesture ? "22px" : "2px",
                      transition: "left 0.2s"
                    }} />
                  </div>
                </label>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "8px", paddingTop: "16px", borderTop: `1px solid ${colors.border}` }}>
                <div>
                  <div style={{ fontSize: "14px", fontWeight: 600 }}>Crosshair Style</div>
                  <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Visual design of the targeting HUD.</div>
                </div>
                <select 
                  value={config.crosshairStyle} 
                  onChange={(e) => setConfigValue("crosshairStyle", e.target.value)}
                  style={{
                    backgroundColor: colors.bg, color: colors.text, border: `1px solid ${colors.border}`,
                    padding: "6px 12px", borderRadius: "6px", fontSize: "13px", outline: "none", cursor: "pointer"
                  }}
                >
                  <option value="military">Military (Default)</option>
                  <option value="classic">Classic</option>
                  <option value="minimal">Minimalist</option>
                  <option value="sniper">Sniper Scope</option>
                  <option value="dot">Dot Only</option>
                </select>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: "14px", fontWeight: 600 }}>HUD Color</div>
                  <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Primary color for active locks.</div>
                </div>
                <select 
                  value={config.hudColor} 
                  onChange={(e) => setConfigValue("hudColor", e.target.value)}
                  style={{
                    backgroundColor: colors.bg, color: colors.text, border: `1px solid ${colors.border}`,
                    padding: "6px 12px", borderRadius: "6px", fontSize: "13px", outline: "none", cursor: "pointer"
                  }}
                >
                  <option value="green">Neon Green (Default)</option>
                  <option value="cyan">Tactical Cyan</option>
                  <option value="red">Danger Red</option>
                  <option value="amber">Amber</option>
                  <option value="purple">Synthwave Purple</option>
                  <option value="white">Minimal White</option>
                </select>
              </div>

              {/* Divider */}
              <div style={{ borderTop: `1px solid ${colors.border}`, margin: "8px 0" }} />

              {/* Multi-Target Display */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: "14px", fontWeight: 600 }}>Multi-Target Display</div>
                  <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Show dim secondary markers on all detected people.</div>
                </div>
                <label style={{ display: "flex", alignItems: "center", cursor: "pointer" }}>
                  <input type="checkbox" checked={config.multiTarget} onChange={() => toggleConfig("multiTarget")} style={{ display: "none" }} />
                  <div style={{ width: "40px", height: "20px", borderRadius: "10px", backgroundColor: config.multiTarget ? colors.accent : colors.border, position: "relative", transition: "background-color 0.2s" }}>
                    <div style={{ width: "16px", height: "16px", borderRadius: "50%", backgroundColor: "#fff", position: "absolute", top: "2px", left: config.multiTarget ? "22px" : "2px", transition: "left 0.2s" }} />
                  </div>
                </label>
              </div>

              {/* Voice Commands */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: "14px", fontWeight: 600 }}>🎤 Voice Commands</div>
                  <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Say "lock nose" or "track left shoulder". Requires Whisper.</div>
                </div>
                <label style={{ display: "flex", alignItems: "center", cursor: "pointer" }}>
                  <input type="checkbox" checked={config.voice} onChange={() => toggleConfig("voice")} style={{ display: "none" }} />
                  <div style={{ width: "40px", height: "20px", borderRadius: "10px", backgroundColor: config.voice ? colors.accent : colors.border, position: "relative", transition: "background-color 0.2s" }}>
                    <div style={{ width: "16px", height: "16px", borderRadius: "50%", backgroundColor: "#fff", position: "absolute", top: "2px", left: config.voice ? "22px" : "2px", transition: "left 0.2s" }} />
                  </div>
                </label>
              </div>

              {/* Confidence Threshold */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <div>
                    <div style={{ fontSize: "14px", fontWeight: 600 }}>Confidence Threshold</div>
                    <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Min YOLO certainty required to commit a lock.</div>
                  </div>
                  <span style={{ fontSize: "14px", fontWeight: 600, color: colors.accent }}>{Math.round(config.confidence * 100)}%</span>
                </div>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={config.confidence}
                  onChange={(e) => setConfig(prev => ({ ...prev, confidence: parseFloat(e.target.value) }))}
                  onMouseUp={(e) => setConfigValue("confidence", parseFloat(e.target.value))}
                  onTouchEnd={(e) => setConfigValue("confidence", parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: colors.accent, cursor: "pointer" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", opacity: 0.5, marginTop: "4px" }}>
                  <span>Aggressive (0%)</span><span>Strict (100%)</span>
                </div>
              </div>
            </div>
            
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
