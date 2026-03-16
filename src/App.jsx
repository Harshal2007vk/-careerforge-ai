import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";
import Assessment from "./pages/Assessment";
import Results from "./pages/Results";

function Home() {
  const navigate = useNavigate();
  return (
    <div style={{
      minHeight: "100vh", background: "#0f0f1a",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      color: "white", fontFamily: "sans-serif", textAlign: "center", padding: "2rem"
    }}>
      <div style={{ fontSize: "4rem", marginBottom: "1rem" }}>🎯</div>
      <h1 style={{ fontSize: "2.5rem", fontWeight: "700", color: "#a78bfa", marginBottom: "1rem" }}>
        AI Career Guidance
      </h1>
      <p style={{ color: "#888", fontSize: "1.1rem", maxWidth: "480px", marginBottom: "2rem" }}>
        Discover your perfect career path using AI. Answer a few questions and get personalized recommendations powered by Claude.
      </p>
      <button onClick={() => navigate("/assessment")} style={{
        padding: "16px 40px", background: "#a78bfa", border: "none",
        borderRadius: "10px", color: "white", fontSize: "1.1rem",
        fontWeight: "600", cursor: "pointer"
      }}>
        Start Assessment 🚀
      </button>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/assessment" element={<Assessment />} />
        <Route path="/results" element={<Results />} />
      </Routes>
    </BrowserRouter>
  );
}