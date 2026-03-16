import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer } from "recharts";
import { getCareerRecommendations } from "../services/claudeApi";

export default function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const { profile } = location.state || {};
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeCareer, setActiveCareer] = useState(0);

  useEffect(() => {
    if (!profile) { navigate("/"); return; }
    getCareerRecommendations(profile)
      .then(result => { setData(result); setLoading(false); })
      .catch(err => { setError(err.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div style={{ minHeight: "100vh", background: "#0f0f1a", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "white", fontFamily: "sans-serif" }}>
      <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🤖</div>
      <h2 style={{ color: "#a78bfa" }}>Analyzing your profile...</h2>
      <p style={{ color: "#888" }}>Claude AI is finding your best career matches</p>
    </div>
  );

  if (error) return (
    <div style={{ minHeight: "100vh", background: "#0f0f1a", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "white", fontFamily: "sans-serif" }}>
      <h2 style={{ color: "#f87171" }}>Something went wrong</h2>
      <p style={{ color: "#888" }}>{error}</p>
      <button onClick={() => navigate("/")} style={{ marginTop: "1rem", padding: "10px 24px", background: "#a78bfa", border: "none", borderRadius: "8px", color: "white", cursor: "pointer" }}>Try Again</button>
    </div>
  );

  const radarData = data.skillGaps.map(s => ({
    skill: s.skill, Current: s.current, Required: s.required
  }));

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f1a", color: "white", padding: "2rem", fontFamily: "sans-serif" }}>
      <div style={{ maxWidth: "800px", margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: "2rem" }}>
          <h1 style={{ fontSize: "1.8rem", fontWeight: "700", color: "#a78bfa" }}>
            Your Career Report, {profile.name} 🎯
          </h1>
          <p style={{ color: "#888" }}>Based on your skills and interests, here are your top career matches</p>
        </div>

        {/* Career Cards */}
        <h2 style={{ marginBottom: "1rem" }}>Top Career Matches</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginBottom: "2rem" }}>
          {data.careerMatches.map((career, i) => (
            <div key={i} onClick={() => setActiveCareer(i)} style={{
              padding: "1.2rem", borderRadius: "12px", cursor: "pointer",
              border: "1px solid",
              borderColor: activeCareer === i ? "#a78bfa" : "#2d2d44",
              background: activeCareer === i ? "#1a1030" : "#141424",
              transition: "all 0.2s"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h3 style={{ margin: 0, color: activeCareer === i ? "#a78bfa" : "white" }}>{career.title}</h3>
                <span style={{
                  padding: "4px 12px", borderRadius: "20px", fontSize: "0.85rem", fontWeight: "600",
                  background: career.matchPercent >= 80 ? "#1a3a1a" : "#2a2a1a",
                  color: career.matchPercent >= 80 ? "#4ade80" : "#facc15"
                }}>{career.matchPercent}% match</span>
              </div>
              {activeCareer === i && (
                <div style={{ marginTop: "0.8rem" }}>
                  <p style={{ color: "#ccc", margin: "0 0 0.8rem" }}>{career.description}</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                    {career.requiredSkills.map((skill, j) => (
                      <span key={j} style={{
                        padding: "4px 10px", borderRadius: "12px", fontSize: "0.8rem",
                        background: "#2d1f4e", color: "#a78bfa"
                      }}>{skill}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Skill Gap Radar */}
        <h2 style={{ marginBottom: "1rem" }}>Skill Gap Analysis</h2>
        <div style={{ background: "#141424", borderRadius: "12px", padding: "1.5rem", marginBottom: "2rem", border: "1px solid #2d2d44" }}>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#2d2d44" />
              <PolarAngleAxis dataKey="skill" tick={{ fill: "#888", fontSize: 12 }} />
              <Radar name="Current" dataKey="Current" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.3} />
              <Radar name="Required" dataKey="Required" stroke="#4ade80" fill="#4ade80" fillOpacity={0.1} />
            </RadarChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: "1rem", justifyContent: "center", marginTop: "0.5rem" }}>
            <span style={{ color: "#a78bfa", fontSize: "0.85rem" }}>● Your Skills</span>
            <span style={{ color: "#4ade80", fontSize: "0.85rem" }}>● Required Skills</span>
          </div>
        </div>

        {/* Learning Roadmap */}
        <h2 style={{ marginBottom: "1rem" }}>Your Learning Roadmap</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "2rem" }}>
          {data.roadmap.map((item, i) => (
            <div key={i} style={{
              display: "flex", gap: "1rem", padding: "1rem",
              background: "#141424", borderRadius: "10px", border: "1px solid #2d2d44"
            }}>
              <div style={{
                minWidth: "48px", height: "48px", borderRadius: "50%",
                background: "#2d1f4e", display: "flex", alignItems: "center",
                justifyContent: "center", color: "#a78bfa", fontWeight: "700"
              }}>M{item.month}</div>
              <div>
                <p style={{ margin: "0 0 4px", fontWeight: "600" }}>{item.milestone}</p>
                <p style={{ margin: 0, color: "#888", fontSize: "0.85rem" }}>📚 {item.resources}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Restart */}
        <button onClick={() => navigate("/")} style={{
          width: "100%", padding: "14px", background: "#1e1e2e",
          border: "1px solid #3d3d5c", borderRadius: "8px",
          color: "#a78bfa", fontSize: "1rem", cursor: "pointer"
        }}>← Start New Assessment</button>

      </div>
    </div>
  );
}