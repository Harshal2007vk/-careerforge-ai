import { useState } from "react";
import { useNavigate } from "react-router-dom";

const interests = ["Technology", "Design", "Business", "Science", "Arts", "Healthcare", "Education", "Finance"];

const skillsList = [
  "Programming", "Problem Solving", "Communication",
  "Mathematics", "Creativity", "Leadership",
  "Data Analysis", "Teamwork"
];

export default function Assessment() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [profile, setProfile] = useState({
    name: "",
    selectedInterests: [],
    skills: {},
    learningStyle: ""
  });

  const toggleInterest = (item) => {
    setProfile(p => ({
      ...p,
      selectedInterests: p.selectedInterests.includes(item)
        ? p.selectedInterests.filter(i => i !== item)
        : [...p.selectedInterests, item]
    }));
  };

  const setSkill = (skill, value) => {
    setProfile(p => ({ ...p, skills: { ...p.skills, [skill]: value } }));
  };

  const handleSubmit = () => {
    navigate("/results", { state: { profile } });
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f1a", color: "white", padding: "2rem", fontFamily: "sans-serif" }}>
      <div style={{ maxWidth: "600px", margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: "2rem" }}>
          <h1 style={{ fontSize: "1.8rem", fontWeight: "700", color: "#a78bfa" }}>AI Career Guidance</h1>
          <div style={{ display: "flex", gap: "8px", marginTop: "1rem" }}>
            {[1, 2, 3].map(s => (
              <div key={s} style={{
                height: "4px", flex: 1, borderRadius: "2px",
                background: s <= step ? "#a78bfa" : "#2d2d44"
              }} />
            ))}
          </div>
          <p style={{ color: "#888", marginTop: "0.5rem", fontSize: "0.9rem" }}>Step {step} of 3</p>
        </div>

        {/* Step 1 - Name & Interests */}
        {step === 1 && (
          <div>
            <h2 style={{ marginBottom: "1rem" }}>Tell us about yourself</h2>
            <input
              placeholder="Your name"
              value={profile.name}
              onChange={e => setProfile(p => ({ ...p, name: e.target.value }))}
              style={{
                width: "100%", padding: "12px", borderRadius: "8px",
                background: "#1e1e2e", border: "1px solid #3d3d5c",
                color: "white", fontSize: "1rem", marginBottom: "1.5rem",
                boxSizing: "border-box"
              }}
            />
            <h3 style={{ marginBottom: "1rem" }}>Select your interests</h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
              {interests.map(item => (
                <button key={item} onClick={() => toggleInterest(item)} style={{
                  padding: "8px 16px", borderRadius: "20px", cursor: "pointer",
                  border: "1px solid",
                  borderColor: profile.selectedInterests.includes(item) ? "#a78bfa" : "#3d3d5c",
                  background: profile.selectedInterests.includes(item) ? "#2d1f4e" : "#1e1e2e",
                  color: profile.selectedInterests.includes(item) ? "#a78bfa" : "#888",
                  fontSize: "0.9rem"
                }}>{item}</button>
              ))}
            </div>
            <button onClick={() => setStep(2)} disabled={!profile.name || profile.selectedInterests.length === 0}
              style={{
                marginTop: "2rem", width: "100%", padding: "14px",
                background: "#a78bfa", border: "none", borderRadius: "8px",
                color: "white", fontSize: "1rem", fontWeight: "600", cursor: "pointer"
              }}>Next →</button>
          </div>
        )}

        {/* Step 2 - Skills */}
        {step === 2 && (
          <div>
            <h2 style={{ marginBottom: "1.5rem" }}>Rate your skills</h2>
            {skillsList.map(skill => (
              <div key={skill} style={{ marginBottom: "1.2rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                  <span>{skill}</span>
                  <span style={{ color: "#a78bfa" }}>{profile.skills[skill] || 1}/5</span>
                </div>
                <input type="range" min="1" max="5" value={profile.skills[skill] || 1}
                  onChange={e => setSkill(skill, parseInt(e.target.value))}
                  style={{ width: "100%", accentColor: "#a78bfa" }}
                />
              </div>
            ))}
            <div style={{ display: "flex", gap: "10px", marginTop: "2rem" }}>
              <button onClick={() => setStep(1)} style={{
                flex: 1, padding: "14px", background: "#1e1e2e",
                border: "1px solid #3d3d5c", borderRadius: "8px", color: "white", cursor: "pointer"
              }}>← Back</button>
              <button onClick={() => setStep(3)} style={{
                flex: 2, padding: "14px", background: "#a78bfa",
                border: "none", borderRadius: "8px", color: "white",
                fontSize: "1rem", fontWeight: "600", cursor: "pointer"
              }}>Next →</button>
            </div>
          </div>
        )}

        {/* Step 3 - Learning Style */}
        {step === 3 && (
          <div>
            <h2 style={{ marginBottom: "1.5rem" }}>How do you learn best?</h2>
            {["Visual (videos, diagrams)", "Reading (books, articles)", "Hands-on (projects, practice)", "Social (group study, mentorship)"].map(style => (
              <button key={style} onClick={() => setProfile(p => ({ ...p, learningStyle: style }))}
                style={{
                  display: "block", width: "100%", padding: "14px 16px",
                  marginBottom: "10px", borderRadius: "8px", cursor: "pointer",
                  textAlign: "left", border: "1px solid",
                  borderColor: profile.learningStyle === style ? "#a78bfa" : "#3d3d5c",
                  background: profile.learningStyle === style ? "#2d1f4e" : "#1e1e2e",
                  color: profile.learningStyle === style ? "#a78bfa" : "#888",
                  fontSize: "0.95rem"
                }}>{style}</button>
            ))}
            <div style={{ display: "flex", gap: "10px", marginTop: "2rem" }}>
              <button onClick={() => setStep(2)} style={{
                flex: 1, padding: "14px", background: "#1e1e2e",
                border: "1px solid #3d3d5c", borderRadius: "8px", color: "white", cursor: "pointer"
              }}>← Back</button>
              <button onClick={handleSubmit} disabled={!profile.learningStyle}
                style={{
                  flex: 2, padding: "14px", background: "#a78bfa",
                  border: "none", borderRadius: "8px", color: "white",
                  fontSize: "1rem", fontWeight: "600", cursor: "pointer"
                }}>Get My Career Path 🚀</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
