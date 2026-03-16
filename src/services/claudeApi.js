const API_KEY = import.meta.env.VITE_ANTHROPIC_API_KEY;

export async function getCareerRecommendations(studentProfile) {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true"
    },
    body: JSON.stringify({
      model: "claude-opus-4-6",
      max_tokens: 2000,
      messages: [{
        role: "user",
        content: `You are a career guidance expert. Analyze this student profile and respond ONLY with a JSON object, no markdown, no backticks, no explanation, just raw JSON:

Student Profile:
${JSON.stringify(studentProfile, null, 2)}

Return exactly this JSON structure:
{
  "careerMatches": [
    {
      "title": "Career Title",
      "matchPercent": 85,
      "description": "2 sentence description",
      "requiredSkills": ["skill1", "skill2", "skill3"]
    }
  ],
  "skillGaps": [
    { "skill": "Skill Name", "current": 3, "required": 5 }
  ],
  "roadmap": [
    { "month": 1, "milestone": "What to do", "resources": "Course or resource name" }
  ]
}`
      }]
    })
  });

  const data = await response.json();
  const text = data.content[0].text;
  return JSON.parse(text);
}