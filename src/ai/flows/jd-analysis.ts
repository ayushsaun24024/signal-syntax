// This is an autogenerated file from Firebase Studio.
'use server';
/**
 * @fileOverview A Genkit tool for analyzing job descriptions against Ayush's skills.
 *
 * - jdAnalysisTool - A tool that can be used by an AI to perform a skill-gap analysis.
 */

import { ai } from '@/ai/genkit';
import { cvContentForBio, skills } from '@/lib/data';
import { z } from 'zod';

const JdAnalysisOutputSchema = z.object({
    matchPercentage: z.number().describe("The estimated match percentage (0-100) between the JD and Ayush's skills."),
    matchingSkills: z.array(z.string()).describe("A list of skills Ayush has that are mentioned in the JD."),
    missingSkills: z.array(z.string()).describe("A list of key skills from the JD that are not explicitly listed in Ayush's resume."),
    summary: z.string().describe("A brief, overall summary and recommendation for Ayush regarding this job opportunity.")
});

const ayushContext = `
Resume Summary: ${cvContentForBio}
Skillset: ${JSON.stringify(skills, null, 2)}
`;

const jdAnalysisPrompt = ai.definePrompt({
    name: 'jdAnalysisPrompt',
    input: { schema: z.string() },
    output: { schema: JdAnalysisOutputSchema },
    prompt: `You are an expert technical recruiter. Analyze the following Job Description (JD) against Ayush Saun's resume and skills, provided below.

Your task is to provide a structured analysis including a match percentage, a list of matching skills, a list of skill gaps, and a concise summary.

**Ayush's Context:**
---
${ayushContext}
---

**Job Description to Analyze:**
---
{{{input}}}
---
`,
});

export const jdAnalysisTool = ai.defineTool(
    {
        name: 'analyzeJobDescription',
        description: "Analyzes a job description against Ayush Saun's skills and experience to find how good a fit he is. Use this when the user pastes a large block of text that looks like a job description.",
        inputSchema: z.string(),
        outputSchema: JdAnalysisOutputSchema
    },
    async (jd) => {
        const { output } = await jdAnalysisPrompt(jd);
        return output!;
    }
);
