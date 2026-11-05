'use server';

/**
 * @fileOverview An AI flow for analyzing a user's resume against a specific job role,
 * augmented with RAG retrieval from an industry knowledge base.
 *
 * - analyzeResume - A function that handles the resume analysis process.
 * - AnalyzeResumeInput - The input type for the analyzeResume function.
 * - AnalyzeResumeOutput - The return type for the analyzeResume function.
 */

import { ai } from '@/ai/genkit';
import { z } from 'genkit';
import { retrieveRelevantKnowledge, formatRetrievedContext } from '@/lib/rag/retriever';

const AnalyzeResumeInputSchema = z.object({
  resumeText: z.string().describe("The full text content of the user's resume."),
  jobRole: z.string().describe('The target job role the user is applying for.'),
});
export type AnalyzeResumeInput = z.infer<typeof AnalyzeResumeInputSchema>;

const ResumeAnalyzerInternalInputSchema = z.object({
  resumeText: z.string().describe("The full text content of the user's resume."),
  jobRole: z.string().describe('The target job role the user is applying for.'),
  retrievedContext: z.string().describe('Retrieved job role requirements, skills, ATS criteria, and industry standards from the RAG knowledge base.'),
});

const AnalyzeResumeOutputSchema = z.object({
  weaknesses: z.array(z.string()).describe("A list of key weaknesses or missing elements in the resume."),
  suggestions: z.array(z.string()).describe("A list of actionable suggestions for improving bullet points, skills, or overall presentation."),
  rewritten_summary: z.string().optional().describe("A rewritten, more impactful summary section if the original was too vague or weak. Null if the original is good."),
  recommendation_score: z.number().min(0).max(10).describe("An overall score from 0-10 indicating how well the resume is aligned with the target job role."),
});
export type AnalyzeResumeOutput = z.infer<typeof AnalyzeResumeOutputSchema>;

export async function analyzeResume(input: AnalyzeResumeInput): Promise<AnalyzeResumeOutput> {
  return analyzeResumeFlow(input);
}

const prompt = ai.definePrompt({
  name: 'resumeAnalyzerPrompt',
  input: { schema: ResumeAnalyzerInternalInputSchema },
  output: { schema: AnalyzeResumeOutputSchema },
  prompt: `You are a professional career coach and an expert technical recruiter at a top tech company. Your task is to review the provided resume content for a specific job role and provide a rigorous, constructive, context-aware analysis powered by verified industry benchmarks.

  **Target Job Role:**
  {{{jobRole}}}

  **Retrieved Job-Role & Industry Knowledge Context (from RAG Knowledge Base):**
  {{{retrievedContext}}}

  **Resume Content:**
  {{{resumeText}}}

  **Your Analysis Instructions:**

  1.  **Skills Alignment & Key Weaknesses:**
      - Compare the candidate's stated skills against the technical skills, frameworks, and tooling retrieved from the knowledge base for this role.
      - Identify critical missing or weakly represented skills that are standard expectations for this role.
      - Evaluate resume structure, impact metrics, and ATS compliance based on the retrieved ATS and evaluation criteria.
      - Be specific with clear examples (e.g., "Missing essential backend technologies like PostgreSQL or Redis mentioned in industry benchmarks," "Lacks quantifiable metrics in project bullet points," "Summary lacks clear years of experience and core tech specialization").

  2.  **Actionable Improvement Suggestions:**
      - Provide concrete, prioritized suggestions for improving the resume.
      - Advise how to reframe bullet points using the XYZ or STAR formula (Accomplished [X] as measured by [Y] by doing [Z]).
      - Recommend adding role-critical keywords and frameworks identified from the retrieved context.
      - Guide ATS readability and section layout improvements.

  3.  **Rewritten Summary:**
      - If the candidate's summary/objective is weak, vague, generic, or missing, generate a concise, powerful, 2-3 sentence professional summary tailored to the target role using the retrieved summary guidelines.
      - Never fabricate unearned qualifications or experiences; build upon genuine skills evident in the resume.
      - If the existing summary is already strong and impactful, return null.

  4.  **Recommendation Score:**
      - Provide an overall score from 0 (very poor match) to 10 (exceptional match) that objectively reflects how well the resume aligns with the target job role and the retrieved industry expectations.

  **Accuracy Rules:**
  - Never invent candidate experience or claim skills not supported by the resume.
  - Distinguish missing skills from skills that are simply not mentioned.
  - Do not guarantee ATS selection.
  - Use retrieved knowledge as benchmark context.

  Your response MUST be a JSON object that strictly follows the defined output schema.
  `,
  model: 'googleai/gemini-2.5-flash',
});

const analyzeResumeFlow = ai.defineFlow(
  {
    name: 'analyzeResumeFlow',
    inputSchema: AnalyzeResumeInputSchema,
    outputSchema: AnalyzeResumeOutputSchema,
  },
  async input => {
    // 1. Retrieve relevant job-role knowledge from the RAG vector store
    let retrievedContext = 'No external knowledge retrieved.';
    try {
      const retrievedChunks = await retrieveRelevantKnowledge(input.jobRole, input.resumeText, {
        topK: 5,
        similarityThreshold: 0.30,
      });
      retrievedContext = formatRetrievedContext(retrievedChunks);
    } catch (retrievalError) {
      console.warn('[Resume Analyzer Flow] Knowledge retrieval failed, proceeding with baseline context:', retrievalError);
    }

    // 2. Pass resume, job role, and retrieved knowledge context to Gemini 2.5 Flash
    const { output } = await prompt({
      jobRole: input.jobRole,
      resumeText: input.resumeText,
      retrievedContext,
    });

    return output!;
  }
);
