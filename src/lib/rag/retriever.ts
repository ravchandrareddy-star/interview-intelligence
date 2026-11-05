import { RetrievedChunk, VectorRecord, JobRole } from './types';
import { getVectorStore } from './vector-store';
import { generateEmbedding } from './embedder';

export interface RetrievalOptions {
  topK?: number;
  similarityThreshold?: number;
}

const DEFAULT_TOP_K = 5;
const DEFAULT_SIMILARITY_THRESHOLD = 0.35;

/**
 * Computes cosine similarity between two numeric vectors.
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) {
    return 0;
  }

  const length = Math.min(vecA.length, vecB.length);
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < length; i++) {
    const a = vecA[i];
    const b = vecB[i];
    dotProduct += a * b;
    normA += a * a;
    normB += b * b;
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Normalizes input job role to one of the canonical job roles or returns 'General'.
 */
export function normalizeJobRole(rawRole: string): JobRole {
  const lower = rawRole.toLowerCase();

  if (lower.includes('front') || lower.includes('react') || lower.includes('ui') || lower.includes('web dev')) {
    return 'Frontend Developer';
  }
  if (lower.includes('back') || lower.includes('api') || lower.includes('server') || lower.includes('node') || lower.includes('database')) {
    return 'Backend Developer';
  }
  if (lower.includes('full') || lower.includes('mern') || lower.includes('mean') || lower.includes('stack')) {
    return 'Full Stack Developer';
  }
  if (lower.includes('data') || lower.includes('analyst') || lower.includes('analytics') || lower.includes('bi ') || lower.includes('business intel')) {
    return 'Data Analyst';
  }
  if (lower.includes('soft') || lower.includes('sde') || lower.includes('swe') || lower.includes('engineer') || lower.includes('developer') || lower.includes('programmer')) {
    return 'Software Developer';
  }

  return 'General';
}

/**
 * Constructs a retrieval query incorporating the target job role and salient resume keywords.
 */
export function constructRetrievalQuery(jobRole: string, resumeText: string): string {
  // Extract key technical words or summary highlights from the resume
  const cleanRole = jobRole.trim();
  const resumeSnippet = resumeText.slice(0, 500).replace(/\s+/g, ' ').trim();
  return `Target role: ${cleanRole}. Industry technical requirements, programming skills, frameworks, tools, ATS optimization criteria, and professional summary evaluation: ${resumeSnippet}`;
}

/**
 * Retrieves top-k relevant knowledge chunks from vector store using cosine similarity.
 */
export async function retrieveRelevantKnowledge(
  jobRole: string,
  resumeText: string,
  options: RetrievalOptions = {}
): Promise<RetrievedChunk[]> {
  const topK = options.topK ?? DEFAULT_TOP_K;
  const threshold = options.similarityThreshold ?? DEFAULT_SIMILARITY_THRESHOLD;

  const records: VectorRecord[] = await getVectorStore();
  if (!records || records.length === 0) {
    console.warn('[RAG Retriever] Vector store is empty, returning empty context.');
    return [];
  }

  const query = constructRetrievalQuery(jobRole, resumeText);
  const queryEmbedding = await generateEmbedding(query);

  const canonicalRole = normalizeJobRole(jobRole);

  const scored: RetrievedChunk[] = [];

  for (const record of records) {
    let similarity = cosineSimilarity(queryEmbedding, record.embedding);

    // Apply a slight relevance bonus if the chunk matches the target job role or is universal ATS guidance
    const isDirectRoleMatch = record.jobRole.toLowerCase() === canonicalRole.toLowerCase();
    const isGeneralGuideline = record.jobRole === 'General';

    if (isDirectRoleMatch) {
      similarity += 0.08;
    } else if (isGeneralGuideline) {
      similarity += 0.04;
    }

    // Only include if it meets the similarity threshold
    if (similarity >= threshold) {
      scored.push({
        ...record,
        similarity: Math.min(1.0, Math.max(0, similarity)),
      });
    }
  }

  // Sort descending by similarity score
  scored.sort((a, b) => b.similarity - a.similarity);

  // Return top K chunks
  return scored.slice(0, topK);
}

/**
 * Formats retrieved chunks into a structured prompt context string for Gemini.
 */
export function formatRetrievedContext(chunks: RetrievedChunk[]): string {
  if (!chunks || chunks.length === 0) {
    return 'No additional job role knowledge retrieved.';
  }

  return chunks
    .map((chunk, idx) => {
      return `[Context Chunk ${idx + 1}] (Role: ${chunk.jobRole} | Category: ${chunk.category} | Source: ${chunk.source})
Title: ${chunk.title}
Content: ${chunk.text}`;
    })
    .join('\n\n');
}

