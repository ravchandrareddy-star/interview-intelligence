import { ai } from '@/ai/genkit';
import { textEmbedding004 } from '@genkit-ai/googleai';

export const EXPECTED_EMBEDDING_DIM = 768;

/**
 * Deterministic hash-based fallback embedding (768 dimensions) used if the
 * Gemini API key is missing or the external embedding call fails.
 * This guarantees the RAG pipeline continues functioning reliably in all environments.
 */
export function generateDeterministicEmbedding(text: string, dim: number = EXPECTED_EMBEDDING_DIM): number[] {
  const vector = new Array(dim).fill(0);
  const words = text.toLowerCase().replace(/[^a-z0-9_\-\s]/g, '').split(/\s+/).filter(Boolean);

  if (words.length === 0) {
    // Return unit vector for empty strings
    vector[0] = 1;
    return vector;
  }

  for (let wIdx = 0; wIdx < words.length; wIdx++) {
    const word = words[wIdx];
    let hash = 0;
    for (let c = 0; c < word.length; c++) {
      hash = (hash << 5) - hash + word.charCodeAt(c);
      hash |= 0;
    }

    const primaryIdx = Math.abs(hash) % dim;
    const secondaryIdx = Math.abs(hash * 31 + 7) % dim;

    vector[primaryIdx] += 1.0;
    vector[secondaryIdx] += 0.5;
  }

  // Normalize to unit length (L2 norm)
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm);

  if (norm > 0) {
    for (let i = 0; i < dim; i++) {
      vector[i] = vector[i] / norm;
    }
  }

  return vector;
}

/**
 * Generates an embedding for a string using Google's text-embedding-004 model via Genkit.
 * If the API is unconfigured or encounters an error, falls back gracefully.
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  const cleanText = text ? text.trim() : '';
  if (!cleanText) {
    return generateDeterministicEmbedding('', EXPECTED_EMBEDDING_DIM);
  }

  const apiKey = process.env.GEMINI_API_KEY;

  if (apiKey) {
    try {
      const response = await ai.embed({
        embedder: textEmbedding004,
        content: cleanText,
      });

      // Genkit embed returns an array of embeddings or objects with an `embedding` property
      let rawVector: number[] | null = null;
      if (Array.isArray(response)) {
        if (response.length > 0) {
          const first = response[0] as unknown;
          if (first && typeof first === 'object' && 'embedding' in first && Array.isArray((first as { embedding: number[] }).embedding)) {
            rawVector = (first as { embedding: number[] }).embedding;
          } else if (Array.isArray(first)) {
            rawVector = first as number[];
          }
        }
      } else if (response && typeof response === 'object' && 'embedding' in response) {
        rawVector = (response as { embedding: number[] }).embedding;
      }

      if (rawVector && Array.isArray(rawVector) && rawVector.length > 0) {
        return rawVector;
      }
    } catch (error) {
      console.warn('[RAG Embedder] Google AI embedding call failed, falling back to deterministic embedding:', (error as Error)?.message || error);
    }
  }

  return generateDeterministicEmbedding(cleanText, EXPECTED_EMBEDDING_DIM);
}

