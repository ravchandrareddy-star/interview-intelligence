import { KnowledgeChunk, KnowledgeDocument } from './types';

export interface ChunkOptions {
  maxChunkSize?: number;
  chunkOverlap?: number;
}

const DEFAULT_MAX_CHUNK_SIZE = 500;
const DEFAULT_CHUNK_OVERLAP = 80;

/**
 * Splits text into meaningful sentences/segments while preserving technical terms.
 */
function splitIntoSentences(text: string): string[] {
  // Normalize whitespace
  const normalized = text.replace(/\r\n/g, '\n').trim();
  // Split on paragraph or sentence boundaries (. ! ?) followed by whitespace or newline
  const rawSegments = normalized.split(/(?<=[.!?])\s+|\n+/);
  return rawSegments.map(s => s.trim()).filter(s => s.length > 0);
}

/**
 * Chunks a single KnowledgeDocument into semantic chunks with metadata preserved.
 */
export function chunkDocument(
  doc: KnowledgeDocument,
  options: ChunkOptions = {}
): KnowledgeChunk[] {
  const maxChunkSize = options.maxChunkSize || DEFAULT_MAX_CHUNK_SIZE;
  const chunkOverlap = options.chunkOverlap || DEFAULT_CHUNK_OVERLAP;

  if (!doc.content || !doc.content.trim()) {
    return [];
  }

  const text = doc.content.trim();

  // If text is already concise and within maxChunkSize, keep as a single cohesive chunk
  if (text.length <= maxChunkSize) {
    return [
      {
        chunkId: `${doc.id}-chunk-0`,
        documentId: doc.id,
        jobRole: doc.jobRole,
        category: doc.category,
        source: doc.source,
        title: doc.title,
        chunkIndex: 0,
        text,
      },
    ];
  }

  const sentences = splitIntoSentences(text);
  const chunks: KnowledgeChunk[] = [];
  let currentChunkText = '';
  let chunkIndex = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];

    if (!currentChunkText) {
      currentChunkText = sentence;
    } else if (currentChunkText.length + sentence.length + 1 <= maxChunkSize) {
      currentChunkText += ' ' + sentence;
    } else {
      // Current chunk is full, record it
      chunks.push({
        chunkId: `${doc.id}-chunk-${chunkIndex}`,
        documentId: doc.id,
        jobRole: doc.jobRole,
        category: doc.category,
        source: doc.source,
        title: doc.title,
        chunkIndex,
        text: currentChunkText.trim(),
      });
      chunkIndex++;

      // Create overlap from the end of currentChunkText
      const overlapStart = Math.max(0, currentChunkText.length - chunkOverlap);
      const overlapSlice = currentChunkText.slice(overlapStart).trim();
      currentChunkText = overlapSlice ? `${overlapSlice} ${sentence}` : sentence;
    }
  }

  if (currentChunkText.trim()) {
    chunks.push({
      chunkId: `${doc.id}-chunk-${chunkIndex}`,
      documentId: doc.id,
      jobRole: doc.jobRole,
      category: doc.category,
      source: doc.source,
      title: doc.title,
      chunkIndex,
      text: currentChunkText.trim(),
    });
  }

  return chunks;
}

/**
 * Chunks an array of KnowledgeDocuments.
 */
export function chunkAllDocuments(
  docs: KnowledgeDocument[],
  options: ChunkOptions = {}
): KnowledgeChunk[] {
  const allChunks: KnowledgeChunk[] = [];
  for (const doc of docs) {
    const docChunks = chunkDocument(doc, options);
    allChunks.push(...docChunks);
  }
  return allChunks;
}

