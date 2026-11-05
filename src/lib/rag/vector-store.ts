import fs from 'fs';
import path from 'path';
import { VectorRecord } from './types';
import { loadKnowledgeDocuments } from './loader';
import { chunkAllDocuments } from './chunker';
import { generateEmbedding, EXPECTED_EMBEDDING_DIM } from './embedder';

let memoryCache: VectorRecord[] | null = null;

/**
 * Resolves the persistent vector storage JSON file path.
 */
export function getVectorStoreFilePath(): string {
  const candidates = [
    path.join(process.cwd(), 'src', 'lib', 'rag', 'data', 'vector-store.json'),
    path.join(__dirname, 'data', 'vector-store.json'),
    path.join(process.cwd(), 'AI-MOCK-INTERVIEW--main', 'src', 'lib', 'rag', 'data', 'vector-store.json'),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  // Default to the first candidate path if not yet created
  const targetPath = candidates[0];
  const targetDir = path.dirname(targetPath);
  if (!fs.existsSync(targetDir)) {
    fs.mkdirSync(targetDir, { recursive: true });
  }
  return targetPath;
}

/**
 * Validates a single vector record.
 */
export function isValidVectorRecord(record: unknown): record is VectorRecord {
  if (!record || typeof record !== 'object') return false;
  const r = record as Partial<VectorRecord>;
  if (!r.chunkId || !r.text || !Array.isArray(r.embedding)) return false;
  if (r.embedding.length === 0) return false;
  // Ensure elements are valid numbers and not NaN
  for (let i = 0; i < Math.min(r.embedding.length, 10); i++) {
    if (typeof r.embedding[i] !== 'number' || isNaN(r.embedding[i])) {
      return false;
    }
  }
  return true;
}

/**
 * Loads vector records from disk or memory cache.
 * Reuses existing precomputed embeddings.
 */
export async function getVectorStore(): Promise<VectorRecord[]> {
  if (memoryCache && memoryCache.length > 0) {
    return memoryCache;
  }

  const filePath = getVectorStoreFilePath();

  if (fs.existsSync(filePath)) {
    try {
      const raw = fs.readFileSync(filePath, 'utf-8');
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const validRecords: VectorRecord[] = [];
        let hasInvalid = false;

        for (const item of parsed) {
          if (isValidVectorRecord(item)) {
            validRecords.push(item);
          } else {
            hasInvalid = true;
          }
        }

        if (validRecords.length > 0 && !hasInvalid) {
          memoryCache = validRecords;
          return validRecords;
        } else if (validRecords.length > 0) {
          console.warn(`[Vector Store] Filtered out invalid records. Kept ${validRecords.length} valid records.`);
          memoryCache = validRecords;
          return validRecords;
        }
      }
    } catch (err) {
      console.warn('[Vector Store] Error reading existing vector store file. Will rebuild:', err);
    }
  }

  // If file does not exist or was corrupted, build and persist it
  return await buildVectorStore();
}

/**
 * Builds or rebuilds the vector store by loading knowledge docs, chunking them,
 * generating embeddings, and persisting to disk.
 */
export async function buildVectorStore(): Promise<VectorRecord[]> {
  console.log('[Vector Store] Building vector store from knowledge base documents...');
  const docs = loadKnowledgeDocuments();
  if (docs.length === 0) {
    console.warn('[Vector Store] No documents found to index.');
    return [];
  }

  const chunks = chunkAllDocuments(docs);
  const records: VectorRecord[] = [];

  for (const chunk of chunks) {
    // Generate embedding for chunk
    const embedding = await generateEmbedding(`${chunk.title}: ${chunk.text}`);
    records.push({
      ...chunk,
      embedding,
    });
  }

  // Persist to disk
  const filePath = getVectorStoreFilePath();
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(filePath, JSON.stringify(records, null, 2), 'utf-8');
  console.log(`[Vector Store] Successfully built and saved ${records.length} vector records to ${filePath}`);

  memoryCache = records;
  return records;
}

/**
 * Clears the in-memory cache (for testing purposes).
 */
export function clearVectorStoreCache(): void {
  memoryCache = null;
}

