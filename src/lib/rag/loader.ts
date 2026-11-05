import fs from 'fs';
import path from 'path';
import { KnowledgeDocument } from './types';

let cachedDocuments: KnowledgeDocument[] | null = null;

/**
 * Loads all knowledge base documents from the knowledge directory.
 * Preserves all metadata, handles missing or empty documents, and deduplicates.
 */
export function loadKnowledgeDocuments(): KnowledgeDocument[] {
  if (cachedDocuments && cachedDocuments.length > 0) {
    return cachedDocuments;
  }

  // Resolve directory path relative to current working directory or source
  const knowledgeDirCandidates = [
    path.join(process.cwd(), 'src', 'lib', 'rag', 'knowledge'),
    path.join(__dirname, 'knowledge'),
    path.join(process.cwd(), 'AI-MOCK-INTERVIEW--main', 'src', 'lib', 'rag', 'knowledge'),
  ];

  let resolvedDir: string | null = null;
  for (const candidate of knowledgeDirCandidates) {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
      resolvedDir = candidate;
      break;
    }
  }

  if (!resolvedDir) {
    console.warn('[RAG Loader] Knowledge directory not found. Checked:', knowledgeDirCandidates);
    return [];
  }

  const documents: KnowledgeDocument[] = [];
  const seenIds = new Set<string>();

  try {
    const files = fs.readdirSync(resolvedDir);
    for (const file of files) {
      if (!file.endsWith('.json')) continue;

      const filePath = path.join(resolvedDir, file);
      try {
        const fileContent = fs.readFileSync(filePath, 'utf-8').trim();
        if (!fileContent) {
          console.warn(`[RAG Loader] Skipping empty file: ${file}`);
          continue;
        }

        const parsed = JSON.parse(fileContent);
        const docList: KnowledgeDocument[] = Array.isArray(parsed) ? parsed : [parsed];

        for (const doc of docList) {
          if (!doc || !doc.id || !doc.content || !doc.content.trim()) {
            console.warn(`[RAG Loader] Skipping invalid or empty document in ${file}:`, doc);
            continue;
          }

          if (seenIds.has(doc.id)) {
            console.warn(`[RAG Loader] Skipping duplicate document ID: ${doc.id}`);
            continue;
          }

          seenIds.add(doc.id);
          documents.push({
            id: doc.id,
            jobRole: doc.jobRole || 'General',
            category: doc.category || 'job-requirements',
            source: doc.source || file,
            title: doc.title || 'Knowledge Reference',
            content: doc.content.trim(),
            metadata: doc.metadata || {},
          });
        }
      } catch (err) {
        console.error(`[RAG Loader] Error parsing knowledge file ${file}:`, err);
      }
    }
  } catch (err) {
    console.error(`[RAG Loader] Error reading knowledge directory ${resolvedDir}:`, err);
  }

  cachedDocuments = documents;
  return documents;
}

/**
 * Resets the in-memory document cache (useful for testing or hot-reloading).
 */
export function clearDocumentCache(): void {
  cachedDocuments = null;
}

