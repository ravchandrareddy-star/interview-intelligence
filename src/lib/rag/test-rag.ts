/**
 * Comprehensive automated verification script for RAG Knowledge Base,
 * Chunker, Vector Store, Similarity Search, and Document Extraction.
 */

import { loadKnowledgeDocuments } from './loader';
import { chunkAllDocuments } from './chunker';
import { generateEmbedding, EXPECTED_EMBEDDING_DIM } from './embedder';
import { getVectorStore, isValidVectorRecord } from './vector-store';
import { cosineSimilarity, retrieveRelevantKnowledge, normalizeJobRole, formatRetrievedContext } from './retriever';
import { extractResumeText } from './document-extractor';

async function runTests() {
  console.log('=====================================================');
  console.log('Starting RAG Pipeline and Resume Analyzer Test Suite');
  console.log('=====================================================\n');

  let passed = 0;
  let failed = 0;

  function assert(condition: boolean, testName: string, detail?: string) {
    if (condition) {
      console.log(`[PASS] ${testName}`);
      passed++;
    } else {
      console.error(`[FAIL] ${testName} - Detail: ${detail || 'Assertion failed'}`);
      failed++;
    }
  }

  // 1. Document Loading Tests
  console.log('--- 1. Document Loading ---');
  const docs = loadKnowledgeDocuments();
  assert(docs.length >= 10, `Loaded ${docs.length} knowledge documents (expected >= 10)`);
  const hasMetadata = docs.every(d => d.id && d.jobRole && d.category && d.source && d.content);
  assert(hasMetadata, 'All documents have valid IDs, job roles, categories, sources, and content');

  // Verify roles
  const rolesFound = new Set(docs.map(d => d.jobRole));
  assert(rolesFound.has('Software Developer'), 'Software Developer documents present');
  assert(rolesFound.has('Frontend Developer'), 'Frontend Developer documents present');
  assert(rolesFound.has('Backend Developer'), 'Backend Developer documents present');
  assert(rolesFound.has('Full Stack Developer'), 'Full Stack Developer documents present');
  assert(rolesFound.has('Data Analyst'), 'Data Analyst documents present');
  assert(rolesFound.has('General'), 'General ATS and evaluation criteria documents present');

  // 2. Document Chunking Tests
  console.log('\n--- 2. Document Chunking ---');
  const chunks = chunkAllDocuments(docs, { maxChunkSize: 500, chunkOverlap: 80 });
  assert(chunks.length >= docs.length, `Generated ${chunks.length} chunks from ${docs.length} documents`);
  const chunksValid = chunks.every(c => c.chunkId && c.documentId && c.text && c.text.length > 20 && c.jobRole);
  assert(chunksValid, 'All chunks have chunkId, documentId, text, and preserved metadata');

  // 3. Embedding Generation & Vector Dimension Tests
  console.log('\n--- 3. Embedding Generation ---');
  const testEmbedding = await generateEmbedding('React and TypeScript frontend engineering');
  assert(Array.isArray(testEmbedding), 'Embedding is an array of numbers');
  assert(testEmbedding.length === EXPECTED_EMBEDDING_DIM, `Embedding dimensions match ${EXPECTED_EMBEDDING_DIM} (got ${testEmbedding.length})`);
  const hasValidNumbers = testEmbedding.every(n => typeof n === 'number' && !isNaN(n));
  assert(hasValidNumbers, 'Embedding contains valid non-NaN float values');

  // 4. Vector Store Tests
  console.log('\n--- 4. Vector Store & Storage Persistence ---');
  const vectorStore = await getVectorStore();
  assert(vectorStore.length > 0, `Vector store loaded ${vectorStore.length} records`);
  const recordsValid = vectorStore.every(r => isValidVectorRecord(r));
  assert(recordsValid, 'Every vector record in storage has valid chunkId, text, and embedding vector');

  // 5. Cosine Similarity Tests
  console.log('\n--- 5. Cosine Similarity Math ---');
  const vec1 = [1, 0, 0];
  const vec2 = [1, 0, 0];
  const vec3 = [0, 1, 0];
  const vec4 = [-1, 0, 0];
  assert(Math.abs(cosineSimilarity(vec1, vec2) - 1.0) < 0.0001, 'Identical vectors produce cosine similarity ~ 1.0');
  assert(Math.abs(cosineSimilarity(vec1, vec3) - 0.0) < 0.0001, 'Orthogonal vectors produce cosine similarity ~ 0.0');
  assert(Math.abs(cosineSimilarity(vec1, vec4) - (-1.0)) < 0.0001, 'Opposite vectors produce cosine similarity ~ -1.0');

  // 6. Retrieval Tests for Multiple Job Roles
  console.log('\n--- 6. Retrieval & Filtering by Job Role ---');

  // Test Role A: Frontend Developer
  const feResume = `
  Jane Doe - Frontend Engineer
  Skills: React, JavaScript, HTML, CSS.
  Experience: Built user interfaces for e-commerce website.
  `;
  const feChunks = await retrieveRelevantKnowledge('Frontend Developer', feResume, { topK: 4 });
  assert(feChunks.length > 0, `Retrieved ${feChunks.length} chunks for Frontend Developer`);
  const feHasFrontendChunk = feChunks.some(c => c.jobRole === 'Frontend Developer');
  assert(feHasFrontendChunk, 'Frontend retrieval returned chunks specific to Frontend Developer');
  const feFormatted = formatRetrievedContext(feChunks);
  assert(feFormatted.includes('Frontend Developer'), 'Formatted context contains Frontend Developer benchmark info');

  // Test Role B: Data Analyst
  const daResume = `
  John Smith - Data Analyst
  Skills: SQL, Excel, basic Python.
  Experience: Extracted reports from relational database and created monthly pivot tables.
  `;
  const daChunks = await retrieveRelevantKnowledge('Data Analyst', daResume, { topK: 4 });
  assert(daChunks.length > 0, `Retrieved ${daChunks.length} chunks for Data Analyst`);
  const daHasDataAnalystChunk = daChunks.some(c => c.jobRole === 'Data Analyst');
  assert(daHasDataAnalystChunk, 'Data Analyst retrieval returned chunks specific to Data Analyst');

  // Test Role C: Backend Developer
  const beChunks = await retrieveRelevantKnowledge('Backend Developer', 'Node.js, PostgreSQL, API design, Docker');
  assert(beChunks.length > 0, `Retrieved ${beChunks.length} chunks for Backend Developer`);
  assert(beChunks.some(c => c.jobRole === 'Backend Developer'), 'Backend retrieval prioritized Backend Developer knowledge');

  // 7. Document Text Extractor Tests
  console.log('\n--- 7. Document Text Extractor ---');

  // Test Plain Text File
  const txtFormData = new FormData();
  const txtFile = new File(['John Doe\nSoftware Engineer\nSkills: Java, Python, SQL, Git'], 'resume.txt', { type: 'text/plain' });
  txtFormData.append('file', txtFile);
  const txtResult = await extractResumeText(txtFormData);
  assert(txtResult.success === true, 'TXT extraction succeeded');
  assert(txtResult.text?.includes('Software Engineer') === true, 'TXT extracted correct textual content');

  // Test Empty File
  const emptyFormData = new FormData();
  const emptyFile = new File([''], 'empty.txt', { type: 'text/plain' });
  emptyFormData.append('file', emptyFile);
  const emptyResult = await extractResumeText(emptyFormData);
  assert(emptyResult.success === false, 'Empty file correctly rejected');
  assert(!!emptyResult.error, `Empty file returned meaningful error: "${emptyResult.error}"`);

  // Test Unsupported File Type
  const badFormData = new FormData();
  const badFile = new File(['corrupt binary data'], 'image.png', { type: 'image/png' });
  badFormData.append('file', badFile);
  const badResult = await extractResumeText(badFormData);
  assert(badResult.success === false, 'Unsupported file type correctly rejected');
  assert(badResult.error?.includes('Unsupported file format') === true, 'Unsupported file returned clear format error');

  // Summary
  console.log('\n=====================================================');
  console.log(`Test Execution Finished: ${passed} PASSED, ${failed} FAILED`);
  console.log('=====================================================');

  if (failed > 0) {
    process.exit(1);
  }
}

runTests().catch(err => {
  console.error('Test runner fatal error:', err);
  process.exit(1);
});

