export type JobRole =
  | 'Software Developer'
  | 'Frontend Developer'
  | 'Backend Developer'
  | 'Full Stack Developer'
  | 'Data Analyst'
  | 'General';

export type KnowledgeCategory =
  | 'technical-skills'
  | 'job-requirements'
  | 'industry-expectations'
  | 'ats-guidelines'
  | 'evaluation-criteria'
  | 'summary-writing';

export interface KnowledgeDocument {
  id: string;
  jobRole: JobRole;
  category: KnowledgeCategory;
  source: string;
  title: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface KnowledgeChunk {
  chunkId: string;
  documentId: string;
  jobRole: JobRole;
  category: KnowledgeCategory;
  source: string;
  title: string;
  chunkIndex: number;
  text: string;
}

export interface VectorRecord extends KnowledgeChunk {
  embedding: number[];
}

export interface RetrievedChunk extends KnowledgeChunk {
  similarity: number;
}

