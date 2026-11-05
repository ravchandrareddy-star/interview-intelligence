'use server';

/**
 * Server-side document text extraction for PDF, DOCX, and TXT/MD resumes.
 */

const MAX_FILE_SIZE = 2 * 1024 * 1024; // 2MB

const ALLOWED_MIME_TYPES = [
  'text/plain',
  'text/markdown',
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
];

const ALLOWED_EXTENSIONS = ['.txt', '.md', '.pdf', '.docx'];

/**
 * Extracts text from a PDF Buffer.
 * Uses unpdf as primary parser (worker-free, Turbopack/Next.js compatible)
 * with pdf-parse as a fallback.
 */
async function extractFromPdf(buffer: Buffer): Promise<string> {
  // 1. Primary parser: unpdf
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { extractText } = require('unpdf');
    if (typeof extractText === 'function') {
      const result = await extractText(new Uint8Array(buffer));
      const pages = result?.text;
      const combined = Array.isArray(pages) ? pages.join('\n\n') : (typeof pages === 'string' ? pages : '');
      if (combined && combined.trim()) {
        return combined.trim();
      }
    }
  } catch (unpdfErr) {
    console.warn('[Document Extractor] unpdf parser attempt failed:', unpdfErr);
  }

  // 2. Secondary fallback parser: pdf-parse
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const pdfModule = require('pdf-parse');
    if (typeof pdfModule === 'function') {
      const data = await pdfModule(buffer);
      if (data?.text && data.text.trim()) {
        return data.text.trim();
      }
    } else if (pdfModule.PDFParse) {
      const parser = new pdfModule.PDFParse({ data: buffer });
      try {
        const result = await parser.getText();
        const text = typeof result === 'string' ? result : result?.text;
        if (text && text.trim()) {
          return text.trim();
        }
      } finally {
        if (typeof parser.destroy === 'function') {
          await parser.destroy();
        }
      }
    }
  } catch (pdfParseErr) {
    console.warn('[Document Extractor] pdf-parse fallback attempt failed:', pdfParseErr);
  }

  throw new Error('Failed to read PDF document. The file may be password-protected, image-only (scanned), or corrupted.');
}

/**
 * Extracts text from a DOCX Buffer using mammoth.
 */
async function extractFromDocx(buffer: Buffer): Promise<string> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mammoth = require('mammoth');
    const result = await mammoth.extractRawText({ buffer });
    return result?.value || '';
  } catch (err) {
    console.error('[Document Extractor] DOCX extraction error:', err);
    throw new Error('Failed to read DOCX document. The file may be damaged or invalid.');
  }
}

export interface ExtractResult {
  success: boolean;
  text?: string;
  error?: string;
}

/**
 * Server action to extract text from an uploaded resume file.
 */
export async function extractResumeText(formData: FormData): Promise<ExtractResult> {
  try {
    const file = formData.get('file') as File | null;

    if (!file) {
      return { success: false, error: 'No file was provided for extraction.' };
    }

    if (file.size === 0) {
      return { success: false, error: 'The uploaded file is empty (0 bytes).' };
    }

    if (file.size > MAX_FILE_SIZE) {
      return { success: false, error: 'File size exceeds the 2MB limit.' };
    }

    const fileName = file.name || 'resume';
    const fileExt = fileName.includes('.') ? `.${fileName.split('.').pop()?.toLowerCase()}` : '';
    const fileType = file.type || '';

    const isAllowedExt = ALLOWED_EXTENSIONS.includes(fileExt);
    const isAllowedMime = ALLOWED_MIME_TYPES.includes(fileType);

    if (!isAllowedExt && !isAllowedMime) {
      return {
        success: false,
        error: `Unsupported file format. Please upload a .pdf, .docx, .txt, or .md file.`,
      };
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    let extractedText = '';

    if (fileExt === '.pdf' || fileType === 'application/pdf') {
      extractedText = await extractFromPdf(buffer);
    } else if (
      fileExt === '.docx' ||
      fileType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ) {
      extractedText = await extractFromDocx(buffer);
    } else {
      // Plain text or Markdown
      extractedText = buffer.toString('utf-8');
    }

    const cleanText = extractedText.trim();
    if (!cleanText || cleanText.length < 20) {
      return {
        success: false,
        error: 'The uploaded file does not contain readable text. If this is a scanned PDF, please use a text-based PDF.',
      };
    }

    return {
      success: true,
      text: cleanText,
    };
  } catch (error) {
    console.error('[Document Extractor] Error processing file:', error);
    return {
      success: false,
      error: (error as Error)?.message || 'An unexpected error occurred while parsing the document.',
    };
  }
}
