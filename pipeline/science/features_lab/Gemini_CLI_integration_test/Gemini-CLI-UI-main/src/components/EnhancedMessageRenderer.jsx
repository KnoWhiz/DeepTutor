import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CodeBlock = ({ language, value, isDarkMode }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(value).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {
      console.error('Failed to copy code');
    });
  };

  // Detect language from content if not specified
  let detectedLanguage = language;
  if (!detectedLanguage) {
    // More comprehensive detection based on content patterns
    const lowerValue = value.toLowerCase();
    if (lowerValue.includes('<!doctype html') || lowerValue.includes('<html') || 
        (lowerValue.includes('<script') && lowerValue.includes('</script>'))) {
      detectedLanguage = 'html';
    } else if (lowerValue.includes('<style') || (lowerValue.includes('{') && lowerValue.includes('}'))) {
      // Check for CSS patterns
      if (lowerValue.match(/[.#][\w-]+\s*{/) || lowerValue.match(/^\s*[\w-]+\s*:\s*[\w-]+/m)) {
        detectedLanguage = 'css';
      } else if (value.includes('function ') || value.includes('const ') || value.includes('let ') || value.includes('var ')) {
        detectedLanguage = 'javascript';
      }
    } else if (value.includes('function ') || value.includes('const ') || value.includes('let ') || 
               value.includes('var ') || value.includes('=>') || value.includes('console.')) {
      detectedLanguage = 'javascript';
    } else if (value.includes('def ') || value.includes('import ') || value.includes('from ') || 
               value.includes('class ') || value.includes('print(')) {
      detectedLanguage = 'python';
    } else if (value.includes('<?php')) {
      detectedLanguage = 'php';
    } else if (lowerValue.includes('select ') || lowerValue.includes('from ') || 
               lowerValue.includes('insert ') || lowerValue.includes('update ')) {
      detectedLanguage = 'sql';
    } else if (value.trim().startsWith('{') && value.trim().endsWith('}')) {
      try {
        JSON.parse(value);
        detectedLanguage = 'json';
      } catch (e) {
        // Not valid JSON
      }
    } else if (value.includes('#!/bin/bash') || value.includes('#!/bin/sh')) {
      detectedLanguage = 'bash';
    }
  }

  return (
    <div className="relative group my-3 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 shadow-sm">
      <div className="flex items-center justify-between px-4 py-2 bg-gradient-to-r from-gray-100 to-gray-50 dark:from-gray-800 dark:to-gray-900 border-b border-gray-200 dark:border-gray-700">
        <span className="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wider">
          {detectedLanguage || 'plaintext'}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-3 py-1 text-xs bg-white dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-md transition-all duration-200 shadow-sm"
        >
          {copied ? (
            <>
              <svg className="w-3.5 h-3.5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-green-600 dark:text-green-400">Copied!</span>
            </>
          ) : (
            <>
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
      <div className="relative overflow-x-auto">
        <SyntaxHighlighter
          language={detectedLanguage || 'text'}
          style={isDarkMode ? oneDark : oneLight}
          customStyle={{
            margin: 0,
            padding: '1rem',
            background: isDarkMode ? '#1e1e2e' : '#ffffff',
            fontSize: '0.8125rem',
            lineHeight: '1.6',
            fontFamily: 'JetBrains Mono, Monaco, Consolas, monospace',
          }}
          showLineNumbers={value.split('\n').length > 5}
          wrapLines={true}
          wrapLongLines={true}
          lineNumberStyle={{
            minWidth: '2.5em',
            paddingRight: '1em',
            color: isDarkMode ? '#4a5568' : '#a0aec0',
            fontSize: '0.75rem'
          }}
        >
          {value}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};

export const EnhancedMessageRenderer = ({ content, isDarkMode = true }) => {
  // Filter out error messages
  let processedContent = content?.replace(/^Error:\s*Loaded cached credentials\.?\s*\n?/gim, '');
  
  // Pre-process to ensure code blocks are properly formatted
  if (processedContent) {
    // Handle fenced code blocks to ensure they stay as single blocks
    const codeBlockRegex = /```(\w*)\n?([\s\S]*?)```/g;
    const codeBlocks = [];
    let tempContent = processedContent;
    
    // Extract code blocks and replace with placeholders
    tempContent = tempContent.replace(codeBlockRegex, (match, lang, code) => {
      const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
      codeBlocks.push({ lang: lang || '', code: code.trim() });
      return placeholder;
    });
    
    // Clean up spacing around placeholders
    tempContent = tempContent
      .replace(/\n{3,}/g, '\n\n')
      .replace(/(__CODE_BLOCK_\d+__)\n{2,}/g, '$1\n')
      .replace(/\n{2,}(__CODE_BLOCK_\d+__)/g, '\n$1')
      .trim();
    
    // Restore code blocks with proper formatting
    codeBlocks.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      tempContent = tempContent.replace(
        placeholder,
        `\n\n\`\`\`${block.lang}\n${block.code}\n\`\`\`\n\n`
      );
    });
    
    processedContent = tempContent.trim();
  }

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: ({ node, inline, className, children, ...props }) => {
            if (inline) {
              return (
                <code className="px-1.5 py-0.5 mx-0.5 bg-blue-50 dark:bg-gray-800 text-blue-600 dark:text-blue-400 rounded text-xs font-mono">
                  {children}
                </code>
              );
            }
            
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            const value = String(children).replace(/\n$/, '');
            
            return (
              <CodeBlock
                language={language}
                value={value}
                isDarkMode={isDarkMode}
              />
            );
          },
          pre: ({ children }) => {
            // Pass through - code blocks are handled by code component
            return <>{children}</>;
          },
          h1: ({ children }) => (
            <h1 className="text-xl font-bold mt-4 mb-2 text-gray-900 dark:text-gray-100">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg font-semibold mt-3 mb-1.5 text-gray-900 dark:text-gray-100">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold mt-2 mb-1 text-gray-900 dark:text-gray-100">
              {children}
            </h3>
          ),
          p: ({ children }) => {
            const text = children?.toString().trim();
            if (!text || text === '') return null;
            
            return (
              <p className="mb-2 text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                {children}
              </p>
            );
          },
          ul: ({ children }) => (
            <ul className="list-disc pl-5 mb-2 space-y-1 text-sm text-gray-700 dark:text-gray-300">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-5 mb-2 space-y-1 text-sm text-gray-700 dark:text-gray-300">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed">
              {children}
            </li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-blue-500 dark:border-blue-400 pl-4 py-2 my-3 bg-blue-50 dark:bg-blue-900/20 italic text-sm text-gray-700 dark:text-gray-300">
              {children}
            </blockquote>
          ),
          a: ({ href, children }) => (
            <a 
              href={href} 
              className="text-blue-600 dark:text-blue-400 hover:underline hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
              target="_blank" 
              rel="noopener noreferrer"
            >
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-3">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 border border-gray-200 dark:border-gray-700">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-gray-50 dark:bg-gray-800">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              {children}
            </tbody>
          ),
          th: ({ children }) => (
            <th className="px-4 py-2 text-left text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wider">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300">
              {children}
            </td>
          ),
          hr: () => (
            <hr className="my-4 border-gray-300 dark:border-gray-600" />
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-gray-900 dark:text-gray-100">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-gray-700 dark:text-gray-300">
              {children}
            </em>
          ),
        }}
      >
        {processedContent || ''}
      </ReactMarkdown>
    </div>
  );
};

export default EnhancedMessageRenderer;