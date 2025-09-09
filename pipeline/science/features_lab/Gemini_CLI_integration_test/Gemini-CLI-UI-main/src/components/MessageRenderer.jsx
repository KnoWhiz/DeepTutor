import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CodeBlock = ({ language, value, inline, isDarkMode, className }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(value).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {
      console.error('Failed to copy code');
    });
  };

  if (inline) {
    return (
      <code className="px-1.5 py-0.5 mx-0.5 bg-gray-100 dark:bg-gray-800 text-blue-600 dark:text-blue-400 rounded text-sm font-mono">
        {value}
      </code>
    );
  }

  return (
    <div className="relative group my-2 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
      {language && (
        <div className="flex items-center justify-between px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
            {language}
          </span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded transition-all duration-200"
          >
            {copied ? (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Copied!
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
      )}
      <div className="relative">
        {!language && (
          <button
            onClick={handleCopy}
            className="absolute top-2 right-2 z-10 flex items-center gap-1.5 px-2.5 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200"
          >
            {copied ? (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Copied!
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
        )}
        <SyntaxHighlighter
          language={language || 'text'}
          style={isDarkMode ? oneDark : oneLight}
          customStyle={{
            margin: 0,
            padding: '0.75rem',
            background: isDarkMode ? '#1a1b26' : '#f8f9fa',
            fontSize: '0.8125rem',
            lineHeight: '1.5',
          }}
          showLineNumbers={value.split('\n').length > 5}
          wrapLines={true}
          wrapLongLines={true}
        >
          {value}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};

export const MessageRenderer = ({ content, isDarkMode = true }) => {
  // Filter out "Error: Loaded cached credentials" messages
  let filteredContent = content?.replace(/^Error:\s*Loaded cached credentials\.?\s*\n?/gim, '');
  
  // Process content to ensure code blocks stay together
  // This helps prevent markdown parser from splitting code blocks
  if (filteredContent) {
    // Fix code blocks that might be split
    filteredContent = filteredContent
      // Ensure triple backticks are properly formatted
      .replace(/```([\s\S]*?)```/g, (match, content) => {
        // Remove excessive line breaks within code blocks
        const cleanedContent = content.replace(/\n{3,}/g, '\n\n');
        return '```' + cleanedContent + '```';
      })
      // Normalize line breaks outside code blocks
      .replace(/\n{3,}/g, '\n\n')  // Replace 3+ newlines with 2
      .replace(/\r\n/g, '\n')      // Normalize Windows line endings
      .trim();
  }

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert prose-gray leading-normal">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          code: ({ node, inline, className, children, ...props }) => {
            // For inline code, just render simple styled code
            if (inline) {
              return (
                <code className="px-1.5 py-0.5 mx-0.5 bg-gray-100 dark:bg-gray-800 text-blue-600 dark:text-blue-400 rounded text-sm font-mono">
                  {children}
                </code>
              );
            }
            // Block code is handled by the pre component
            return <>{children}</>;
          },
          pre: ({ node, children, ...props }) => {
            // Extract the code content from pre > code structure
            if (children && children.props) {
              const { className, children: codeContent } = children.props;
              const match = /language-(\w+)/.exec(className || '');
              const language = match ? match[1] : '';
              const value = String(codeContent).replace(/\n$/, '');
              
              return (
                <CodeBlock
                  language={language}
                  value={value}
                  inline={false}
                  isDarkMode={isDarkMode}
                />
              );
            }
            return <div className="my-2">{children}</div>;
          },
          h1: ({ children }) => (
            <h1 className="text-xl font-bold mt-3 mb-2 text-gray-900 dark:text-gray-100">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg font-semibold mt-2.5 mb-1.5 text-gray-900 dark:text-gray-100">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold mt-2 mb-1 text-gray-900 dark:text-gray-100">
              {children}
            </h3>
          ),
          p: ({ children }) => {
            // Check if paragraph only contains whitespace or is empty
            const text = children?.toString().trim();
            if (!text || text === '') return null;
            
            return (
              <p className="mb-2 leading-relaxed text-sm text-gray-700 dark:text-gray-300">
                {children}
              </p>
            );
          },
          ul: ({ children }) => (
            <ul className="list-disc pl-5 mb-2 space-y-0.5 text-sm text-gray-700 dark:text-gray-300">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-5 mb-2 space-y-0.5 text-sm text-gray-700 dark:text-gray-300">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-normal">
              {children}
            </li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-blue-500 dark:border-blue-400 pl-3 py-1.5 my-2 bg-blue-50 dark:bg-blue-900/20 italic text-sm text-gray-700 dark:text-gray-300">
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
            <div className="overflow-x-auto my-4">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
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
            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300">
              {children}
            </td>
          ),
          hr: () => (
            <hr className="my-3 border-gray-200 dark:border-gray-700" />
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
        {filteredContent || ''}
      </ReactMarkdown>
    </div>
  );
};

export default MessageRenderer;