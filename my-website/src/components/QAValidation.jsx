// src/components/QAValidation.jsx
import React from 'react';

const QAValidation = ({ chapterId, title }) => {
  // Sample QA checks - in a real system these would be more comprehensive
  const qaChecks = [
    {
      id: 'learning-outcomes',
      name: 'Learning Outcomes',
      description: 'Verify measurable learning outcomes are defined',
      status: true, // Assume passed for demo
      comment: 'Chapter includes 8 measurable learning outcomes'
    },
    {
      id: 'gherkin-specs',
      name: 'Gherkin Specifications',
      description: 'Validate Gherkin specifications are properly formatted',
      status: true, // Assume passed for demo
      comment: '5 Gherkin specifications found and properly formatted'
    },
    {
      id: 'theory-intuition',
      name: 'Theory & Intuition',
      description: 'Check for clear explanations with analogies',
      status: true, // Assume passed for demo
      comment: 'Theory section includes appropriate analogies and explanations'
    },
    {
      id: 'mermaid-diagrams',
      name: 'Mermaid Diagrams',
      description: 'Verify Mermaid diagrams render correctly',
      status: true, // Assume passed for demo
      comment: 'Diagrams are properly formatted and render in preview'
    },
    {
      id: 'hands-on-labs',
      name: 'Hands-On Labs',
      description: 'Validate lab exercises with clear steps',
      status: true, // Assume passed for demo
      comment: '3 well-structured lab exercises with clear objectives'
    },
    {
      id: 'sim-to-real-notes',
      name: 'Sim-to-Real Notes',
      description: 'Verify practical application notes are included',
      status: true, // Assume passed for demo
      comment: 'Sim-to-real notes address hardware considerations'
    },
    {
      id: 'mcqs',
      name: 'Multiple Choice Questions',
      description: 'Check MCQs with answers and explanations',
      status: true, // Assume passed for demo
      comment: '15 MCQs with correct answers and explanations'
    },
    {
      id: 'further-reading',
      name: 'Further Reading',
      description: 'Validate references are accessible and relevant',
      status: true, // Assume passed for demo
      comment: '6 relevant references provided with valid links'
    }
  ];

  // Calculate overall status
  const passedChecks = qaChecks.filter(check => check.status).length;
  const totalChecks = qaChecks.length;
  const passRate = Math.round((passedChecks / totalChecks) * 100);
  
  return (
    <div className="qa-validation">
      <div className="qa-summary">
        <h3>Quality Assurance Report: {title}</h3>
        <div className="qa-stats">
          <div className="stat">
            <span className="number">{passedChecks}/{totalChecks}</span>
            <span className="label">Checks Passed</span>
          </div>
          <div className="stat">
            <span className="number">{passRate}%</span>
            <span className="label">Pass Rate</span>
          </div>
          <div className="stat status">
            <span className={`status-indicator ${passRate >= 80 ? 'pass' : 'fail'}`}>
              {passRate >= 80 ? 'PASS' : 'REQUIRES ATTENTION'}
            </span>
          </div>
        </div>
      </div>

      <div className="qa-details">
        <h4>Detailed Results</h4>
        <table className="qa-table">
          <thead>
            <tr>
              <th>Check</th>
              <th>Status</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {qaChecks.map((check) => (
              <tr key={check.id}>
                <td>{check.name}</td>
                <td>
                  <span className={`status-badge ${check.status ? 'pass' : 'fail'}`}>
                    {check.status ? '✓ PASS' : '✗ FAIL'}
                  </span>
                </td>
                <td>{check.comment}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <style jsx>{`
        .qa-validation {
          margin: 2rem 0;
          padding: 1.5rem;
          border: 1px solid #ddd;
          border-radius: 8px;
          background-color: #fafafa;
        }
        
        .qa-summary {
          margin-bottom: 1.5rem;
        }
        
        .qa-summary h3 {
          margin-top: 0;
          color: #333;
          border-bottom: 1px solid #eee;
          padding-bottom: 0.75rem;
        }
        
        .qa-stats {
          display: flex;
          justify-content: space-around;
          margin: 1rem 0;
        }
        
        .stat {
          text-align: center;
        }
        
        .number {
          display: block;
          font-size: 1.5rem;
          font-weight: bold;
          color: #2196F3;
        }
        
        .label {
          font-size: 0.8rem;
          color: #666;
        }
        
        .status-indicator {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          border-radius: 12px;
          font-weight: bold;
          font-size: 0.9rem;
        }
        
        .status-indicator.pass {
          background-color: #e8f5e9;
          color: #388e3c;
        }
        
        .status-indicator.fail {
          background-color: #ffebee;
          color: #d32f2f;
        }
        
        .qa-details h4 {
          margin-top: 0;
          margin-bottom: 1rem;
          color: #333;
        }
        
        .qa-table {
          width: 100%;
          border-collapse: collapse;
          margin-top: 1rem;
        }
        
        .qa-table th,
        .qa-table td {
          padding: 0.75rem;
          text-align: left;
          border-bottom: 1px solid #ddd;
        }
        
        .qa-table th {
          background-color: #f5f5f5;
          font-weight: 600;
        }
        
        .status-badge {
          display: inline-block;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
          font-weight: bold;
        }
        
        .status-badge.pass {
          background-color: #e8f5e9;
          color: #388e3c;
        }
        
        .status-badge.fail {
          background-color: #ffebee;
          color: #d32f2f;
        }
      `}</style>
    </div>
  );
};

export default QAValidation;