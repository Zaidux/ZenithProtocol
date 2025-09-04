import React, { useState } from 'react';
import './ConfidenceScore.css';

const ConfidenceScore = ({ score, onExplainDetails }) => {
    const [showDetails, setShowDetails] = useState(false);

    const getScoreColor = (value) => {
        if (value > 0.8) return 'score-high';
        if (value > 0.5) return 'score-medium';
        return 'score-low';
    };

    const handleToggleDetails = () => {
        setShowDetails(!showDetails);
        if (onExplainDetails) {
            onExplainDetails(); // This could trigger a more detailed modal
        }
    };

    return (
        <div className="confidence-score-container">
            <div 
                className={`confidence-badge ${getScoreColor(score)}`}
                onClick={handleToggleDetails}
                title="Click for details"
            >
                Confidence: {score.toFixed(2)}
            </div>
            {/* TODO: Implement a modal to show a detailed breakdown of the score */}
            {/* {showDetails && (
                <div className="confidence-details">
                    <p>Score Breakdown:</p>
                    <ul>
                        <li>CKG Reliance: 0.5</li>
                        <li>Counterfactuals: 0.2</li>
                        <li>Blockchain Verified: 0.1</li>
                    </ul>
                </div>
            )} */}
        </div>
    );
};

export default ConfidenceScore;

