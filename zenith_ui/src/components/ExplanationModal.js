import React from 'react';
import './ExplanationModal.css';

const ExplanationModal = ({ onClose, data }) => {
    if (!data) return null;

    const { narrative, conceptual_reasoning, counterfactual_reasoning } = data;
    const { confidence_score, decision_narrative } = data;

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <div className="modal-header">
                    <h3 className="modal-title">AI Reasoning Breakdown</h3>
                    <button onClick={onClose} className="modal-close-button">&times;</button>
                </div>
                <div className="modal-body">
                    <p className="modal-narrative">{narrative}</p>
                    <div className="modal-section">
                        <h4>Decision Analysis</h4>
                        <p>{decision_narrative}</p>
                        {counterfactual_reasoning && (
                            <div className="counterfactuals">
                                <h5>Counterfactuals:</h5>
                                <p>{counterfactual_reasoning}</p>
                            </div>
                        )}
                    </div>
                    <div className="modal-section">
                        <h4>Conceptual Reasoning</h4>
                        <p>{conceptual_reasoning}</p>
                    </div>
                    <div className="modal-footer">
                        <p className="confidence-text">
                            Confidence Score: <span className="confidence-score-value">{confidence_score.toFixed(2)}</span>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ExplanationModal;

