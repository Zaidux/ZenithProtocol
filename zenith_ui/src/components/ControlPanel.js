import React, { useState } from 'react';
import './ControlPanel.css';

const ControlPanel = ({ onToggleCorrection, onStartAutonomousLearning, onAdjustExploration }) => {
    const [isCorrectionEnabled, setIsCorrectionEnabled] = useState(false);
    const [explorationWeight, setExplorationWeight] = useState(5.0);

    const handleToggleCorrection = () => {
        const newState = !isCorrectionEnabled;
        setIsCorrectionEnabled(newState);
        onToggleCorrection(newState);
    };

    const handleStartLearning = () => {
        const topic = prompt("Enter a topic for the AI to learn autonomously:");
        if (topic) {
            onStartAutonomousLearning(topic);
        }
    };

    const handleAdjustExploration = (e) => {
        const newWeight = parseFloat(e.target.value);
        setExplorationWeight(newWeight);
        onAdjustExploration(newWeight);
    };

    return (
        <div className="control-panel">
            <h3 className="panel-title">Zenith Control Panel</h3>
            <div className="control-group">
                <label className="control-label">Self-Correction Loop:</label>
                <button 
                    onClick={handleToggleCorrection}
                    className={`control-button ${isCorrectionEnabled ? 'button-on' : 'button-off'}`}
                >
                    {isCorrectionEnabled ? 'Enabled' : 'Disabled'}
                </button>
            </div>
            <div className="control-group">
                <label className="control-label">Autonomous Learning:</label>
                <button 
                    onClick={handleStartLearning}
                    className="control-button button-action"
                >
                    Start Learning
                </button>
            </div>
            <div className="control-group">
                <label className="control-label">Exploration Weight: {explorationWeight.toFixed(1)}</label>
                <input 
                    type="range"
                    min="0"
                    max="10"
                    step="0.1"
                    value={explorationWeight}
                    onChange={handleAdjustExploration}
                    className="control-slider"
                />
            </div>
        </div>
    );
};

export default ControlPanel;

