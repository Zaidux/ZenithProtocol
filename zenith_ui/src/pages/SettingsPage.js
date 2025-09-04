import React, { useState } from 'react';
import './SettingsPage.css';

const SettingsPage = () => {
    const [tone, setTone] = useState('neutral');
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [explorationWeight, setExplorationWeight] = useState(5.0);

    const handleSaveSettings = () => {
        const settings = {
            tone,
            isDarkMode,
            explorationWeight
        };
        console.log('Saving settings:', settings);
        // TODO: Call API to save settings to the backend
    };

    return (
        <div className="settings-page-container">
            <header className="settings-header">
                <h1>User Settings</h1>
                <p>Customize the Zenith AI to your preferences.</p>
            </header>
            <main className="settings-content">
                <div className="setting-group">
                    <label>AI Tone:</label>
                    <select value={tone} onChange={(e) => setTone(e.target.value)}>
                        <option value="friendly">Friendly</option>
                        <option value="neutral">Neutral</option>
                        <option value="professional">Professional</option>
                    </select>
                </div>
                <div className="setting-group">
                    <label>Dark Mode:</label>
                    <input 
                        type="checkbox"
                        checked={isDarkMode}
                        onChange={(e) => setIsDarkMode(e.target.checked)}
                    />
                </div>
                <div className="setting-group">
                    <label>Exploration Weight: {explorationWeight.toFixed(1)}</label>
                    <input
                        type="range"
                        min="0"
                        max="10"
                        step="0.1"
                        value={explorationWeight}
                        onChange={(e) => setExplorationWeight(parseFloat(e.target.value))}
                    />
                </div>
                <button 
                    className="save-button"
                    onClick={handleSaveSettings}
                >
                    Save Settings
                </button>
            </main>
        </div>
    );
};

export default SettingsPage;

