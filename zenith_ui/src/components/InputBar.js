
import React, { useState, useRef } from 'react';
import './InputBar.css';

const InputBar = ({ onSendMessage }) => {
    const [input, setInput] = useState('');
    const fileInputRef = useRef(null);

    const handleSendClick = () => {
        if (input.trim() !== '') {
            onSendMessage(input);
            setInput(''); // Clear the input field
        }
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            // In a real app, you would handle the file upload here
            onSendMessage(`File uploaded: ${file.name}`);
        }
    };

    const handleVoiceClick = () => {
        // TODO: Implement voice-to-text functionality
        alert('Voice command functionality coming soon!');
    };
    
    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            handleSendClick();
        }
    };

    return (
        <div className="input-bar-container">
            <input
                type="text"
                className="text-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type a message or command..."
            />
            <div className="input-actions">
                <button 
                    className="action-button file-button" 
                    onClick={() => fileInputRef.current.click()}
                    title="Attach file"
                >
                    📎
                </button>
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                />
                <button 
                    className="action-button voice-button"
                    onClick={handleVoiceClick}
                    title="Use voice command"
                >
                    🎙️
                </button>
                <button 
                    className="action-button send-button" 
                    onClick={handleSendClick}
                    title="Send"
                >
                    ➤
                </button>
            </div>
        </div>
    );
};

export default InputBar;
