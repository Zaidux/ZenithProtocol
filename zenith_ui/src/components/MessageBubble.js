import React from 'react';
import './MessageBubble.css';
import ConfidenceScore from './ConfidenceScore';

const MessageBubble = ({ message, onExplain }) => {
    const isUser = message.sender === 'user';
    const hasExplanation = message.explanation && onExplain;
    
    return (
        <div className={`message-bubble-container ${isUser ? 'user' : 'ai'}`}>
            <div className="message-content">
                <p>{message.text}</p>
                {/* Render confidence score for AI messages */}
                {!isUser && message.confidenceScore && (
                    <ConfidenceScore score={message.confidenceScore} />
                )}
            </div>
            {/* Render "Explain" button for AI messages if an explanation is available */}
            {hasExplanation && (
                <button 
                    className="explain-button" 
                    onClick={() => onExplain(message.explanation)}
                >
                    Explain
                </button>
            )}
        </div>
    );
};

export default MessageBubble;

