import React, { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import InputBar from './InputBar';
import ExplanationModal from './ExplanationModal';

const ChatWindow = () => {
    const [messages, setMessages] = useState([]);
    const [isExplanationOpen, setIsExplanationOpen] = useState(false);
    const [explanationData, setExplanationData] = useState(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSendMessage = (text) => {
        if (text.trim() === '') return;
        
        // Add user's message to the chat
        const newUserMessage = {
            id: messages.length + 1,
            text,
            sender: 'user',
            timestamp: new Date().toISOString()
        };
        setMessages(prevMessages => [...prevMessages, newUserMessage]);
        
        // TODO: Call API to get AI response
        // const aiResponse = await getAiResponseFromApi(text);
        
        // This is a placeholder for the AI's response
        const aiResponse = {
            id: messages.length + 2,
            text: "This is a response from the Zenith AI.",
            sender: 'ai',
            timestamp: new Date().toISOString(),
            explanation: {
                narrative: "I chose this response because it is conceptually aligned with your question.",
                counterfactual: "An alternative response would have been rejected because it lacked a clear causal link."
            }
        };
        
        setMessages(prevMessages => [...prevMessages, aiResponse]);
    };
    
    const handleExplain = (explanation) => {
        setExplanationData(explanation);
        setIsExplanationOpen(true);
    };

    return (
        <div className="chat-window">
            <div className="messages-container">
                {messages.map(message => (
                    <MessageBubble 
                        key={message.id} 
                        message={message} 
                        onExplain={message.sender === 'ai' ? () => handleExplain(message.explanation) : null}
                    />
                ))}
                <div ref={messagesEndRef} />
            </div>
            <InputBar onSendMessage={handleSendMessage} />
            {isExplanationOpen && (
                <ExplanationModal
                    onClose={() => setIsExplanationOpen(false)}
                    data={explanationData}
                />
            )}
        </div>
    );
};

export default ChatWindow;

