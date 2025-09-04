import React from 'react';
import ChatWindow from '../components/ChatWindow';
import './HomePage.css';

const HomePage = () => {
    return (
        <div className="home-page-container">
            <header className="home-page-header">
                <h1>Zenith AI Chat</h1>
                <p>A multi-modal, explainable AI assistant.</p>
            </header>
            <main className="home-page-content">
                <ChatWindow />
            </main>
        </div>
    );
};

export default HomePage;

