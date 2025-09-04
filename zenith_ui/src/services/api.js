// src/services/api.js

const API_BASE_URL = 'http://localhost:5000/api'; // The address of your Python backend

// Function to send a text prompt to the AI
export const sendPromptToAI = async (prompt) => {
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt }),
        });
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error sending prompt to AI:', error);
        return { error: 'Failed to get a response from the AI.' };
    }
};

// Function to get an explanation for a specific response
export const getExplanation = async (messageId) => {
    try {
        const response = await fetch(`${API_BASE_URL}/explain/${messageId}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching explanation:', error);
        return { error: 'Failed to retrieve explanation.' };
    }
};

// Function to get a verifiable record from the blockchain
export const getVerifiableRecord = async (recordId) => {
    try {
        const response = await fetch(`${API_BASE_URL}/verify/${recordId}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching verifiable record:', error);
        return { error: 'Failed to retrieve verifiable record.' };
    }
};

