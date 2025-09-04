// src/services/websocket.js

// This is a placeholder. In a real application, you would use a library
// like 'socket.io-client' or a custom hook to manage the WebSocket connection.

const WEBSOCKET_URL = 'ws://localhost:5000/ws';
let websocket = null;

export const connectWebSocket = (onMessageReceived) => {
    websocket = new WebSocket(WEBSOCKET_URL);

    websocket.onopen = () => {
        console.log('WebSocket connected.');
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        onMessageReceived(data);
    };

    websocket.onclose = () => {
        console.log('WebSocket disconnected.');
        // TODO: Implement reconnection logic
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
};

export const sendMessage = (message) => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(message));
    } else {
        console.error('WebSocket is not connected.');
    }
};

