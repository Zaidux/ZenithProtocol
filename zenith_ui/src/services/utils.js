// src/services/utils.js

/**
 * Formats a given timestamp into a human-readable string.
 * @param {string} timestamp - The ISO 8601 timestamp string.
 * @returns {string} The formatted date and time.
 */
export const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

/**
 * Validates a user's input before sending it to the AI.
 * @param {string} input - The user's input string.
 * @returns {boolean} True if the input is valid, false otherwise.
 */
export const validateInput = (input) => {
    return input && input.trim().length > 0;
};

