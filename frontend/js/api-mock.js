// This file simulates backend API responses for demo purposes
// In a production environment, this would be replaced with actual API calls

// Mock response data
const responses = [
    "I understand that you're feeling anxious. Can you tell me more about what's causing these feelings?",
    "It sounds like you're going through a challenging time. Remember that it's okay to feel this way, and there are techniques we can explore to help manage these emotions.",
    "I'm here to support you. Have you tried any relaxation techniques like deep breathing or mindfulness meditation?",
    "Your feelings are valid. Sometimes anxiety can be our body's way of signaling that we need to pay attention to something in our lives.",
    "It might help to break down what you're experiencing into smaller parts. Could you identify one specific concern that's at the forefront of your mind right now?",
    "I'm glad to hear you're feeling good today! What positive things have happened recently?",
    "That's wonderful to hear. Acknowledging positive emotions is just as important as addressing difficult ones.",
    "Stress can certainly be challenging. Let's explore some stress management techniques that might work for you.",
    "Managing stress effectively often involves a combination of approaches. Would you like to try a quick grounding exercise?",
    "I hear that you're feeling stressed. Sometimes it helps to identify what aspects of a situation are within your control and which aren't."
];

// Mock API endpoint for chat
async function mockChatAPI(message) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Simple keyword matching for demo purposes
    let responseIndex = Math.floor(Math.random() * 5);
    
    if (message.toLowerCase().includes('good') || message.toLowerCase().includes('happy')) {
        responseIndex = 5 + Math.floor(Math.random() * 2);
    } else if (message.toLowerCase().includes('stress')) {
        responseIndex = 7 + Math.floor(Math.random() * 3);
    }
    
    // Calculate mock sentiment
    let sentiment = calculateMockSentiment(message);
    
    return {
        response: responses[responseIndex],
        sentiment: sentiment,
        confidence: 0.85
    };
}

// Mock API endpoint for audio chat
async function mockAudioChatAPI(audioData) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock transcription
    const transcriptions = [
        "I've been feeling really anxious lately.",
        "I'm not sure how to handle all this stress.",
        "I had a good day today, but I'm worried about tomorrow.",
        "Sometimes I feel overwhelmed by everything going on."
    ];
    
    const transcription = transcriptions[Math.floor(Math.random() * transcriptions.length)];
    
    // Calculate mock sentiment
    let sentiment = calculateMockSentiment(transcription);
    
    // Get response
    const responseIndex = Math.floor(Math.random() * responses.length);
    
    return {
        transcription: transcription,
        response: responses[responseIndex],
        sentiment: sentiment,
        confidence: 0.8
    };
}

// Helper function to calculate mock sentiment
function calculateMockSentiment(text) {
    const positiveWords = ['happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'joy', 'positive', 'hopeful'];
    const negativeWords = ['sad', 'bad', 'terrible', 'awful', 'horrible', 'hate', 'depressed', 'anxious', 'worried', 'negative', 'stress', 'overwhelmed'];
    
    const words = text.toLowerCase().split(/\W+/);
    let positiveScore = 0;
    let negativeScore = 0;
    
    words.forEach(word => {
        if (positiveWords.includes(word)) positiveScore++;
        if (negativeWords.includes(word)) negativeScore++;
    });
    
    // Calculate sentiment score between -1 and 1
    const totalScore = positiveScore + negativeScore;
    let sentimentScore = 0;
    
    if (totalScore > 0) {
        sentimentScore = (positiveScore - negativeScore) / totalScore;
    }
    
    return sentimentScore;
}

// Mock API handler (intercepts fetch requests)
const originalFetch = window.fetch;
window.fetch = function(url, options) {
    if (url === '/api/chat' && options.method === 'POST') {
        const data = JSON.parse(options.body);
        return Promise.resolve({
            json: () => mockChatAPI(data.message)
        });
    } else if (url === '/api/audio-chat' && options.method === 'POST') {
        const data = JSON.parse(options.body);
        return Promise.resolve({
            json: () => mockAudioChatAPI(data.audio)
        });
    }
    
    // Pass through to original fetch for all other requests
    return originalFetch(url, options);
};