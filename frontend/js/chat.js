// // Initialize required variables
// let chatHistory = [];
// const API_KEY = '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'; // You'll need to set your Groq API key here
// let speechSynthesis = window.speechSynthesis;
// let selectedVoice = null;
// let hindiVoice = null;console.log("Selected English voice:", englishVoice ? englishVoice.name : "None found");
// console.log("Selected Hindi voice:", hindiVoice ? hindiVoice.name : "None found");
// console.log("Selected voice for Serene:", selectedVoice ? selectedVoice.name : "None found");
// let englishVoice = null;

// document.addEventListener('DOMContentLoaded', function() {
//     initChatInterface();
//     initTextToSpeech();
// });

// function initTextToSpeech() {
//     // Initialize speech synthesis
//     if ('speechSynthesis' in window) {
//         // Try loading voices immediately
//         let voices = speechSynthesis.getVoices();
        
//         // If voices aren't loaded yet, wait for the event
//         if (voices.length === 0) {
//             speechSynthesis.onvoiceschanged = setupVoices;
//         } else {
//             setupVoices();
//         }
//     } else {
//         console.warn("Text-to-speech not supported in this browser");
//     }
// }

// function setupVoices() {
//     const voices = speechSynthesis.getVoices();
    
//     // Log all available voices to console for debugging
//     console.log("All available voices:", voices.map(v => `${v.name} (${v.lang})`));
    
//     // Find Hindi male voice
//     hindiVoice = voices.find(voice => 
//         voice.lang.includes('hi') && 
//         (voice.name.toLowerCase().includes('male') || 
//         voice.name.toLowerCase().includes('purush'))
//     );
    
//     // If no Hindi male voice, try any Hindi voice
//     if (!hindiVoice) {
//         hindiVoice = voices.find(voice => voice.lang.includes('hi'));
//     }
    
//     // List of common male voice IDs across different browsers/platforms
//     const commonMaleVoiceNames = [
//         'Microsoft David', 'Google UK English Male', 'Daniel', 
//         'Microsoft Mark', 'Microsoft James', 'Google US English Male',
//         'Alex', 'Fred', 'Aaron', 'Reed', 'Ralph', 'Bruce', 'Samantha', 'Tom',
//         'male'
//     ];
    
//     // Find English male voice
//     englishVoice = voices.find(voice => 
//         voice.lang.includes('en') && 
//         commonMaleVoiceNames.some(name => 
//             voice.name.toLowerCase().includes(name.toLowerCase())
//         )
//     );
    
//     // If no specific male voice is found, use any English voice
//     if (!englishVoice) {
//         englishVoice = voices.find(voice => voice.lang.includes('en'));
//     }
    
//     // Set default voice
//     selectedVoice = englishVoice || voices[0];
    
//     console.log("Selected English voice:", englishVoice ? englishVoice.name : "None found");
//     console.log("Selected Hindi voice:", hindiVoice ? hindiVoice.name : "None found");
    
//     // Create voice selector
//     createVoiceSelector(voices);
// }

// function createVoiceSelector(voices) {
//     // Check if the selector already exists
//     if (document.getElementById('voice-selector')) return;
    
//     // Create a voice selection dropdown
//     const selector = document.createElement('select');
//     selector.id = 'voice-selector';
//     selector.className = 'ml-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white';
    
//     // Add an option for each voice
//     voices.forEach((voice, index) => {
//         const option = document.createElement('option');
//         option.value = index;
//         option.textContent = `${voice.name} (${voice.lang})`;
//         selector.appendChild(option);
//     });
    
//     // Add language selector
//     const langSelector = document.createElement('select');
//     langSelector.id = 'language-selector';
//     langSelector.className = 'ml-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white';
    
//     const autoOption = document.createElement('option');
//     autoOption.value = 'auto';
//     autoOption.textContent = 'Auto-detect';
//     langSelector.appendChild(autoOption);
    
//     const engOption = document.createElement('option');
//     engOption.value = 'english';
//     engOption.textContent = 'English';
//     langSelector.appendChild(engOption);
    
//     const hindiOption = document.createElement('option');
//     hindiOption.value = 'hindi';
//     hindiOption.textContent = 'Hindi';
//     langSelector.appendChild(hindiOption);
    
//     // Handle selection change
//     selector.addEventListener('change', function() {
//         selectedVoice = voices[this.value];
//         console.log("User selected voice:", selectedVoice.name);
//     });
    
//     // Add to interface - find a good place to insert it
//     const inputArea = document.getElementById('message-input');
//     if (inputArea && inputArea.parentNode) {
//         const container = document.createElement('div');
//         container.className = 'flex items-center mt-2';
//         container.innerHTML = '<label class="text-white mr-2">Voice:</label>';
//         container.appendChild(selector);
        
//         const langContainer = document.createElement('div');
//         langContainer.className = 'flex items-center mt-2 ml-4';
//         langContainer.innerHTML = '<label class="text-white mr-2">Language:</label>';
//         langContainer.appendChild(langSelector);
        
//         inputArea.parentNode.parentNode.appendChild(container);
//         inputArea.parentNode.parentNode.appendChild(langContainer);
//     }
// }

// // Function to detect language
// function detectLanguage(text) {
//     // Simple detection based on Unicode character ranges
//     const hindiPattern = /[\u0900-\u097F]/;
//     return hindiPattern.test(text) ? 'hindi' : 'english';
// }

// function speakText(text) {
//     if (!speechSynthesis) return;
    
//     // Stop any ongoing speech
//     speechSynthesis.cancel();
    
//     // Break the text into sentences (include Hindi sentence endings)
//     const sentences = text.split(/(?<=[.!?।])\s+/);
//     let currentIndex = 0;
    
//     // Get language preference
//     const languageSelector = document.getElementById('language-selector');
//     const languagePreference = languageSelector ? languageSelector.value : 'auto';
    
//     // Function to speak the next chunk
//     function speakNextChunk() {
//         if (currentIndex >= sentences.length) return;
        
//         // Create manageable chunks of 1-3 sentences
//         let chunk = sentences[currentIndex];
//         const maxChunkLength = 250; // Maximum characters per utterance
        
//         // If the current sentence is short, combine with next sentences up to maxChunkLength
//         while (currentIndex + 1 < sentences.length && 
//                chunk.length + sentences[currentIndex + 1].length < maxChunkLength) {
//             currentIndex++;
//             chunk += " " + sentences[currentIndex];
//         }
        
//         // Determine language for this chunk
//         let chunkLanguage;
//         if (languagePreference === 'auto') {
//             chunkLanguage = detectLanguage(chunk);
//         } else {
//             chunkLanguage = languagePreference;
//         }
        
//         // Create utterance for this chunk
//         const utterance = new SpeechSynthesisUtterance(chunk);
        
//         // Set voice based on language
//         if (chunkLanguage === 'hindi' && hindiVoice) {
//             utterance.voice = hindiVoice;
//             utterance.lang = 'hi-IN';
//         } else {
//             utterance.voice = englishVoice || selectedVoice;
//             utterance.lang = 'en-US';
//         }
        
//         // Voice characteristics
//         utterance.rate = 1.1;
//         utterance.pitch = 1.0;
        
//         // Set up callback for when this chunk finishes
//         utterance.onend = () => {
//             currentIndex++;
//             speakNextChunk();
//         };
        
//         // Handle errors and cancellations
//         utterance.onerror = (event) => {
//             console.error("Speech synthesis error:", event);
//             currentIndex = sentences.length; // Stop on error
//         };
        
//         // Start speaking this chunk
//         speechSynthesis.speak(utterance);
//     }
    
//     // Start the first chunk
//     speakNextChunk();
// }

// function initChatInterface() {
//     const chatForm = document.getElementById('chat-form');
//     const messageInput = document.getElementById('message-input');
//     const chatMessages = document.getElementById('chat-messages');
//     const speakToggle = document.getElementById('speak-toggle');
//     let speakResponses = true;
    
//     // Initialize speak toggle
//     if (speakToggle) {
//         speakToggle.addEventListener('change', function() {
//             speakResponses = this.checked;
//         });
//     }
    
//     // Handle form submission
//     if (chatForm) {
//         chatForm.addEventListener('submit', function(e) {
//             e.preventDefault();
            
//             const message = messageInput.value.trim();
//             if (message) {
//                 sendMessage(message);
//                 messageInput.value = '';
//             }
//         });
//     }

//     // Handle quick reply buttons
//     document.querySelectorAll('.quick-reply').forEach(button => {
//         button.addEventListener('click', function() {
//             const message = this.textContent.trim();
//             sendMessage(message);
//         });
//     });

//     // Function to send message
//     function sendMessage(message) {
//         // Add user message to chat
//         addMessageToChat('user', message);
        
//         // Show typing indicator
//         showTypingIndicator();
        
//         // Update chat history
//         chatHistory.push({
//             role: 'user',
//             content: message
//         });
        
//         // Send to Groq API
//         callGroqAPI(message)
//             .then(response => {
//                 // Hide typing indicator
//                 hideTypingIndicator();
                
//                 // Add bot response to chat
//                 addMessageToChat('bot', response);
                
//                 // Update chat history
//                 chatHistory.push({
//                     role: 'assistant',
//                     content: response
//                 });
                
//                 // Speak the response if enabled
//                 if (speakResponses) {
//                     // Extract text without markdown formatting for speech
//                     const plainText = response.replace(/```[\s\S]*?```/g, 'code block omitted')
//                                             .replace(/`([^`]+)`/g, '$1')
//                                             .replace(/\*\*([^*]+)\*\*/g, '$1')
//                                             .replace(/\*([^*]+)\*/g, '$1');
//                     speakText(plainText);
//                 }
//             })
//             .catch(error => {
//                 console.error('Error:', error);
//                 hideTypingIndicator();
                
//                 // Show error message
//                 const errorMessage = "I'm sorry, I'm having trouble connecting to Groq right now. Please check your API key and try again.";
//                 addMessageToChat('bot', errorMessage);
                
//                 // Speak the error message if enabled
//                 if (speakResponses) {
//                     speakText(errorMessage);
//                 }
//             });
//     }

//     // Function to call Groq API
//     async function callGroqAPI(message) {
//         if (!API_KEY) {
//             return "Please set your Groq API key in the chat.js file.";
//         }
        
//         try {
//             const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                     'Authorization': `Bearer ${API_KEY}`
//                 },
//                 body: JSON.stringify({
//                     model: 'llama3-70b-8192', // You can change to other Groq models
//                     messages: chatHistory,
//                     temperature: 0.7,
//                     max_tokens: 1024
//                 })
//             });
            
//             const data = await response.json();
            
//             if (data.error) {
//                 throw new Error(data.error.message);
//             }
            
//             return data.choices[0].message.content;
//         } catch (error) {
//             console.error('Error calling Groq API:', error);
//             throw error;
//         }
//     }

//     // Function to add message to chat
//     function addMessageToChat(sender, message) {
//         const messageElement = document.createElement('div');
//         messageElement.classList.add('flex', 'items-start', 'space-x-4', `${sender}-message`, 'animate-fade-in');
        
//         if (sender === 'user') {
//             messageElement.innerHTML = `
//                 <div class="flex-1 flex justify-end">
//                     <div class="bg-gradient-to-r from-purple-600 to-blue-500 rounded-2xl rounded-tr-none p-4 shadow-lg max-w-md message-bubble">
//                         <p>${formatMessage(message)}</p>
//                     </div>
//                 </div>
//                 <div class="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0 mt-1">
//                     <i class="fas fa-user"></i>
//                 </div>
//             `;
//         } else {
//             // For bot messages, add a speak button
//             const formattedMessage = formatMessage(message);
//             messageElement.innerHTML = `
//                 <div class="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-blue-500 flex items-center justify-center flex-shrink-0 mt-1">
//                     <i class="fas fa-robot"></i>
//                 </div>
//                 <div class="flex-1">
//                     <div class="bg-gray-800/80 backdrop-blur-md rounded-2xl rounded-tl-none p-4 shadow-lg border border-white/10 message-bubble">
//                         <p>${formattedMessage}</p>
//                         <button class="speak-btn text-xs mt-2 text-blue-400 hover:text-blue-300">
//                             <i class="fas fa-volume-up mr-1"></i>Speak
//                         </button>
//                     </div>
//                 </div>
//             `;
            
//             // Add event listener to speak button
//             setTimeout(() => {
//                 const speakBtn = messageElement.querySelector('.speak-btn');
//                 if (speakBtn) {
//                     speakBtn.addEventListener('click', function() {
//                         // Extract text without markdown formatting for speech
//                         const plainText = message.replace(/```[\s\S]*?```/g, 'code block omitted')
//                                             .replace(/`([^`]+)`/g, '$1')
//                                             .replace(/\*\*([^*]+)\*\*/g, '$1')
//                                             .replace(/\*([^*]+)\*/g, '$1');
//                         speakText(plainText);
//                     });
//                 }
//             }, 0);
//         }
        
//         chatMessages.appendChild(messageElement);
        
//         // Scroll to bottom
//         chatMessages.scrollTop = chatMessages.scrollHeight;
//     }

//     // Function to format message with markdown support
//     function formatMessage(message) {
//         // Basic markdown formatting
//         // Convert code blocks
//         message = message.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
        
//         // Convert inline code
//         message = message.replace(/`([^`]+)`/g, '<code>$1</code>');
        
//         // Convert bold text
//         message = message.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
//         // Convert italic text
//         message = message.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
//         // Convert line breaks
//         message = message.replace(/\n/g, '<br>');
        
//         return message;
//     }

//     // Function to show typing indicator
//     function showTypingIndicator() {
//         const typingElement = document.createElement('div');
//         typingElement.id = 'typing-indicator';
//         typingElement.classList.add('flex', 'items-start', 'space-x-4', 'bot-message');
        
//         typingElement.innerHTML = `
//             <div class="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-blue-500 flex items-center justify-center flex-shrink-0 mt-1">
//                 <i class="fas fa-robot"></i>
//             </div>
//             <div class="bg-gray-800/80 backdrop-blur-md rounded-2xl rounded-tl-none p-2 shadow-lg border border-white/10">
//                 <div class="typing-indicator">
//                     <span></span>
//                     <span></span>
//                     <span></span>
//                 </div>
//             </div>
//         `;
        
//         chatMessages.appendChild(typingElement);
        
//         // Scroll to bottom
//         chatMessages.scrollTop = chatMessages.scrollHeight;
//     }

//     // Function to hide typing indicator
//     function hideTypingIndicator() {
//         const typingIndicator = document.getElementById('typing-indicator');
//         if (typingIndicator) {
//             typingIndicator.remove();
//         }
//     }
    
//     // Add a clear chat button function
//     const clearChatButton = document.getElementById('clear-chat');
//     if (clearChatButton) {
//         clearChatButton.addEventListener('click', function() {
//             chatMessages.innerHTML = '';
//             chatHistory = [];
            
//             // Add a system message
//             addMessageToChat('bot', 'Chat history cleared. How can I help you today?');
            
//             // Speak the message if enabled
//             if (speakResponses) {
//                 speakText('Chat history cleared. How can I help you today?');
//             }
//         });
//     }
    
//     // Add a stop speaking button
//     const stopSpeakingButton = document.getElementById('stop-speaking');
//     if (stopSpeakingButton) {
//         stopSpeakingButton.addEventListener('click', function() {
//             if (speechSynthesis) {
//                 speechSynthesis.cancel();
//             }
//         });
//     }
// }




















// // // Initialize required variables
// // let chatHistory = [];
// // const API_KEY = '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'; // You'll need to set your Groq API key here
// // let speechSynthesis = window.speechSynthesis;
// // let selectedVoice = null;
// // let hindiVoice = null;
// // let englishVoice = null;
// // let isMascotSpeaking = false;

// // document.addEventListener('DOMContentLoaded', function() {
// //     initChatInterface();
// //     initTextToSpeech();
// // });

// // function initTextToSpeech() {
// //     // Initialize speech synthesis
// //     if ('speechSynthesis' in window) {
// //         // Try loading voices immediately
// //         let voices = speechSynthesis.getVoices();
        
// //         // If voices aren't loaded yet, wait for the event
// //         if (voices.length === 0) {
// //             speechSynthesis.onvoiceschanged = setupVoices;
// //         } else {
// //             setupVoices();
// //         }
// //     } else {
// //         console.warn("Text-to-speech not supported in this browser");
// //     }
// // }

// // function setupVoices() {
// //     const voices = speechSynthesis.getVoices();
    
// //     // Log all available voices to console for debugging
// //     console.log("All available voices:", voices.map(v => `${v.name} (${v.lang})`));
    
// //     // Find Hindi male voice
// //     hindiVoice = voices.find(voice => 
// //         voice.lang.includes('hi') && 
// //         (voice.name.toLowerCase().includes('male') || 
// //         voice.name.toLowerCase().includes('purush'))
// //     );
    
// //     // If no Hindi male voice, try any Hindi voice
// //     if (!hindiVoice) {
// //         hindiVoice = voices.find(voice => voice.lang.includes('hi'));
// //     }
    
// //     // List of common male voice IDs across different browsers/platforms
// //     const commonMaleVoiceNames = [
// //         'Microsoft David', 'Google UK English Male', 'Daniel', 
// //         'Microsoft Mark', 'Microsoft James', 'Google US English Male',
// //         'Alex', 'Fred', 'Aaron', 'Reed', 'Ralph', 'Bruce', 'Samantha', 'Tom',
// //         'male'
// //     ];
    
// //     // Find English male voice
// //     englishVoice = voices.find(voice => 
// //         voice.lang.includes('en') && 
// //         commonMaleVoiceNames.some(name => 
// //             voice.name.toLowerCase().includes(name.toLowerCase())
// //         )
// //     );
    
// //     // If no specific male voice is found, use any English voice
// //     if (!englishVoice) {
// //         englishVoice = voices.find(voice => voice.lang.includes('en'));
// //     }
    
// //     // Set default voice
// //     selectedVoice = englishVoice || voices[0];
    
// //     console.log("Selected English voice:", englishVoice ? englishVoice.name : "None found");
// //     console.log("Selected Hindi voice:", hindiVoice ? hindiVoice.name : "None found");
    
// //     // Create voice selector
// //     createVoiceSelector(voices);
// // }

// // function createVoiceSelector(voices) {
// //     // Check if the selector already exists
// //     if (document.getElementById('voice-selector')) return;
    
// //     // Create a voice selection dropdown
// //     const selector = document.createElement('select');
// //     selector.id = 'voice-selector';
// //     selector.className = 'ml-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white';
    
// //     // Add an option for each voice
// //     voices.forEach((voice, index) => {
// //         const option = document.createElement('option');
// //         option.value = index;
// //         option.textContent = `${voice.name} (${voice.lang})`;
// //         selector.appendChild(option);
// //     });
    
// //     // Add language selector
// //     const langSelector = document.createElement('select');
// //     langSelector.id = 'language-selector';
// //     langSelector.className = 'ml-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white';
    
// //     const autoOption = document.createElement('option');
// //     autoOption.value = 'auto';
// //     autoOption.textContent = 'Auto-detect';
// //     langSelector.appendChild(autoOption);
    
// //     const engOption = document.createElement('option');
// //     engOption.value = 'english';
// //     engOption.textContent = 'English';
// //     langSelector.appendChild(engOption);
    
// //     const hindiOption = document.createElement('option');
// //     hindiOption.value = 'hindi';
// //     hindiOption.textContent = 'Hindi';
// //     langSelector.appendChild(hindiOption);
    
// //     // Handle selection change
// //     selector.addEventListener('change', function() {
// //         selectedVoice = voices[this.value];
// //         console.log("User selected voice:", selectedVoice.name);
// //     });
    
// //     // Add to interface - find a good place to insert it
// //     const inputArea = document.getElementById('message-input');
// //     if (inputArea && inputArea.parentNode) {
// //         const container = document.createElement('div');
// //         container.className = 'flex items-center mt-2';
// //         container.innerHTML = '<label class="text-white mr-2">Voice:</label>';
// //         container.appendChild(selector);
        
// //         const langContainer = document.createElement('div');
// //         langContainer.className = 'flex items-center mt-2 ml-4';
// //         langContainer.innerHTML = '<label class="text-white mr-2">Language:</label>';
// //         langContainer.appendChild(langSelector);
        
// //         inputArea.parentNode.parentNode.appendChild(container);
// //         inputArea.parentNode.parentNode.appendChild(langContainer);
// //     }
// // }

// // // Function to detect language
// // function detectLanguage(text) {
// //     // Simple detection based on Unicode character ranges
// //     const hindiPattern = /[\u0900-\u097F]/;
// //     return hindiPattern.test(text) ? 'hindi' : 'english';
// // }

// // function animateMascot(isActive) {
// //     const mascots = document.querySelectorAll('.mascot-container');
    
// //     mascots.forEach(mascot => {
// //         if (isActive) {
// //             mascot.classList.add('speaking');
// //         } else {
// //             mascot.classList.remove('speaking');
// //         }
// //     });
    
// //     isMascotSpeaking = isActive;
// // }

// // function speakText(text) {
// //     if (!speechSynthesis) return;
    
// //     // Stop any ongoing speech
// //     speechSynthesis.cancel();
    
// //     // Break the text into sentences (include Hindi sentence endings)
// //     const sentences = text.split(/(?<=[.!?।])\s+/);
// //     let currentIndex = 0;
    
// //     // Get language preference
// //     const languageSelector = document.getElementById('language-selector');
// //     const languagePreference = languageSelector ? languageSelector.value : 'auto';
    
// //     // Activate mascot animation
// //     animateMascot(true);
    
// //     // Function to speak the next chunk
// //     function speakNextChunk() {
// //         if (currentIndex >= sentences.length) {
// //             // Stop mascot animation when done speaking
// //             animateMascot(false);
// //             return;
// //         }
        
// //         // Create manageable chunks of 1-3 sentences
// //         let chunk = sentences[currentIndex];
// //         const maxChunkLength = 250; // Maximum characters per utterance
        
// //         // If the current sentence is short, combine with next sentences up to maxChunkLength
// //         while (currentIndex + 1 < sentences.length && 
// //                chunk.length + sentences[currentIndex + 1].length < maxChunkLength) {
// //             currentIndex++;
// //             chunk += " " + sentences[currentIndex];
// //         }
        
// //         // Determine language for this chunk
// //         let chunkLanguage;
// //         if (languagePreference === 'auto') {
// //             chunkLanguage = detectLanguage(chunk);
// //         } else {
// //             chunkLanguage = languagePreference;
// //         }
        
// //         // Create utterance for this chunk
// //         const utterance = new SpeechSynthesisUtterance(chunk);
        
// //         // Set voice based on language
// //         if (chunkLanguage === 'hindi' && hindiVoice) {
// //             utterance.voice = hindiVoice;
// //             utterance.lang = 'hi-IN';
// //         } else {
// //             utterance.voice = englishVoice || selectedVoice;
// //             utterance.lang = 'en-US';
// //         }
        
// //         // Voice characteristics
// //         utterance.rate = 1.1;
// //         utterance.pitch = 1.0;
        
// //         // Set up callback for when this chunk finishes
// //         utterance.onend = () => {
// //             currentIndex++;
// //             speakNextChunk();
// //         };
        
// //         // Handle errors and cancellations
// //         utterance.onerror = (event) => {
// //             console.error("Speech synthesis error:", event);
// //             currentIndex = sentences.length; // Stop on error
// //             animateMascot(false); // Stop animation on error
// //         };
        
// //         // Start speaking this chunk
// //         speechSynthesis.speak(utterance);
// //     }
    
// //     // Start the first chunk
// //     speakNextChunk();
// // }

// // function initChatInterface() {
// //     const chatForm = document.getElementById('chat-form');
// //     const messageInput = document.getElementById('message-input');
// //     const chatMessages = document.getElementById('chat-messages');
// //     const speakToggle = document.getElementById('speak-toggle');
// //     let speakResponses = true;
    
// //     // Add CSS for mascot animation
// //     addMascotStyles();
    
// //     // Initialize speak toggle
// //     if (speakToggle) {
// //         speakToggle.addEventListener('change', function() {
// //             speakResponses = this.checked;
// //         });
// //     }
    
// //     // Handle form submission
// //     if (chatForm) {
// //         chatForm.addEventListener('submit', function(e) {
// //             e.preventDefault();
            
// //             const message = messageInput.value.trim();
// //             if (message) {
// //                 sendMessage(message);
// //                 messageInput.value = '';
// //             }
// //         });
// //     }

// //     // Handle quick reply buttons
// //     document.querySelectorAll('.quick-reply').forEach(button => {
// //         button.addEventListener('click', function() {
// //             const message = this.textContent.trim();
// //             sendMessage(message);
// //         });
// //     });

// //     // Function to send message
// //     function sendMessage(message) {
// //         // Add user message to chat
// //         addMessageToChat('user', message);
        
// //         // Show typing indicator
// //         showTypingIndicator();
        
// //         // Update chat history
// //         chatHistory.push({
// //             role: 'user',
// //             content: message
// //         });
        
// //         // Send to Groq API
// //         callGroqAPI(message)
// //             .then(response => {
// //                 // Hide typing indicator
// //                 hideTypingIndicator();
                
// //                 // Add bot response to chat
// //                 addMessageToChat('bot', response);
                
// //                 // Update chat history
// //                 chatHistory.push({
// //                     role: 'assistant',
// //                     content: response
// //                 });
                
// //                 // Speak the response if enabled
// //                 if (speakResponses) {
// //                     // Extract text without markdown formatting for speech
// //                     const plainText = response.replace(/```[\s\S]*?```/g, 'code block omitted')
// //                                             .replace(/`([^`]+)`/g, '$1')
// //                                             .replace(/\*\*([^*]+)\*\*/g, '$1')
// //                                             .replace(/\*([^*]+)\*/g, '$1');
// //                     speakText(plainText);
// //                 }
// //             })
// //             .catch(error => {
// //                 console.error('Error:', error);
// //                 hideTypingIndicator();
                
// //                 // Show error message
// //                 const errorMessage = "I'm sorry, I'm having trouble connecting to Groq right now. Please check your API key and try again.";
// //                 addMessageToChat('bot', errorMessage);
                
// //                 // Speak the error message if enabled
// //                 if (speakResponses) {
// //                     speakText(errorMessage);
// //                 }
// //             });
// //     }

// //     // Function to call Groq API
// //     async function callGroqAPI(message) {
// //         if (!API_KEY) {
// //             return "Please set your Groq API key in the chat.js file.";
// //         }
        
// //         try {
// //             const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
// //                 method: 'POST',
// //                 headers: {
// //                     'Content-Type': 'application/json',
// //                     'Authorization': `Bearer ${API_KEY}`
// //                 },
// //                 body: JSON.stringify({
// //                     model: 'llama3-70b-8192', // You can change to other Groq models
// //                     messages: chatHistory,
// //                     temperature: 0.7,
// //                     max_tokens: 1024
// //                 })
// //             });
            
// //             const data = await response.json();
            
// //             if (data.error) {
// //                 throw new Error(data.error.message);
// //             }
            
// //             return data.choices[0].message.content;
// //         } catch (error) {
// //             console.error('Error calling Groq API:', error);
// //             throw error;
// //         }
// //     }

// //     // Function to add message to chat
// //     function addMessageToChat(sender, message) {
// //         const messageElement = document.createElement('div');
// //         messageElement.classList.add('flex', 'items-start', 'space-x-4', `${sender}-message`, 'animate-fade-in');
        
// //         if (sender === 'user') {
// //             messageElement.innerHTML = `
// //                 <div class="flex-1 flex justify-end">
// //                     <div class="bg-gradient-to-r from-purple-600 to-blue-500 rounded-2xl rounded-tr-none p-4 shadow-lg max-w-md message-bubble">
// //                         <p>${formatMessage(message)}</p>
// //                     </div>
// //                 </div>
// //                 <div class="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0 mt-1">
// //                     <i class="fas fa-user"></i>
// //                 </div>
// //             `;
// //         } else {
// //             // For bot messages, add mascot and speak button
// //             const formattedMessage = formatMessage(message);
// //             messageElement.innerHTML = `
// //                 <div class="mascot-container ${isMascotSpeaking ? 'speaking' : ''}">
// //                     <div class="mascot">
// //                         <div class="mascot-head">
// //                             <div class="mascot-face">
// //                                 <div class="mascot-eyes">
// //                                     <div class="mascot-eye"></div>
// //                                     <div class="mascot-eye"></div>
// //                                 </div>
// //                                 <div class="mascot-mouth"></div>
// //                             </div>
// //                         </div>
// //                         <div class="mascot-body"></div>
// //                     </div>
// //                 </div>
// //                 <div class="flex-1">
// //                     <div class="bg-gray-800/80 backdrop-blur-md rounded-2xl rounded-tl-none p-4 shadow-lg border border-white/10 message-bubble">
// //                         <p>${formattedMessage}</p>
// //                         <button class="speak-btn text-xs mt-2 text-blue-400 hover:text-blue-300">
// //                             <i class="fas fa-volume-up mr-1"></i>Speak
// //                         </button>
// //                     </div>
// //                 </div>
// //             `;
            
// //             // Add event listener to speak button
// //             setTimeout(() => {
// //                 const speakBtn = messageElement.querySelector('.speak-btn');
// //                 if (speakBtn) {
// //                     speakBtn.addEventListener('click', function() {
// //                         // Extract text without markdown formatting for speech
// //                         const plainText = message.replace(/```[\s\S]*?```/g, 'code block omitted')
// //                                             .replace(/`([^`]+)`/g, '$1')
// //                                             .replace(/\*\*([^*]+)\*\*/g, '$1')
// //                                             .replace(/\*([^*]+)\*/g, '$1');
// //                         speakText(plainText);
// //                     });
// //                 }
                
// //                 // Add click event to mascot to trigger speech
// //                 const mascot = messageElement.querySelector('.mascot-container');
// //                 if (mascot) {
// //                     mascot.addEventListener('click', function() {
// //                         if (!isMascotSpeaking) {
// //                             const plainText = message.replace(/```[\s\S]*?```/g, 'code block omitted')
// //                                                     .replace(/`([^`]+)`/g, '$1')
// //                                                     .replace(/\*\*([^*]+)\*\*/g, '$1')
// //                                                     .replace(/\*([^*]+)\*/g, '$1');
// //                             speakText(plainText);
// //                         } else {
// //                             // If already speaking, stop
// //                             speechSynthesis.cancel();
// //                             animateMascot(false);
// //                         }
// //                     });
// //                 }
// //             }, 0);
// //         }
        
// //         chatMessages.appendChild(messageElement);
        
// //         // Scroll to bottom
// //         chatMessages.scrollTop = chatMessages.scrollHeight;
// //     }

// //     // Function to format message with markdown support
// //     function formatMessage(message) {
// //         // Basic markdown formatting
// //         // Convert code blocks
// //         message = message.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
        
// //         // Convert inline code
// //         message = message.replace(/`([^`]+)`/g, '<code>$1</code>');
        
// //         // Convert bold text
// //         message = message.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
// //         // Convert italic text
// //         message = message.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
// //         // Convert line breaks
// //         message = message.replace(/\n/g, '<br>');
        
// //         return message;
// //     }

// //     // Function to show typing indicator
// //     function showTypingIndicator() {
// //         const typingElement = document.createElement('div');
// //         typingElement.id = 'typing-indicator';
// //         typingElement.classList.add('flex', 'items-start', 'space-x-4', 'bot-message');
        
// //         typingElement.innerHTML = `
// //             <div class="mascot-container typing">
// //                 <div class="mascot">
// //                     <div class="mascot-head">
// //                         <div class="mascot-face">
// //                             <div class="mascot-eyes">
// //                                 <div class="mascot-eye"></div>
// //                                 <div class="mascot-eye"></div>
// //                             </div>
// //                             <div class="mascot-mouth thinking"></div>
// //                         </div>
// //                     </div>
// //                     <div class="mascot-body"></div>
// //                 </div>
// //             </div>
// //             <div class="bg-gray-800/80 backdrop-blur-md rounded-2xl rounded-tl-none p-2 shadow-lg border border-white/10">
// //                 <div class="typing-indicator">
// //                     <span></span>
// //                     <span></span>
// //                     <span></span>
// //                 </div>
// //             </div>
// //         `;
        
// //         chatMessages.appendChild(typingElement);
        
// //         // Scroll to bottom
// //         chatMessages.scrollTop = chatMessages.scrollHeight;
// //     }

// //     // Function to hide typing indicator
// //     function hideTypingIndicator() {
// //         const typingIndicator = document.getElementById('typing-indicator');
// //         if (typingIndicator) {
// //             typingIndicator.remove();
// //         }
// //     }
    
// //     // Add a clear chat button function
// //     const clearChatButton = document.getElementById('clear-chat');
// //     if (clearChatButton) {
// //         clearChatButton.addEventListener('click', function() {
// //             chatMessages.innerHTML = '';
// //             chatHistory = [];
            
// //             // Add a system message
// //             addMessageToChat('bot', 'Chat history cleared. How can I help you today?');
            
// //             // Speak the message if enabled
// //             if (speakResponses) {
// //                 speakText('Chat history cleared. How can I help you today?');
// //             }
// //         });
// //     }
    
// //     // Add a stop speaking button
// //     const stopSpeakingButton = document.getElementById('stop-speaking');
// //     if (stopSpeakingButton) {
// //         stopSpeakingButton.addEventListener('click', function() {
// //             if (speechSynthesis) {
// //                 speechSynthesis.cancel();
// //                 animateMascot(false);
// //             }
// //         });
// //     }
// // }

// // // Function to add mascot styles
// // function addMascotStyles() {
// //     const styleSheet = document.createElement("style");
// //     styleSheet.textContent = `
// //         .mascot-container {
// //             width: 50px;
// //             height: 50px;
// //             position: relative;
// //             cursor: pointer;
// //             transition: transform 0.3s ease;
// //         }
        
// //         .mascot-container:hover {
// //             transform: scale(1.1);
// //         }
        
// //         .mascot {
// //             width: 100%;
// //             height: 100%;
// //             position: relative;
// //         }
        
// //         .mascot-head {
// //             width: 40px;
// //             height: 40px;
// //             border-radius: 50%;
// //             background: linear-gradient(135deg, #6366f1, #8b5cf6);
// //             position: absolute;
// //             top: 0;
// //             left: 5px;
// //             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
// //             overflow: hidden;
// //             z-index: 2;
// //         }
        
// //         .mascot-face {
// //             width: 100%;
// //             height: 100%;
// //             position: relative;
// //             display: flex;
// //             flex-direction: column;
// //             align-items: center;
// //             justify-content: center;
// //         }
        
// //         .mascot-eyes {
// //             display: flex;
// //             justify-content: space-around;
// //             width: 70%;
// //             margin-top: 10px;
// //         }
        
// //         .mascot-eye {
// //             width: 8px;
// //             height: 8px;
// //             background-color: white;
// //             border-radius: 50%;
// //             position: relative;
// //         }
        
// //         .mascot-mouth {
// //             width: 16px;
// //             height: 6px;
// //             background-color: white;
// //             border-radius: 50%;
// //             margin-top: 6px;
// //             position: relative;
// //             transition: all 0.2s ease;
// //         }
        
// //         .mascot-mouth.thinking {
// //             width: 10px;
// //             height: 10px;
// //             border-radius: 50%;
// //             background-color: transparent;
// //             border: 2px solid white;
// //             margin-top: 5px;
// //         }
        
// //         .mascot-body {
// //             width: 30px;
// //             height: 15px;
// //             background: linear-gradient(135deg, #6366f1, #8b5cf6);
// //             border-radius: 0 0 15px 15px;
// //             position: absolute;
// //             top: 35px;
// //             left: 10px;
// //             z-index: 1;
// //         }
        
// //         /* Speaking animation */
// //         @keyframes speak {
// //             0%, 100% { transform: scaleY(1); }
// //             50% { transform: scaleY(1.5); }
// //         }
        
// //         .mascot-container.speaking .mascot-mouth {
// //             animation: speak 0.3s infinite;
// //             height: 5px;
// //             width: 14px;
// //         }
        
// //         /* Breathing animation for idle state */
// //         @keyframes breathe {
// //             0%, 100% { transform: translateY(0); }
// //             50% { transform: translateY(2px); }
// //         }
        
// //         .mascot-container:not(.speaking) .mascot {
// //             animation: breathe 3s infinite ease-in-out;
// //         }
        
// //         /* Blinking animation */
// //         @keyframes blink {
// //             0%, 95%, 100% { transform: scaleY(1); }
// //             97% { transform: scaleY(0.1); }
// //         }
        
// //         .mascot-eye {
// //             animation: blink 4s infinite;
// //         }
        
// //         /* Thinking animation */
// //         @keyframes thinking {
// //             0%, 100% { transform: scale(1); }
// //             50% { transform: scale(1.1); }
// //         }
        
// //         .mascot-container.typing .mascot-mouth.thinking {
// //             animation: thinking 1.5s infinite;
// //         }
// //     `;
    
// //     document.head.appendChild(styleSheet);
// // }






































// Initialize required variables
let chatHistory = [];
const API_KEY = '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'; // Replace with your actual Groq API key
let speechSynthesis = window.speechSynthesis;
let selectedVoice = null;
let hindiVoice = null;
let englishVoice = null;
let isMascotSpeaking = false;

document.addEventListener('DOMContentLoaded', function() {
    // Add CSS for Serene avatar
    addSereneStyles();
    
    // Add Serene's template to the body
    addSereneTemplate();
    
    initChatInterface();
    initTextToSpeech();
});

function addSereneTemplate() {
    // Check if the template already exists
    if (document.getElementById('serene-template')) return;
    
    // Create a template div that will be cloned for each message
    const template = document.createElement('div');
    template.id = 'serene-template';
    template.style.display = 'none';
    
    template.innerHTML = `
        <div class="serene-container">
            <div class="serene-avatar">
                <div class="avatar-frame">
                    <img src="https://cdnjs.cloudflare.com/ajax/libs/ionicons/5.5.4/collection/components/icon/svg/avatar-female.svg" 
                         alt="Serene" class="serene-base-image">
                    <div class="serene-face">
                        <div class="serene-eyes">
                            <div class="serene-eye left">
                                <div class="serene-pupil"></div>
                                <div class="serene-eyelid"></div>
                            </div>
                            <div class="serene-eye right">
                                <div class="serene-pupil"></div>
                                <div class="serene-eyelid"></div>
                            </div>
                        </div>
                        <div class="serene-mouth">
                            <div class="serene-mouth-inner"></div>
                        </div>
                    </div>
                </div>
                <div class="serene-status">Serene is listening...</div>
            </div>
        </div>
    `;
    
    document.body.appendChild(template);
    
    // Load additional libraries dynamically
    loadExternalLibraries();
}

function loadExternalLibraries() {
    // Load Lottie animation library
    const lottieScript = document.createElement('script');
    lottieScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.9.6/lottie.min.js';
    document.head.appendChild(lottieScript);
    
    // Load FontAwesome for icons if not already loaded
    if (!document.querySelector('link[href*="font-awesome"]')) {
        const fontAwesomeLink = document.createElement('link');
        fontAwesomeLink.rel = 'stylesheet';
        fontAwesomeLink.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css';
        document.head.appendChild(fontAwesomeLink);
    }
}

function initTextToSpeech() {
    // Initialize speech synthesis
    if ('speechSynthesis' in window) {
        // Try loading voices immediately
        let voices = speechSynthesis.getVoices();
        
        // If voices aren't loaded yet, wait for the event
        if (voices.length === 0) {
            speechSynthesis.onvoiceschanged = setupVoices;
        } else {
            setupVoices();
        }
    } else {
        console.warn("Text-to-speech not supported in this browser");
    }
}

function setupVoices() {
    const voices = speechSynthesis.getVoices();
    
    // Log all available voices to console for debugging
    console.log("All available voices:", voices.map(v => `${v.name} (${v.lang})`));
    
    // Find Hindi male voice
    hindiVoice = voices.find(voice => 
        voice.lang.includes('hi') && 
        (voice.name.toLowerCase().includes('male') || 
        voice.name.toLowerCase().includes('purush'))
    );
    
    // If no Hindi male voice, try any Hindi voice
    if (!hindiVoice) {
        hindiVoice = voices.find(voice => voice.lang.includes('hi'));
    }
    
    // Find an appropriate female voice for Serene
    const femaleSoundingVoices = [
        'Google US English Female', 'Microsoft Zira', 'Samantha', 
        'Google UK English Female', 'Victoria', 'Fiona', 'Moira', 
        'Lisa', 'Karen', 'Ellen', 'Amy', 'female'
    ];
    
    // Try to find a female voice first (for Serene)
    const femaleVoice = voices.find(voice => 
        voice.lang.includes('en') && 
        femaleSoundingVoices.some(name => 
            voice.name.toLowerCase().includes(name.toLowerCase())
        )
    );
    
    // Find English male voice (as backup)
    const commonMaleVoiceNames = [
        'Microsoft David', 'Google UK English Male', 'Daniel', 
        'Microsoft Mark', 'Microsoft James', 'Google US English Male',
        'Alex', 'Fred', 'Aaron', 'Reed', 'Ralph', 'Bruce', 'Tom',
        'male'
    ];
    
    // Find English male voice
    englishVoice = voices.find(voice => 
        voice.lang.includes('en') && 
        commonMaleVoiceNames.some(name => 
            voice.name.toLowerCase().includes(name.toLowerCase())
        )
    );
    
    // If no specific male voice is found, use any English voice
    if (!englishVoice) {
        englishVoice = voices.find(voice => voice.lang.includes('en'));
    }
    
    // Set default voice - prefer female for Serene
    selectedVoice = femaleVoice || englishVoice || voices[0];
    
    console.log("Selected voice for Serene:", selectedVoice ? selectedVoice.name : "None found");
    console.log("Selected English voice (backup):", englishVoice ? englishVoice.name : "None found");
    console.log("Selected Hindi voice:", hindiVoice ? hindiVoice.name : "None found");
    
    // Create voice selector
    createVoiceSelector(voices);
}

function createVoiceSelector(voices) {
    // Check if the selector already exists
    if (document.getElementById('voice-selector')) return;
    
    // Create a voice selection dropdown
    const selector = document.createElement('select');
    selector.id = 'voice-selector';
    selector.className = 'ml-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white';
    
    // Add an option for each voice
    voices.forEach((voice, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${voice.name} (${voice.lang})`;
        selector.appendChild(option);
    });
    
    // Add language selector
    const langSelector = document.createElement('select');
    langSelector.id = 'language-selector';
    langSelector.className = 'ml-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white';
    
    const autoOption = document.createElement('option');
    autoOption.value = 'auto';
    autoOption.textContent = 'Auto-detect';
    langSelector.appendChild(autoOption);
    
    const engOption = document.createElement('option');
    engOption.value = 'english';
    engOption.textContent = 'English';
    langSelector.appendChild(engOption);
    
    const hindiOption = document.createElement('option');
    hindiOption.value = 'hindi';
    hindiOption.textContent = 'Hindi';
    langSelector.appendChild(hindiOption);
    
    // Handle selection change
    selector.addEventListener('change', function() {
        selectedVoice = voices[this.value];
        console.log("User selected voice:", selectedVoice.name);
    });
    
    // Add to interface - find a good place to insert it
    const inputArea = document.getElementById('message-input');
    if (inputArea && inputArea.parentNode) {
        const container = document.createElement('div');
        container.className = 'flex items-center mt-2';
        container.innerHTML = '<label class="text-white mr-2">Serene\'s Voice:</label>';
        container.appendChild(selector);
        
        const langContainer = document.createElement('div');
        langContainer.className = 'flex items-center mt-2 ml-4';
        langContainer.innerHTML = '<label class="text-white mr-2">Language:</label>';
        langContainer.appendChild(langSelector);
        
        inputArea.parentNode.parentNode.appendChild(container);
        inputArea.parentNode.parentNode.appendChild(langContainer);
    }
}

// Function to detect language
function detectLanguage(text) {
    // Simple detection based on Unicode character ranges
    const hindiPattern = /[\u0900-\u097F]/;
    return hindiPattern.test(text) ? 'hindi' : 'english';
}

function animateSerene(state) {
    const sereneElements = document.querySelectorAll('.serene-container');
    
    sereneElements.forEach(container => {
        // Remove all current states
        container.classList.remove('idle', 'speaking', 'thinking', 'listening');
        
        // Add the new state
        container.classList.add(state);
        
        // Update status text
        const statusElement = container.querySelector('.serene-status');
        if (statusElement) {
            switch (state) {
                case 'speaking':
                    statusElement.textContent = "Serene is speaking...";
                    break;
                case 'thinking':
                    statusElement.textContent = "Serene is thinking...";
                    break;
                case 'listening':
                    statusElement.textContent = "Serene is listening...";
                    break;
                default:
                    statusElement.textContent = "I'm Serene. Click me to hear me speak!";
            }
        }
    });
    
    isMascotSpeaking = (state === 'speaking');
}

function speakText(text) {
    if (!speechSynthesis) return;
    
    // Stop any ongoing speech
    speechSynthesis.cancel();
    
    // Break the text into sentences (include Hindi sentence endings)
    const sentences = text.split(/(?<=[.!?।])\s+/);
    let currentIndex = 0;
    
    // Get language preference
    const languageSelector = document.getElementById('language-selector');
    const languagePreference = languageSelector ? languageSelector.value : 'auto';
    
    // Activate Serene animation
    animateSerene('speaking');
    
    // Function to speak the next chunk
    function speakNextChunk() {
        if (currentIndex >= sentences.length) {
            // Stop Serene animation when done speaking
            animateSerene('idle');
            return;
        }
        
        // Create manageable chunks of 1-3 sentences
        let chunk = sentences[currentIndex];
        const maxChunkLength = 250; // Maximum characters per utterance
        
        // If the current sentence is short, combine with next sentences up to maxChunkLength
        while (currentIndex + 1 < sentences.length && 
               chunk.length + sentences[currentIndex + 1].length < maxChunkLength) {
            currentIndex++;
            chunk += " " + sentences[currentIndex];
        }
        
        // Determine language for this chunk
        let chunkLanguage;
        if (languagePreference === 'auto') {
            chunkLanguage = detectLanguage(chunk);
        } else {
            chunkLanguage = languagePreference;
        }
        
        // Create utterance for this chunk
        const utterance = new SpeechSynthesisUtterance(chunk);
        
        // Set voice based on language
        if (chunkLanguage === 'hindi' && hindiVoice) {
            utterance.voice = hindiVoice;
            utterance.lang = 'hi-IN';
        } else {
            utterance.voice = selectedVoice; // Use selected voice (preferably female)
            utterance.lang = 'en-US';
        }
        
        // Voice characteristics
        utterance.rate = 1.0;
        utterance.pitch = 1.2; // Slightly higher pitch for female-like voice
        
        // Set up callback for when this chunk finishes
        utterance.onend = () => {
            currentIndex++;
            speakNextChunk();
        };
        
        // Handle errors and cancellations
        utterance.onerror = (event) => {
            console.error("Speech synthesis error:", event);
            currentIndex = sentences.length; // Stop on error
            animateSerene('idle'); // Stop animation on error
        };
        
        // Start speaking this chunk
        speechSynthesis.speak(utterance);
    }
    
    // Start the first chunk
    speakNextChunk();
}

function initChatInterface() {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const speakToggle = document.getElementById('speak-toggle');
    let speakResponses = true;
    
    // Initialize speak toggle
    if (speakToggle) {
        speakToggle.addEventListener('change', function() {
            speakResponses = this.checked;
        });
    }
    
    // Add a welcome message with Serene introduction
    setTimeout(() => {
        addMessageToChat('bot', "Hi, I'm Serene! I'm here to assist you today. You can ask me anything, and I'll do my best to help. Click on my avatar if you'd like to hear me speak my responses.");
    }, 1000);
    
    // Handle form submission
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const message = messageInput.value.trim();
            if (message) {
                sendMessage(message);
                messageInput.value = '';
            }
        });
    }

    // Handle quick reply buttons
    document.querySelectorAll('.quick-reply').forEach(button => {
        button.addEventListener('click', function() {
            const message = this.textContent.trim();
            sendMessage(message);
        });
    });

    // Function to send message
    function sendMessage(message) {
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Show typing indicator
        showTypingIndicator();
        
        // Update chat history
        chatHistory.push({
            role: 'user',
            content: message
        });
        
        // Set Serene to thinking state
        animateSerene('thinking');
        
        // Send to Groq API
        callGroqAPI(message)
            .then(response => {
                // Hide typing indicator
                hideTypingIndicator();
                
                // Add bot response to chat
                addMessageToChat('bot', response);
                
                // Update chat history
                chatHistory.push({
                    role: 'assistant',
                    content: response
                });
                
                // Speak the response if enabled
                if (speakResponses) {
                    // Extract text without markdown formatting for speech
                    const plainText = response.replace(/```[\s\S]*?```/g, 'code block omitted')
                                            .replace(/`([^`]+)`/g, '$1')
                                            .replace(/\*\*([^*]+)\*\*/g, '$1')
                                            .replace(/\*([^*]+)\*/g, '$1');
                    speakText(plainText);
                } else {
                    // If not speaking, set Serene to idle state
                    animateSerene('idle');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                hideTypingIndicator();
                
                // Show error message
                const errorMessage = "I'm sorry, I'm having trouble connecting to Groq right now. Please check your API key and try again.";
                addMessageToChat('bot', errorMessage);
                
                // Set Serene to idle state
                animateSerene('idle');
                
                // Speak the error message if enabled
                if (speakResponses) {
                    speakText(errorMessage);
                }
            });
    }

    // Function to call Groq API
    async function callGroqAPI(message) {
        if (!API_KEY || API_KEY === 'YOUR_GROQ_API_KEY') {
            return "Please set your Groq API key in the chat.js file.";
        }
        
        try {
            const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEY}`
                },
                body: JSON.stringify({
                    model: 'llama3-70b-8192', // You can change to other Groq models
                    messages: chatHistory,
                    temperature: 0.7,
                    max_tokens: 1024
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error.message);
            }
            
            return data.choices[0].message.content;
        } catch (error) {
            console.error('Error calling Groq API:', error);
            throw error;
        }
    }

    // Function to add message to chat
    function addMessageToChat(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('flex', 'items-start', 'space-x-4', `${sender}-message`, 'animate-fade-in');
        
        if (sender === 'user') {
            messageElement.innerHTML = `
                <div class="flex-1 flex justify-end">
                    <div class="bg-gradient-to-r from-purple-600 to-blue-500 rounded-2xl rounded-tr-none p-4 shadow-lg max-w-md message-bubble">
                        <p>${formatMessage(message)}</p>
                    </div>
                </div>
                <div class="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0 mt-1">
                    <i class="fas fa-user"></i>
                </div>
            `;
        } else {
            // For bot messages, add Serene avatar and speak button
            const formattedMessage = formatMessage(message);
            
            // Clone the Serene template
            const sereneTemplate = document.getElementById('serene-template');
            const sereneClone = sereneTemplate.cloneNode(true);
            sereneClone.style.display = 'block';
            sereneClone.id = ''; // Remove the id from the clone
            
            // Create the message container
            messageElement.innerHTML = `
                <div class="serene-message-container">
                </div>
                <div class="flex-1">
                    <div class="bg-gray-800/80 backdrop-blur-md rounded-2xl rounded-tl-none p-4 shadow-lg border border-white/10 message-bubble">
                        <p>${formattedMessage}</p>
                        <button class="speak-btn text-xs mt-2 text-blue-400 hover:text-blue-300">
                            <i class="fas fa-volume-up mr-1"></i>Hear Serene
                        </button>
                    </div>
                </div>
            `;
            
            // Insert the Serene clone into the message container
            const container = messageElement.querySelector('.serene-message-container');
            container.appendChild(sereneClone);
            
            // Add event listener to speak button
            setTimeout(() => {
                const speakBtn = messageElement.querySelector('.speak-btn');
                if (speakBtn) {
                    speakBtn.addEventListener('click', function() {
                        // Extract text without markdown formatting for speech
                        const plainText = message.replace(/```[\s\S]*?```/g, 'code block omitted')
                                            .replace(/`([^`]+)`/g, '$1')
                                            .replace(/\*\*([^*]+)\*\*/g, '$1')
                                            .replace(/\*([^*]+)\*/g, '$1');
                        speakText(plainText);
                    });
                }
                
                // Add click event to Serene avatar to trigger speech
                const sereneAvatar = messageElement.querySelector('.serene-avatar');
                if (sereneAvatar) {
                    sereneAvatar.addEventListener('click', function() {
                        if (!isMascotSpeaking) {
                            const plainText = message.replace(/```[\s\S]*?```/g, 'code block omitted')
                                                    .replace(/`([^`]+)`/g, '$1')
                                                    .replace(/\*\*([^*]+)\*\*/g, '$1')
                                                    .replace(/\*([^*]+)\*/g, '$1');
                            speakText(plainText);
                        } else {
                            // If already speaking, stop
                            speechSynthesis.cancel();
                            animateSerene('idle');
                        }
                    });
                }
            }, 0);
        }
        
        chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to format message with markdown support
    function formatMessage(message) {
        // Basic markdown formatting
        // Convert code blocks
        message = message.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
        
        // Convert inline code
        message = message.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Convert bold text
        message = message.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Convert italic text
        message = message.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
        // Convert line breaks
        message = message.replace(/\n/g, '<br>');
        
        return message;
    }

    // Function to show typing indicator
    function showTypingIndicator() {
        const typingElement = document.createElement('div');
        typingElement.id = 'typing-indicator';
        typingElement.classList.add('flex', 'items-start', 'space-x-4', 'bot-message');
        
        // Clone the Serene template for the typing indicator
        const sereneTemplate = document.getElementById('serene-template');
        const sereneClone = sereneTemplate.cloneNode(true);
        sereneClone.style.display = 'block';
        sereneClone.id = ''; // Remove the id from the clone
        sereneClone.querySelector('.serene-container').classList.add('thinking');
        
        typingElement.innerHTML = `
            <div class="serene-message-container">
            </div>
            <div class="bg-gray-800/80 backdrop-blur-md rounded-2xl rounded-tl-none p-2 shadow-lg border border-white/10">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        // Insert the Serene clone into the typing indicator
        const container = typingElement.querySelector('.serene-message-container');
        container.appendChild(sereneClone);
        
        chatMessages.appendChild(typingElement);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to hide typing indicator
    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    // Add a clear chat button function
    const clearChatButton = document.getElementById('clear-chat');
    if (clearChatButton) {
        clearChatButton.addEventListener('click', function() {
            chatMessages.innerHTML = '';
            chatHistory = [];
            
            // Add a system message with Serene
            addMessageToChat('bot', "Chat history cleared. How can I help you today?");
            
            // Speak the message if enabled
            if (speakResponses) {
                speakText("Chat history cleared. How can I help you today?");
            }
        });
    }
    
    // Add a stop speaking button
    const stopSpeakingButton = document.getElementById('stop-speaking');
    if (stopSpeakingButton) {
        stopSpeakingButton.addEventListener('click', function() {
            if (speechSynthesis) {
                speechSynthesis.cancel();
                animateSerene('idle');
            }
        });
    }
}

// Function to add Serene styles
function addSereneStyles() {
    const styleSheet = document.createElement("style");
    styleSheet.textContent = `
        /* Serene Container */
        .serene-container {
            width: 120px;
            height: 160px;
            position: relative;
            cursor: pointer;
            transition: transform 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .serene-container:hover {
            transform: scale(1.05);
        }
        
        /* Serene Avatar */
        .serene-avatar {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }
        
        .avatar-frame {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            overflow: hidden;
            background: linear-gradient(135deg, #f3e7fa, #d8b5ff);
            position: relative;
            box-shadow: 0 4px 15px rgba(107, 70, 193, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .serene-base-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.9;
        }
        
        .serene-face {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            pointer-events: none;
        }
        
        /* Serene Eyes */
        .serene-eyes {
            display: flex;
            width: 80%;
            justify-content: space-around;
            margin-top: -10px;
        }
        
        .serene-eye {
            width: 24px;
            height: 24px;
            background-color: #ffffff;
            border-radius: 50%;
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }
        
        .serene-pupil {
            width: 12px;
            height: 12px;
            background-color: #2a9d8f;
            border-radius: 50%;
            position: relative;
        }
        
        .serene-eyelid {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 0%;
            background-color: #f3e7fa;
            transition: height 0.2s ease;
        }
        
        /* Serene Mouth */
        .serene-mouth {
            width: 30px;
            height: 15px;
            margin-top: 10px;
            position: relative;
            overflow: hidden;
        }
        
        .serene-mouth-inner {
            width: 100%;
            height: 30px;
            border-radius: 50%;
            background-color: #ff7096;
            position: absolute;
            top: 0;
            transform: scaleX(1.5);
        }
        
        /* Serene Status Text */
        .serene-status {
            font-size: 12px;
            color: #ffffff;
            text-align: center;
            margin-top: 5px;
            background-color: rgba(107, 70, 193, 0.7);
            padding: 4px 8px;
            border-radius: 12px;
            max-width: 90%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Animation States */
        
        /* Blinking animation */
        @keyframes blink {
            0%, 95%, 100% { height: 0%; }
            97%, 98% { height: 100%; }
        }
        
        /* Speaking animation */
        @keyframes speak {
            0%, 100% { height: 15px; }
            50% { height: 10px; }
        }
        
        @keyframes mouth-move {
            0%, 100% { top: 0; }
            50% { top: 5px; }
        }
        
        /* Idle state - occasional blinks */
        .serene-container .serene-eyelid {
            animation: blink 4s infinite;
        }
        
        /* Speaking state */
        .serene-container.speaking .serene-mouth {
            animation: speak 0.3s infinite;
        }
        
        .serene-container.speaking .serene-mouth-inner {
            animation: mouth-move 0.3s infinite;
        }
        
        /* Thinking state - rapid eye movement */
        @keyframes thinking-eyes {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-2px); }
            75% { transform: translateX(2px); }
        }
        
        .serene-container.thinking .serene-pupil {
            animation: thinking-eyes 2s infinite;
        }
        
        .serene-container.thinking .serene-mouth-inner {
            background-color: #9d4edd;
            height: 5px;
            top: 10px;
            transform: scaleX(0.8);
        }
        
        /* Message container for Serene */
        .serene-message-container {
            min-width: 120px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            padding: 6px 12px;
        }
        
        .typing-indicator span {
            height: 8px;
            width: 8px;
            background-color: #a0aec0;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: typing 1s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-5px);
            }
        }
        
        /* Animation for fade-in messages */
        .animate-fade-in {
            animation: fadeIn 0.5s ease-in forwards;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Message bubble styling */
        .message-bubble {
            transition: all 0.3s ease;
        }
        
        .message-bubble:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* Advanced Serene Styling - More realistic avatar */
        .serene-realistic-avatar {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            border: 3px solid #f0f0f0;
            transition: all 0.3s ease;
        }
        
        .serene-realistic-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        /* Floating animation for the avatar */
        @keyframes floating {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .serene-realistic-container {
            animation: floating 3s ease-in-out infinite;
        }
        
        /* Expression animations */
        .serene-expression {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            transition: opacity 0.5s ease;
        }
        
        .serene-expression.active {
            opacity: 1;
        }
        
        /* Chat container */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            background-color: rgba(15, 23, 42, 0.8);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Header styling */
        .chat-header {
            padding: 16px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        /* Messages area */
        .chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            background-color: rgba(15, 23, 42, 0.6);
        }
        
        /* Input area */
        .chat-input {
            display: flex;
            padding: 16px;
            background-color: rgba(30, 41, 59, 0.8);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .input-field {
            flex: 1;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background-color: rgba(15, 23, 42, 0.6);
            color: white;
            font-size: 16px;
        }
        
        .send-button {
            margin-left: 8px;
            padding: 12px 20px;
            border-radius: 8px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }
        
        /* Controls area */
        .chat-controls {
            display: flex;
            padding: 8px 16px;
            background-color: rgba(30, 41, 59, 0.5);
            align-items: center;
            gap: 16px;
        }
        
        .control-button {
            background: rgba(99, 102, 241, 0.2);
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .control-button:hover {
            background: rgba(99, 102, 241, 0.4);
        }
        
        /* Toggle switch */
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 40px;
            height: 24px;
        }
        
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(30, 41, 59, 0.8);
            transition: .4s;
            border-radius: 24px;
        }
        
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        
        input:checked + .toggle-slider {
            background-color: #6366f1;
        }
        
        input:checked + .toggle-slider:before {
            transform: translateX(16px);
        }
    `;
    
    document.head.appendChild(styleSheet);
}