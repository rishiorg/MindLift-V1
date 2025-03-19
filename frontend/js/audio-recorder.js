document.addEventListener('DOMContentLoaded', function() {
    const voiceInputBtn = document.getElementById('voice-input-btn');
    const recordingIndicator = document.getElementById('recording-indicator');
    
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    
    // Setup voice recording functionality
    if (voiceInputBtn && recordingIndicator) {
        voiceInputBtn.addEventListener('click', toggleRecording);
        recordingIndicator.addEventListener('click', toggleRecording);
    }
    
    function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }
    
    function startRecording() {
        // Request microphone access
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    
                    // Convert to base64 for sending to server
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        const base64Audio = reader.result;
                        sendAudioToServer(base64Audio);
                    };
                    
                    // Reset recording
                    audioChunks = [];
                    isRecording = false;
                    recordingIndicator.classList.add('hidden');
                };
                
                // Start recording
                mediaRecorder.start();
                isRecording = true;
                recordingIndicator.classList.remove('hidden');
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                alert('Could not access microphone. Please make sure you have granted permission.');
            });
    }
    
    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            
            // Stop all tracks in the stream
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
    
    function sendAudioToServer(base64Audio) {
        // Show typing indicator
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            const typingDiv = document.createElement('div');
            typingDiv.classList.add('message', 'assistant', 'typing-indicator');
            
            const bubbleDiv = document.createElement('div');
            bubbleDiv.classList.add('message-bubble');
            
            const dotsDiv = document.createElement('div');
            dotsDiv.classList.add('typing-dots');
            
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement('span');
                dot.classList.add('dot');
                dotsDiv.appendChild(dot);
            }
            
            bubbleDiv.appendChild(dotsDiv);
            typingDiv.appendChild(bubbleDiv);
            
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Send audio to server
        fetch('/api/audio-chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ audio: base64Audio })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            const typingIndicator = document.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            // Add bot response to chat
            addMessage(data.response, 'assistant');
            
            // Speak response if speech synthesis is available
            if ('speechSynthesis' in window && typeof speakText === 'function') {
                speakText(data.response);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            
            // Remove typing indicator
            const typingIndicator = document.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            // Add error message
            addMessage("I'm sorry, I couldn't process your audio. Please try again or type your message.", 'assistant');
        });
    }
    
    // Helper function to add message to chat
    function addMessage(text, sender) {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.classList.add('message-bubble');
        
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        
        bubbleDiv.appendChild(paragraph);
        
        const infoDiv = document.createElement('div');
        infoDiv.classList.add('message-info');
        
        const timeSpan = document.createElement('span');
        timeSpan.classList.add('message-time');
        timeSpan.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        infoDiv.appendChild(timeSpan);
        
        messageDiv.appendChild(bubbleDiv);
        messageDiv.appendChild(infoDiv);
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});