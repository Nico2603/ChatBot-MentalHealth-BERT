document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('send');
    const messageInput = document.getElementById('message');
    const voiceButton = document.getElementById('voice');
    const chatbox = document.getElementById('chatbox');
    const recordingIndicator = document.getElementById('recordingIndicator');

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    voiceButton.addEventListener('click', startRecognition);

    function sendMessage() {
        const message = messageInput.value.trim();
        if (message === '') return;

        addMessageToChatbox('Usuario', message);
        messageInput.value = '';
        toggleInput(false);

        // Mostrar indicador de carga
        addTypingIndicator();

        fetch('/get_response', {
            method: 'POST',
            body: new URLSearchParams({'message': message}),
        })
        .then(response => response.ok ? response.json() : response.json().then(err => Promise.reject(err)))
        .then(data => {
            removeTypingIndicator();
            addMessageToChatbox('Asistente', data.response);
            playResponse(data.audio_path);
        })
        .catch(error => {
            removeTypingIndicator();
            console.error('Error:', error);
            addMessageToChatbox('Asistente', error.response || 'Lo siento, ha ocurrido un error al procesar tu solicitud.');
        })
        .finally(() => {
            toggleInput(true);
        });
    }

    function addMessageToChatbox(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = sender === 'Usuario' ? 'message user-message' : 'message bot-message';
        messageDiv.innerHTML = `<p>${message}</p>`;
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function addTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.id = 'typingIndicator';
        typingIndicator.className = 'message bot-message';
        typingIndicator.innerHTML = '<p>Escribiendo<span class="dot-one">.</span><span class="dot-two">.</span><span class="dot-three">.</span></p>';
        chatbox.appendChild(typingIndicator);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            chatbox.removeChild(typingIndicator);
        }
    }

    function startRecognition() {
        if (!('webkitSpeechRecognition' in window)) {
            alert('Tu navegador no soporta reconocimiento de voz.');
            return;
        }

        const recognition = new webkitSpeechRecognition();
        recognition.lang = 'es-ES';
        recognition.start();

        if (recordingIndicator) {
            recordingIndicator.style.display = 'block';
        }

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            messageInput.value = transcript;
            sendMessage();
        };

        recognition.onerror = (event) => {
            console.error('Error en el reconocimiento de voz:', event.error);
            alert('Ocurrió un error durante el reconocimiento de voz: ' + event.error);
        };

        recognition.onend = () => {
            if (recordingIndicator) {
                recordingIndicator.style.display = 'none';
            }
        };
    }

    function playResponse(audioPath) {
        if (audioPath) {
            console.log('Reproduciendo audio desde:', audioPath);
            const audio = new Audio(audioPath);
            audio.play().catch(error => {
                console.error('Error al reproducir el audio:', error);
            });
        }
    }

    function toggleInput(enable) {
        messageInput.disabled = !enable;
        sendButton.disabled = !enable;
        voiceButton.disabled = !enable;
    }
});