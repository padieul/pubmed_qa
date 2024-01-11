// public/script.js
function sendMessage() {
  const userMessage = document.getElementById('user-input').value;
  appendMessage('You', userMessage);

  // TODO: Send the user's message to the Python middleware for processing
  // and retrieve the response from the Elasticsearch backend.

  // sample bot response
  const botResponse = "This is a placeholder response from the backend.";
  appendMessage('Bot', botResponse);

  // Clear the user input
  document.getElementById('user-input').value = '';
}

function appendMessage(sender, message) {
  const chatMessages = document.getElementById('chat-messages');
  const messageElement = document.createElement('div');
  const senderClass = sender === 'You' ? 'user-message' : 'bot-message';
  messageElement.className = senderClass;
  messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
  chatMessages.appendChild(messageElement);

  // Scroll to the bottom of the chat container
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
