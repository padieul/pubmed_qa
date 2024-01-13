

document.getElementById('user-input').addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevents the default action to be performed. If this method is called, the default action of the event will not be triggered.
        sendMessage();
    }
});

// Call the function when the page loads
fetchStorageInfo();

async function sendMessage() {
    const userMessage = document.getElementById('user-input').value;
    appendMessage('You', userMessage);

    try {
        // TODO: Add logic to send the message to the backend and get the bot's response
        const response = await sendToBackend(userMessage);
        const botResponse = response.message;

        // Display the bot's response
        appendMessage('Bot', botResponse);
    } catch (error) {
        console.error('Error sending message to backend:', error);
    }

    // Clear the user input
    document.getElementById('user-input').value = '';
}

async function sendToBackend(message) {
    // Using fetch API to send a GET request to the FastAPI endpoint
    const response = await fetch(`http://localhost:8000/retrieve_documents_dense?message=${encodeURIComponent(message)}`);

    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
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

// main.js
async function fetchStorageInfo() {
    try {
      const response = await fetch('http://localhost:8000/storage_info'); // replace with your actual API endpoint
  
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
  
      const data = await response.json();
      const storageInfo = data.info; // replace with your actual data property
  
      document.getElementById('storage-info').textContent = storageInfo;
    } catch (error) {
      console.error('Error fetching storage info:', error);
      document.getElementById('storage-info').textContent = 'Error fetching storage info';
    }
  }
  
  