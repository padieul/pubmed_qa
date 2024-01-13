

document.getElementById('user-input').addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevents the default action to be performed. If this method is called, the default action of the event will not be triggered.
        sendMessage();
    }
});

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
    const response = await fetch(`http://localhost:8000/read_root?message=${encodeURIComponent(message)}`);

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

document.getElementById('load-button').addEventListener('click', async function() {
    this.disabled = true; // disable the button to prevent multiple clicks
    this.style.backgroundColor = 'yellow'; // change the button color to yellow
    this.textContent = 'Loading...'; // change the button text to 'Loading...'
  
    try {
      const response = await fetch('http://localhost:8000/load_data'); // replace with your actual API endpoint
  
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
  
      const data = await response.json();
  
      if (data.success) { // replace with your actual success condition
        this.style.backgroundColor = 'green'; // change the button color to green
        this.textContent = 'Data Loaded'; // change the button text to 'Data Loaded'
      } else {
        throw new Error('Data loading failed');
      }
    } catch (error) {
      console.error('Error loading data:', error);
      this.style.backgroundColor = 'red'; // change the button color to red
      this.textContent = 'Error'; // change the button text to 'Error'
    }
  });