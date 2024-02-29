<!-- src/Chatbot.svelte -->
<script>
  let messages = [];
  let userInput = '';

  // Call the function when the page loads
fetchStorageInfo();



async function sendToBackend(message) {
    // Using fetch API to send a GET request to the FastAPI endpoint
    const response = await fetch(`http://localhost:8000/retrieve_documents_dense?query_str=${encodeURIComponent(message)}`);

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

  async function sendMessage() {
    if (userInput.trim() !== '') {
      messages = [...messages, { sender: 'You', message: userInput }];

      try {
        // TODO: Add logic to send the message to the backend and get the bot's response
        const response = await sendToBackend(userInput);
        const botResponse = response.message;

        // Display the bot's response
        messages = [...messages, { sender: 'Bot', message: botResponse }];
      } catch (error) {
        console.error('Error sending message to backend:', error);
      }

      userInput = '';
    }
  }
</script>

<style>
  body {
    font-family: Arial, sans-serif;
    background-image: url('images/medicine.jpg');
    background-size: cover;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
      }
  
  #chat-container {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    overflow: hidden;
    width: 600px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
  }

  #storage-info-container {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    overflow: hidden;
    width: 200px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
  }
  
  #chat-messages {
    padding: 10px;
    max-height: 300px;
    overflow-y: auto;
  }
  
  #user-input-container {
    display: flex;
    align-items: center;
    padding: 10px;
  }
  
  #user-input {
    flex: 1;
    padding: 8px;
    margin-right: 10px;
    border: none;
    border-radius: 5px;
  }
  
  #send-button {
    padding: 8px;
    border: none;
    border-radius: 5px;
    background-color: #4CAF50;
    color: white;
    cursor: pointer;
  }
</style>

<div>
  <div>
    {#each messages as { sender, message }}
      <div><strong>{sender}:</strong> {message}</div>
    {/each}
  </div>
  <div>
    <input bind:value={userInput} placeholder="Type a message..." />
    <button on:click={sendMessage}>Send</button>
  </div>
</div>
