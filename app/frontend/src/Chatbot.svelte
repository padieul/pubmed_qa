<!-- src/Chatbot.svelte -->
<script>
  let messages = [];
  let userInput = '';
  let sender = '';

  let messageNew = '';
  let references = [];
  let globalStatusMessage = '';
  let globalStatus = '';

  let intervalId = setInterval(async function() {
    let { statusMessage, status } = await fetchServerStatus();

      if (status === 'OK') {
        globalStatus = 'OK';
        clearInterval(intervalId);
      }
      else if (status === 'NOK') {
        globalStatus = 'NOK';
        globalStatusMessage = statusMessage;
      }
    }, 1000);

  async function fetchServerStatus() {
  try {
      const response = await fetch('http://localhost:8000/server_status'); // replace with your actual API endpoint

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      let serverStatusMessage = await response.json();
      return {statusMessage: serverStatusMessage.serverMessage, status: serverStatusMessage.serverStatus};

      } catch (error) {
      console.error('Backend error:', error);
  }
}
  



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

function extractReferences(message) {

    console.log("Extracting references...")
    // separate message string by "_" character
    let temp = message.split("_");
    if (temp.length > 1) {
        const references = temp[1].split("|").slice(1);
        messageNew = temp[0]
        console.log(messageNew)
        console.log(references)
        return messageNew, references
    }
    else {
      return message, []
    }

    
}

// main.js




  async function sendMessage() {
    if (userInput.trim() !== '') {
      sender = "You"
      messages = [...messages, { sender: 'You', message: userInput, references: [] }];
      

      try {
        // TODO: Add logic to send the message to the backend and get the bot's response
        const response = await sendToBackend(userInput);
        //const botResponse = response.message;
        const responseText = response.message 
        
        // print to console: responseText, references
        // console.log(responseText);

        messageNew, references = extractReferences(responseText)
        console.log(references)
        console.log(messageNew)

        // Display the bot's response
        sender = "Bot"
        messages = [...messages, { sender: 'Bot', message: messageNew, references: references}];
        
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
    background-image: url('/images/medicine.jpg');
    background-size: cover;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    cursor: url('/images/stethoscope.png') 100 100, auto;
  }

  #main-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
  }

  #server-info {
    border-radius: 5px;
  }

  #server-placeholder {
    border-radius: 5px;
    text-align: center;
    color: green;
  }

  #storage-info-container {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    overflow: hidden;
    width: 200px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
  }

  #chat-container {
    display: flex;
    align-items: center;
    padding: 10px;
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    overflow: hidden;
    width: 600px;
    max-height: 1000px; /* Add this line */
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
  }

  #chat-container:empty {
  display: none;
  }

  #user-input-container {
    display: flex;
    align-items: center;
    padding: 10px;
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    overflow: hidden;
    width: 600px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
  }

  #chat-messages {
    padding: 10px;
    max-height: 300px;
    overflow-y: auto;
  }

  #chat-messages:empty {
  display: none;
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

  #bot-message-container {
    padding: 10px;
    border-radius: 5px;
    border: 2px solid rgb(237, 222, 157);
    background-color: rgb(237, 222, 157);
  }

  #sources-container {
    display: flex;
    flex-direction: column; /* Stack words vertically */
    align-items: left; /* Center items horizontally */
    gap: 10px; /* Space between items */
    margin: 5px 0; /* Add this line */
  }

  #reference-container {
    border: 2px solid lighblue;
    background-color: lightblue;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Soft shadow for depth */
    padding: 2px;
    cursor: pointer;
  }

  #reference-container:hover {
    background-color: blue;
  }
</style>

<body>
<div id="main-container">
  
  <!--
  <div id="storage-info-container">
    <p id="storage-info">Storage information goes here</p>
  </div>
  -->

  <div id="chat-container">
    <div id="chat-messages">
    {#if globalStatus !== 'OK'}
      <div id="server-info">
        <strong>Server status: </strong> {globalStatusMessage}
      </div>
    {:else if messages.length === 0}
      <div id="server-placeholder">
        <strong>The server is running now!</strong>
      </div>
    {:else}
    {#each messages as { sender, message, references }}
    {#if sender === 'You'}
      <div>
        <strong>{sender}:</strong> {message}
      </div>
    {:else if sender === 'Bot'}
    <div id="bot-message-container">
      <div>
        <strong>{"Answer:"}:</strong> {message}
      </div>
      <strong>{"Sources:"}</strong>
      <div id="sources-container">
      {#each references as reference}
      <div>
      <a href={reference} target="_blank" id="reference-container">
        {reference}
      </a>
      </div>
      {/each}
      </div>
    </div>
    {/if}
    {/each}
    {/if}
    </div>
  </div>
  <div id="user-input-container">
      <input type="text" id="user-input" bind:value={userInput} placeholder="Type a message..." on:keydown={(e) => e.key === 'Enter' && sendMessage()} />
      <button id="send-button" on:click={sendMessage}>Send</button>
  </div>
  

</body>