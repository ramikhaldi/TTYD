import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_HOST = process.env.REACT_APP_CHATBOT_HOST; 
const API_PORT = process.env.REACT_APP_TTYD_API_PORT; 
const API_URL = `http://${API_HOST}:${API_PORT}/ask`;

console.log("Using API URL:", API_URL);  // Debugging

// A simple component that shows three animated dots while loading.
const LoadingDots = () => (
  <div className="loading-dots">
    <span>.</span>
    <span>.</span>
    <span>.</span>
  </div>
);

function App() {
  const urlParams = new URLSearchParams(window.location.search);
  const sessionID = urlParams.get('session') || "Dummy Session";

  const user = "User";
  const botName = "TTYD Bot";

  const [chatMessages, setChatMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const addMessage = (sender, text) => {
    setChatMessages(prev => [
      ...prev,
      { sender, text, time: new Date().toLocaleTimeString() }
    ]);
  };

  const removeProcessingMessage = () => {
    setChatMessages(prev => {
      if (
        prev.length &&
        prev[prev.length - 1].sender === botName &&
        prev[prev.length - 1].text === "Processing..."
      ) {
        return prev.slice(0, -1);
      }
      return prev;
    });
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;
  
    addMessage(user, inputMessage);
    const messageToSend = inputMessage;
    setInputMessage('');
  
    setLoading(true);
    addMessage(botName, "Processing...");
  
    const payload = JSON.stringify({
      sessionID: sessionID,
      question: messageToSend,
    });
  
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        mode: 'cors',  // ✅ Ensure CORS is enabled
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: payload
      });
  
      if (!response.ok) {
        removeProcessingMessage();
        addMessage(botName, `Error: ${response.statusText}`);
        return;
      }
  
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let result = "";
  
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value, { stream: true });
  
        setChatMessages(prev => {
          if (prev.length > 0 && prev[prev.length - 1].sender === botName) {
            const updatedMsg = { ...prev[prev.length - 1], text: result };
            return [...prev.slice(0, prev.length - 1), updatedMsg];
          }
          return prev;
        });
      }
    } catch (error) {
      console.error(error);
      removeProcessingMessage();
      addMessage(botName, "Error: Network or server error.");
    }
    setLoading(false);
  };
  

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-left">
          <div className="header-title">{botName}</div>
          <div className="session-badge">Session: {sessionID}</div>
        </div>
      </header>
      <div className="chat-container">
        <div className="chat-messages">
          {chatMessages.map((msg, index) => (
            <div
              key={index}
              className={`chat-message ${msg.sender === user ? 'user-message' : 'bot-message'}`}
            >
              <div className="message-sender">{msg.sender}:</div>
              <div className="message-text">
                {msg.sender === botName && msg.text === "Processing..." ? (
                  <LoadingDots />
                ) : msg.sender === botName ? (
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                ) : (
                  msg.text
                )}
              </div>
              <div className="message-time">{msg.time}</div>
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input
            type="text"
            placeholder="Type your message here..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') sendMessage(); }}
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
