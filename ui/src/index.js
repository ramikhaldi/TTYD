// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';  // if you have a global stylesheet

const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);
root.render(<App />);
