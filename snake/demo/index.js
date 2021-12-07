import React from 'react';
import ReactDOM from 'react-dom';
import SnakeGame from '../src/SnakeGame.jsx'
import './index.css'

ReactDOM.render(
  <React.StrictMode>
    <div id='spacer' />
    <SnakeGame />
  </React.StrictMode>,
  document.getElementById('root')
)
