<<<<<<< HEAD:app/frontend/index.js
<<<<<<< HEAD:Frontend/index.js
=======
>>>>>>> f6bf02d (Revert "code refactoring and restructuring"):Frontend/index.js
// index.js
const express = require('express');
const app = express();
const port = 3000;

app.use(express.static('public'));
<<<<<<< HEAD:app/frontend/index.js

app.get('/', (req, res) => {
  res.sendFile('index.html', { root: __dirname + '/public' });
=======
app.use(express.json());

app.get('/', (req, res) => {
  res.sendFile('C:/Users/sushm/OneDrive/Desktop/Sem1/NLP/publicindex.html', { root: __dirname + '/public' });
>>>>>>> f6bf02d (Revert "code refactoring and restructuring"):Frontend/index.js
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
<<<<<<< HEAD:app/frontend/index.js
=======
// index.js
const express = require('express');
const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.json());

app.get('/', (req, res) => {
  res.sendFile('C:/Users/sushm/OneDrive/Desktop/Sem1/NLP/publicindex.html', { root: __dirname + '/public' });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
>>>>>>> 1b3914f (code refactoring and restructuring):app/frontend/index.js
=======
>>>>>>> f6bf02d (Revert "code refactoring and restructuring"):Frontend/index.js
