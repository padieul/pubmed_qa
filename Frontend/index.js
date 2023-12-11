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
