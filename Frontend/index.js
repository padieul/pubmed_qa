// index.js
const express = require('express');
const app = express();
const port = 3000;

app.use(express.static('public'));

app.get('/', (req, res) => {
  res.sendFile('index.html', { root: __dirname + '/public' });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
