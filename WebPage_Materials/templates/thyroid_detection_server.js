const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

const app = express();
const port = 8080;

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join('path/to/cssfile', 'static.css'))); // Ensure correct static path

// Serve the main HTML page
app.get('/', (req, res) => {
    res.sendFile(path.join('path/to/htmlfile', 'home.html')); // Ensure correct HTML file path
});

// Handle form submission and prediction
app.post('/predict', (req, res) => {
    const formData = req.body;

    // Convert form data to query string format
    const queryString = Object.keys(formData).map(key => `${encodeURIComponent(key)}=${encodeURIComponent(formData[key])}`).join('&');

    // Execute Python script for prediction
    const result = spawnSync('python', ['predictFromModel.py', queryString]);

    if (result.error) {
        console.error('Error executing Python script:', result.error);
        return res.status(500).json({ result: 'Error during prediction' });
    }

    // Parse the result from Python script
    const predictionResult = result.stdout.toString().trim();

    // Send result back to the client
    res.json({ result: predictionResult });
});

// Start the server
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});