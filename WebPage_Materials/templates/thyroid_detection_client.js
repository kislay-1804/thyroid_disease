document.addEventListener('DOMContentLoaded', function() {
    // Toggle background color
    const toggleBgBtn = document.getElementById('toggle-bg');
    const toggleDarkBtn = document.getElementById('toggle-dark');
    
    if (toggleBgBtn && toggleDarkBtn) {
        toggleBgBtn.addEventListener('click', function() {
            document.body.classList.remove('bg-dark');
            document.body.classList.add('bg-nav');
        });

        toggleDarkBtn.addEventListener('click', function() {
            document.body.classList.remove('bg-nav');
            document.body.classList.add('bg-dark');
        });
    }

    // Handle form submission
    const thyroidForm = document.getElementById('thyroid-form');
    
    if (thyroidForm) {
        thyroidForm.addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent default form submission

            // Collect form data
            const formData = new FormData(thyroidForm);
            const data = {};

            formData.forEach((value, key) => {
                data[key] = value;
            });

            // Send form data to the server
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                // Display result
                const resultDiv = document.getElementById('result');
                if (resultDiv) {
                    resultDiv.innerHTML = `<h3>Prediction Result: ${data.result}</h3>`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                const resultDiv = document.getElementById('result');
                if (resultDiv) {
                    resultDiv.innerHTML = `<h3>Error: ${error.message}</h3>`;
                }
            });
        });
    }
});