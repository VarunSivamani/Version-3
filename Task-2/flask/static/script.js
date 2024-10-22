document.addEventListener('DOMContentLoaded', () => {
    const animalRadios = document.querySelectorAll('input[name="animal"]');
    const animalImage = document.getElementById('animal-image');
    const fileInput = document.getElementById('file-upload');
    const fileInfo = document.getElementById('file-info');

    animalRadios.forEach(radio => {
        radio.addEventListener('change', fetchAnimalImage);
    });

    // Add event listener for file selection
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            fileInfo.innerHTML = `<p>Selected file: ${file.name}</p>`;
        } else {
            fileInfo.innerHTML = '';
        }
    });
});

async function fetchAnimalImage() {
    const animal = document.querySelector('input[name="animal"]:checked').value;
    const animalImage = document.getElementById('animal-image');

    try {
        const response = await fetch(`/check-image/${animal}`);
        const data = await response.json();

        if (data.exists) {
            animalImage.innerHTML = `<img src="/images/${animal}.jpg" alt="${animal}">`;
        } else {
            animalImage.innerHTML = `<p class="error">Image not found for ${animal}</p>`;
        }
    } catch (error) {
        console.error('Error fetching image:', error);
        animalImage.innerHTML = '<p class="error">Error loading image. Please try again.</p>';
    }
}

function uploadFile() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    const fileInfo = document.getElementById('file-info');

    if (!file) {
        fileInfo.innerHTML = '<p class="error">Please select a file.</p>';
        return;
    }

    fileInfo.innerHTML = `<p>Analyzing file: ${file.name}</p>`;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            fileInfo.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            fileInfo.innerHTML = `
                <h3>File Analysis</h3>
                <p><strong>Name:</strong> ${data.name}</p>
                <p><strong>Size:</strong> ${data.size}</p>
                <p><strong>Type:</strong> ${data.type}</p>
                <p><strong>Potential Uses:</strong> ${suggestUses(data.type)}</p>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        fileInfo.innerHTML = '<p class="error">An error occurred while analyzing the file.</p>';
    });
}

function suggestUses(fileType) {
    const suggestions = {
        'application/pdf': 'Document sharing, e-books, printable materials',
        'image/jpeg': 'Web graphics, photo printing, social media sharing',
        'image/png': 'Web graphics, logos, transparent images',
        'video/mp4': 'Video streaming, social media content, presentations',
        'audio/mpeg': 'Music streaming, podcasts, audio books',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'Data analysis, financial reports, inventory management',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Reports, essays, documentation',
        'application/zip': 'File compression, software distribution, backup archives'
    };

    return suggestions[fileType] || 'Various digital content purposes';
}
