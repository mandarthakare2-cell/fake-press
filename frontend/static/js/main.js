
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const status = document.getElementById('status');
const resultBox = document.getElementById('resultBox');

uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert('Select an image first');
    return;
  }
  status.innerText = 'Uploading...';
  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: form
    });
    const data = await res.json();
    if (!res.ok) {
      status.innerText = 'Error: ' + (data.error || res.statusText);
      return;
    }
    status.innerText = 'Done';
    resultBox.innerText = JSON.stringify(data, null, 2);
  } catch (e) {
    status.innerText = 'Request failed: ' + e.message;
  }
});
