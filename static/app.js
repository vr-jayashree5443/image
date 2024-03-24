const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
let isRecording = false;
let stream;

navigator.mediaDevices.getUserMedia({ video: true })
    .then((mediaStream) => {
        video.srcObject = mediaStream;
        stream = mediaStream;
        video.play();
    })
    .catch(err => {
        console.error('Error accessing camera:', err);
    });

function startRecording() {
    isRecording = true;
    $('#startButton').prop('disabled', true);
    $('#stopButton').prop('disabled', false);

    const intervalId = setInterval(() => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg');
        $.post('/process', { image: imageData }, (response) => {
            video.src = response;
        });
    }, 1000 / 30); // 30 FPS

    $('#stopButton').click(() => {
        clearInterval(intervalId);
        stopRecording();
    });
}

function stopRecording() {
    isRecording = false;
    $('#startButton').prop('disabled', false);
    $('#stopButton').prop('disabled', true);
    stream.getTracks().forEach(track => track.stop());
}

$('#startButton').click(() => {
    if (!isRecording) {
        startRecording();
    }
});
