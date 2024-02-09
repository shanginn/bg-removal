import './style.css';
import { AutoModel, AutoProcessor, env, RawImage } from '@xenova/transformers';

// Adjusting environment settings for model loading
env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = true;

// Reference the necessary DOM elements
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const downloadBtn = document.getElementById('download-btn');
const processAnotherBtn = document.getElementById('process-another-btn');
const modelLoadingOverlay = document.getElementById('model-loading-overlay');
const dropArea = document.getElementById('drop-area');
const uploadButton = document.getElementById('upload-button');
let model, processor; // Declare these outside of any function

// Show loading overlay
// modelLoadingOverlay.style.display = 'flex';
toggleOverlay(true, 'Загрузка инструментов, пожалуйста подождите...');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.classList.add('bg-gray-200'); // Tailwind class for background color
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.add('border-blue-500'), false); // Tailwind classes for border color
});

// Unhighlight drop area when item is dragged out of it or dropped
['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.remove('border-blue-500'), false); // Tailwind classes for border color
});

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

async function handleDrop(e) {
    let dt = e.dataTransfer;
    fileUpload.files = dt.files;
    await handleFiles(fileUpload.files, model, processor);
}

async function loadModelAndProcessor() {
    const model = await AutoModel.from_pretrained('briaai/RMBG-1.4', {
        config: { model_type: 'custom' },
    });

    const processor = await AutoProcessor.from_pretrained('briaai/RMBG-1.4', {
        config: {
            do_normalize: true,
            do_pad: false,
            do_rescale: true,
            do_resize: true,
            image_mean: [0.5, 0.5, 0.5],
            feature_extractor_type: "ImageFeatureExtractor",
            image_std: [1, 1, 1],
            resample: 2,
            rescale_factor: 0.00392156862745098,
            size: { width: 1024, height: 1024 },
        }
    });

    // modelLoadingOverlay.style.display = 'none'; // Hide loading overlay once model is ready
    toggleOverlay(false);
    status.textContent = 'Готово к работе';
    return { model, processor };
}

async function setup() {
    const loadedModels = await loadModelAndProcessor();
    model = loadedModels.model;
    processor = loadedModels.processor;

    // Pass model and processor to the event listener callback
    fileUpload.addEventListener('change', async (e) => {
        await handleFiles(e.target.files, model, processor); // Pass model and processor here
    });

    // Add event listener for processAnotherBtn
    processAnotherBtn.addEventListener('click', () => {
        fileUpload.value = ''; // Reset file input
        downloadBtn.classList.add('hidden'); // Hide download button
        processAnotherBtn.classList.add('hidden'); // Hide "process another" button
        uploadButton.classList.remove('hidden'); // Show upload button
        status.textContent = ''; // Clear status message

        // Clear previous image
        imageContainer.style.removeProperty('max-width');
        imageContainer.style.removeProperty('max-height');
        imageContainer.style.removeProperty('width');
        imageContainer.style.removeProperty('height');
        imageContainer.style.removeProperty('background');

        const canvas = imageContainer.querySelector('canvas');
        if (canvas) {
            canvas.remove();
        }
    });
}

async function handleFiles(files, model, processor) {
    const file = files[0];
    if (!file) {
        return;
    }

    status.textContent = 'Загрузка изображения...';
    const reader = new FileReader();
    reader.onload = async (e) => {
        // Pass model and processor to the predict function
        await predict(e.target.result, model, processor);
    };
    reader.readAsDataURL(file);
}

async function predict(url, model, processor) {
    toggleOverlay(true, 'Удаляем фон...');
    const image = await RawImage.fromURL(url);
    uploadButton.classList.add('hidden'); // Hide upload button

    imageContainer.style.backgroundImage = `url(${url})`;

    const ar = image.width / image.height;
    const [cw, ch] = (ar > 720 / 480) ? [720, 720 / ar] : [480 * ar, 480];
    imageContainer.style.maxWidth = `${cw}px`;
    imageContainer.style.maxHeight = `${ch}px`;
    imageContainer.style.width = `${cw}px`;
    imageContainer.style.height = `${ch}px`;

    // Assuming the processor object has a method named 'process' or similar for processing the image
    // This line needs to be adjusted according to the actual API
    const processedImage = await processor(image); // Correct method to preprocess the image
    const { pixel_values } = processedImage; // Adjust according to the actual structure of the processed image object

    const { output } = await model({ input: pixel_values });

    const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image.toCanvas(), 0, 0);

    const pixelData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < mask.data.length; ++i) {
        pixelData.data[4 * i + 3] = mask.data[i];
    }
    ctx.putImageData(pixelData, 0, 0);

    imageContainer.append(canvas);
    imageContainer.style.removeProperty('background-image');
    imageContainer.style.background = `url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAGUExURb+/v////5nD/3QAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAUSURBVBjTYwABQSCglEENMxgYGAAynwRB8BEAgQAAAABJRU5ErkJggg==")`;
    status.textContent = 'Обработка завершена!';
    toggleOverlay(false);
    enableDownload(canvas);
}
function enableDownload(canvas) {
    downloadBtn.classList.remove('hidden');
    downloadBtn.classList.add('btn', 'btn-primary'); // Tailwind classes for button styling
    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Скачать'; // Font Awesome icon
    processAnotherBtn.classList.remove('hidden'); // Show "process another" button

    downloadBtn.onclick = async () => {
        setDownloadButtonState(true); // Set the button to loading state

        const dataURL = canvas.toDataURL('image/png');
        const blob = dataURLToBlob(dataURL); // Convert data URL to blob
        const objectURL = URL.createObjectURL(blob);

        const link = document.createElement('a');
        const now = new Date().toISOString().replace(/[:.]/g, '-');
        link.download = `no-bg-${now}.png`;
        link.href = objectURL;
        document.body.appendChild(link); // Append to body to ensure visibility in the DOM on mobile

        link.click(); // Trigger the download

        // Cleanup
        document.body.removeChild(link);
        URL.revokeObjectURL(objectURL);

        setDownloadButtonState(false); // Reset the button to normal state after operation
    };
}

// Function to toggle overlay visibility
function toggleOverlay(display = true, message = 'Анализ изображения...') {
    if (display) {
        modelLoadingOverlay.innerHTML = `<div class="text-white flex flex-col items-center"><i class="fas fa-spinner fa-spin fa-3x"></i><p>${message}</p></div>`; // Update message dynamically
        modelLoadingOverlay.style.display = 'flex';
    } else {
        modelLoadingOverlay.style.display = 'none';
    }
}

function setDownloadButtonState(isLoading) {
    if (isLoading) {
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Подготовка...'; // Show loading spinner and text
        downloadBtn.disabled = true; // Disable the button to prevent multiple clicks
    } else {
        downloadBtn.innerHTML = '<i class="fas fa-download"></i> Скачать'; // Reset to original text
        downloadBtn.disabled = false; // Enable the button
    }
}

function dataURLToBlob(dataURL) {
    const parts = dataURL.split(';base64,');
    const contentType = parts[0].split(':')[1];
    const raw = window.atob(parts[1]);
    const rawLength = raw.length;
    const uInt8Array = new Uint8Array(rawLength);

    for (let i = 0; i < rawLength; ++i) {
        uInt8Array[i] = raw.charCodeAt(i);
    }

    return new Blob([uInt8Array], { type: contentType });
}

await setup();