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
let model, processor; // Declare these outside of any function

// Show loading overlay
modelLoadingOverlay.style.display = 'flex';

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

function handleDrop(e) {
    let dt = e.dataTransfer;
    fileUpload.files = dt.files;
    handleFiles(fileUpload.files, model, processor);
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

    modelLoadingOverlay.style.display = 'none'; // Hide loading overlay once model is ready
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
        imageContainer.innerHTML = ''; // Clear previous image
        downloadBtn.classList.add('hidden'); // Hide download button
        processAnotherBtn.classList.add('hidden'); // Hide "process another" button
        status.textContent = ''; // Clear status message
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
    status.textContent = 'Анализ изображения...';
    const image = await RawImage.fromURL(url);
    imageContainer.innerHTML = '';
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
    status.textContent = 'Обработка завершена!';
    enableDownload(canvas);
}

function enableDownload(canvas) {
    downloadBtn.classList.remove('hidden');
    downloadBtn.classList.add('btn', 'btn-primary'); // Tailwind classes for button styling
    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Скачать'; // Font Awesome icon
    processAnotherBtn.classList.remove('hidden'); // Show "process another" button

    downloadBtn.style.display = 'inline';
    downloadBtn.onclick = () => {
        const dataURL = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = 'processed-image.png';
        link.href = dataURL;
        link.click();
    };
}

await setup();
