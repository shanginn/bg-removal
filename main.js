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

// Initial UI setup
status.textContent = 'Модель загружается...';

// Asynchronously load the model and processor
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

    status.textContent = 'Готово к работе';
    return { model, processor };
}

const { model, processor } = await loadModelAndProcessor();

fileUpload.addEventListener('change', async function (e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }

    status.textContent = 'Загрузка изображения...';
    const reader = new FileReader();
    reader.onload = async e2 => {
        await predict(e2.target.result);
    };
    reader.readAsDataURL(file);
});

async function predict(url) {
    status.textContent = 'Анализ изображения...';
    const image = await RawImage.fromURL(url);
    imageContainer.innerHTML = '';
    imageContainer.style.backgroundImage = `url(${url})`;

    const ar = image.width / image.height;
    const [cw, ch] = (ar > 720 / 480) ? [720, 720 / ar] : [480 * ar, 480];
    imageContainer.style.width = `${cw}px`;
    imageContainer.style.height = `${ch}px`;

    const { pixel_values } = await processor(image);
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

    // Make the download button visible and functional
    downloadBtn.style.display = 'inline';
    downloadBtn.onclick = function () {
        const dataURL = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = 'processed-image.png';
        link.href = dataURL;
        link.click();
    };
}
