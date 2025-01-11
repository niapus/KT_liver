let originalCanvas, overlayCanvas, maskCanvas;
let originalCtx, overlayCtx, maskCtx;
let isMouseDown = false;
let isDrawing = false;
let isErasing = false;
let lastX = 0, lastY = 0;
let defaultMask = null;
let lineThickness = 2;
let isProcessing = false;
let currentCanvasRect = null;


document.addEventListener("DOMContentLoaded", () => {
    // Инициализация холстов и контекстов
    initializeCanvas();
    initializeEvents();
});

// Инициализация холстов
function initializeCanvas() {
    originalCanvas = document.getElementById("originalCanvas");
    overlayCanvas = document.getElementById("overlayCanvas");
    maskCanvas = document.getElementById("maskCanvas");

    originalCtx = originalCanvas.getContext("2d");
    overlayCtx = overlayCanvas.getContext("2d");
    maskCtx = maskCanvas.getContext("2d");

    resizeCanvas();
}

function resizeCanvas() {
    [originalCanvas, overlayCanvas, maskCanvas].forEach(canvas => {
        const parentElement = canvas.parentElement;
        const parentWidth = parentElement.offsetWidth;

        canvas.width = parentWidth;
        canvas.height = parentWidth;
    });
  currentCanvasRect = overlayCanvas.getBoundingClientRect(); // Пересчитываем текущий прямоугольник
  updateOverlay();
}

// Инициализация событий для рисования
function initializeEvents() {
    overlayCanvas.addEventListener("mousedown", startDrawing);
    overlayCanvas.addEventListener("mousemove", draw);
    overlayCanvas.addEventListener("mouseup", stopDrawing);
    overlayCanvas.addEventListener("mouseout", stopDrawing); // Остановка рисования, если мышь выходит за пределы

    const drawButton = document.getElementById("drawMode");
    const eraseButton = document.getElementById("eraseMode");
    const clearButton = document.getElementById("clearPred");
    const thicknessSelect = document.getElementById("lineThickness");

    clearButton.addEventListener("click", clearPrediction);

    drawButton.addEventListener("click", () => {
        isDrawing = true;
        isErasing = false;
    });

    eraseButton.addEventListener("click", () => {
        isDrawing = false;
        isErasing = true;
    });

    thicknessSelect.addEventListener("change", (e) => {
        lineThickness = parseInt(e.target.value);
    })

    window.addEventListener('resize', handleResize);
}

// Начало рисования
function startDrawing(e) {
    currentCanvasRect = overlayCanvas.getBoundingClientRect(); // Пересчитываем текущий прямоугольник
    const offsetX = e.clientX - currentCanvasRect.left;
    const offsetY = e.clientY - currentCanvasRect.top;
    [lastX, lastY] = [offsetX, offsetY];
    isMouseDown = true;
}

// Рисование на центральном холсте
function draw(e) {
    if (!isMouseDown) return;

    const { offsetX, offsetY } = e;

    maskCtx.beginPath();
    maskCtx.moveTo(lastX, lastY);
    maskCtx.lineTo(offsetX, offsetY);

    if (isErasing) {
        maskCtx.lineWidth = lineThickness;
        maskCtx.globalCompositeOperation = "source-over"; // Удаление пикселей
        maskCtx.strokeStyle = "black";
        maskCtx.stroke();
    } else if (isDrawing) {
        maskCtx.lineWidth = lineThickness;
        maskCtx.globalCompositeOperation = "source-over"; // Рисование
        maskCtx.strokeStyle = "yellow";
        maskCtx.stroke();
    }

    [lastX, lastY] = [offsetX, offsetY];

    // Обновляем центральный холст
    updateOverlay();
}

// Остановка рисования
function stopDrawing() {
    isMouseDown = false;
}

function handleResize(){
    resizeCanvas();
}


// Отправка файла и получение результата
async function uploadFile() {

    if (isProcessing) {
        alert("Please wait for the current operation is completed.");
        return;
    }

    isProcessing = true;

    const dicomFile = document.getElementById("dicomFile").files[0];
    if (!dicomFile) {
        alert("Please select a file.");
        isProcessing = false;
        return;
    }

    const formData = new FormData();
    formData.append("file", dicomFile);

    const response = await fetch("/predict_liver/", {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        const errorData = await response.json(); // Получаем сообщение об ошибке
        alert(errorData.error || "An error occurred while processing the file.");
        return; // Останавливаем выполнение функции
    }

    const maskBlob = await response.blob();

    const response1 = await fetch("/get_image", {
        method: "GET",
    });

    if (!response1.ok) {
        alert("Failed to retrieve the original image.");
        return;
    }

    const dicomBlob = await response1.blob();
    
    const dicomImg = await loadImage(URL.createObjectURL(dicomBlob));
    const maskImg = await loadImage(URL.createObjectURL(maskBlob));

    defaultMask = maskImg;
    
    // Отрисовка на холстах
    drawOnCanvas(originalCtx, dicomImg, originalCanvas);
    drawOnCanvas(maskCtx, maskImg, maskCanvas);

    updateOverlay();

    isProcessing = false;
}

// Обновление центрального холста с наложенной маской
function updateOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Рисуем оригинальное изображение
    overlayCtx.drawImage(originalCanvas, 0, 0);

    // Рисуем маску поверх
    overlayCtx.globalAlpha = 0.4; // Полупрозрачность маски
    overlayCtx.drawImage(maskCanvas, 0, 0);
    overlayCtx.globalAlpha = 1.0; // Восстанавливаем прозрачность
}

function clearPrediction() {
    if (defaultMask) {
        // Восстанавливаем оригинальное состояние маски
        drawOnCanvas(maskCtx, defaultMask, maskCanvas);
        updateOverlay();
    }
}

// Сохранение отредактированной маски
async function saveEditedMask() {
    const maskData = maskCanvas.toDataURL("image/png");
    const formData = new FormData();
    formData.append("file", dataURItoBlob(maskData));

    const response = await fetch("/save_edited_mask/", {
        method: "POST",
        body: formData
    });

    const result = await response.json();
    alert(result.message);
}

// Конвертация DataURL в Blob
function dataURItoBlob(dataURI) {
    const byteString = atob(dataURI.split(',')[1]);
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uintArray = new Uint8Array(arrayBuffer);

    for (let i = 0; i < byteString.length; i++) {
        uintArray[i] = byteString.charCodeAt(i);
    }

    return new Blob([uintArray], { type: "image/png" });
}

// Загрузка изображения
function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

// Отрисовка изображения на заданный холст с возможностью увеличения яркости
async function drawOnCanvas(context, img, canvas, brightnessFactor = 1) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Извлечение данных пикселей
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Увеличение яркости
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.min(255, data[i] * brightnessFactor);     // Красный
        data[i + 1] = Math.min(255, data[i + 1] * brightnessFactor); // Зеленый
        data[i + 2] = Math.min(255, data[i + 2] * brightnessFactor); // Синий
    }

    // Обновление данных на холсте
    context.putImageData(imageData, 0, 0);
}
