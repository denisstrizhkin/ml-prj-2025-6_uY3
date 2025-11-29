// Обработка формы загрузки данных
document.getElementById('uploadForm')?.addEventListener('submit', function() {
    const button = document.getElementById('uploadButton');
    const spinner = document.getElementById('uploadLoading');

    button.disabled = true;
    button.textContent = 'Загрузка...';
    spinner.style.display = 'block';
});

// Обработка формы обновления точек коллокации
document.getElementById('updateCollocationForm')?.addEventListener('submit', function() {
    const button = document.getElementById('updateCollocationButton');
    const spinner = document.getElementById('updateLoading');

    button.disabled = true;
    button.textContent = 'Обновление...';
    spinner.style.display = 'block';
});

// Обработка формы запуска ML модели
document.getElementById('mlForm')?.addEventListener('submit', function() {
    const button = document.getElementById('mlButton');
    const spinner = document.getElementById('mlLoading');

    button.disabled = true;
    button.textContent = 'Выполняется...';
    spinner.style.display = 'block';
});

// Обработка формы предсказания
document.getElementById('predictForm')?.addEventListener('submit', function() {
    const button = document.getElementById('predictButton');
    const spinner = document.getElementById('predictLoading');

    button.disabled = true;
    button.textContent = 'Построение...';
    spinner.style.display = 'block';
});

// Валидация параметров ML
document.getElementById('num_layers')?.addEventListener('change', function() {
    const value = parseInt(this.value);
    if (value < 1) this.value = 1;
    if (value > 50) this.value = 50;
});

document.getElementById('num_perceptrons')?.addEventListener('change', function() {
    const value = parseInt(this.value);
    if (value < 1) this.value = 1;
    if (value > 200) this.value = 200;
});

document.getElementById('num_epoch')?.addEventListener('change', function() {
    const value = parseInt(this.value);
    if (value < 100) this.value = 100;
    if (value > 100000) this.value = 100000;
});

// Валидация параметров предсказания
document.getElementById('num_x_points')?.addEventListener('change', function() {
    const value = parseInt(this.value);
    if (value < 10) this.value = 10;
    if (value > 1000) this.value = 1000;
});

document.getElementById('prediction_time')?.addEventListener('change', function() {
    const value = parseFloat(this.value);
    if (value < 0) this.value = 0;
});

// Пример конфигурации весов ошибок
document.getElementById('loss_weights_config')?.addEventListener('focus', function() {
    if (!this.value) {
        this.value = '0:10.0,1.0; 1000:3.0,1.0; 5000:1.0,1.0';
    }
});

// Автоматическая прокрутка к результатам после загрузки
window.addEventListener('load', function() {
    // Эта часть будет работать только если есть предсказания
    const predictionCard = document.querySelector('.card.bg-light');
    if (predictionCard) {
        setTimeout(() => {
            predictionCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
});