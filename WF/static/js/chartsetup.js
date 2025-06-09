document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error('Canvas element not found.');
        return;
    }

    const ctx = chartElement.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 100);
    gradient.addColorStop(0, 'rgba(250, 0, 0, 1)');
    gradient.addColorStop(1, 'rgba(136, 255, 0, 1)');

    const forecastItems = document.querySelectorAll('.forecast-item');
    const temps = [];
    const times = [];

    forecastItems.forEach(item => {
        const time = item.querySelector('.forecast-time')?.textContent;
        const tempText = item.querySelector('.forecast-temperaturevalue')?.textContent;
        const temp = parseFloat(tempText?.replace('°', '')); // Remove ° if present

        if (time && !isNaN(temp)) {
            times.push(time);
            temps.push(temp);
        }
    });

    if (temps.length === 0 || times.length === 0) {
        console.error('Temperature or time values missing.');
        return;
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [{
                label: 'Temperature (°C)',
                data: temps,
                borderColor: gradient,
                backgroundColor: 'rgba(136, 255, 0, 0.2)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 3,
            }],
        },
        options: {
            plugins: {
                legend: {
                    display: true,
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time',
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Temperature (°C)',
                    }
                },
            },
            animation: {
                duration: 750,
            },
        },
    });
});
