document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('waveform');
    const ctx = canvas.getContext('2d');
    const synthesizeBtn = document.getElementById('synthesize-btn');
    const voiceItems = document.querySelectorAll('.voice-item');

    // Set canvas resolution
    function resizeCanvas() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Waveform state
    let animationId;
    let isPlaying = false;
    let time = 0;

    // Pulse Animation (ECG style)
    function drawWave() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#ff2d55';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#ff2d55';

        const centerY = canvas.height / 2;
        const width = canvas.width;

        for (let x = 0; x < width; x++) {
            let y = centerY;

            if (isPlaying) {
                // Base static
                let noise = Math.sin(x * 0.05 + time) * 2;

                // The Heartbeat Pulse (P-QRS-T complex simulation)
                const pulsePeriod = 200;
                const phase = (x + time * 100) % pulsePeriod;

                if (phase > 40 && phase < 45) { // P wave
                    y -= Math.sin((phase - 40) / 5 * Math.PI) * 10;
                } else if (phase > 55 && phase < 65) { // QRS complex
                    const qrs = (phase - 55) / 10;
                    if (qrs < 0.2) y += qrs * 50;
                    else if (qrs < 0.5) y -= (qrs - 0.2) * 200;
                    else y += (qrs - 0.5) * 100;
                } else if (phase > 90 && phase < 120) { // T wave
                    y -= Math.sin((phase - 90) / 30 * Math.PI) * 15;
                }

                y += noise;
            }

            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.stroke();

        if (isPlaying) {
            time += 0.05;
        }

        animationId = requestAnimationFrame(drawWave);
    }

    drawWave();

    // Interaction
    synthesizeBtn.addEventListener('click', () => {
        if (!isPlaying) {
            isPlaying = true;
            synthesizeBtn.querySelector('.btn-text').textContent = 'PULSING...';
            synthesizeBtn.classList.add('active');

            // Simulate synthesis time
            setTimeout(() => {
                isPlaying = false;
                synthesizeBtn.querySelector('.btn-text').textContent = 'SYNTHESIZE';
                synthesizeBtn.classList.remove('active');
            }, 3000);
        }
    });

    voiceItems.forEach(item => {
        item.addEventListener('click', () => {
            voiceItems.forEach(v => v.classList.remove('active'));
            item.classList.add('active');
        });
    });

    // Range input value updates
    const sliders = document.querySelectorAll('.styled-slider');
    sliders.forEach(slider => {
        slider.addEventListener('input', (e) => {
            const span = e.target.parentElement.querySelector('.value');
            span.textContent = `${e.target.value}x`;
        });
    });
});
