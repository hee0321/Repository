const carCanvas = document.getElementById("carCanvas");
carCanvas.width = window.innerWidth * 0.7; // Takes majority of space

const networkCanvas = document.getElementById("networkCanvas");
networkCanvas.width = 300; // side panel width

const carCtx = carCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");

const road = new Road(carCanvas.width / 2, carCanvas.width * 0.9);

// Parameters
const N = 100; // Number of cars per generation
let cars = generateCars(N);
let bestCar = cars[0];

// Restore best brain from local storage if available
let generationCount = localStorage.getItem("genCount") ? parseInt(localStorage.getItem("genCount")) : 1;
document.getElementById("gen-count").innerText = generationCount;

if (localStorage.getItem("bestBrain")) {
    for (let i = 0; i < cars.length; i++) {
        cars[i].brain = JSON.parse(localStorage.getItem("bestBrain"));
        if (i != 0) {
            // Mutate all cars except the absolute best (preserves the champion)
            NeuralNetwork.mutate(cars[i].brain, 0.1); // 10% mutation rate
        }
    }
}

// Generate static traffic obstacles
const traffic = [
    new Car(road.getLaneCenter(1), -100, 30, 50, "DUMMY", 2),
    new Car(road.getLaneCenter(0), -300, 30, 50, "DUMMY", 2),
    new Car(road.getLaneCenter(2), -300, 30, 50, "DUMMY", 2),
    new Car(road.getLaneCenter(0), -500, 30, 50, "DUMMY", 2),
    new Car(road.getLaneCenter(1), -500, 30, 50, "DUMMY", 2),
    new Car(road.getLaneCenter(1), -700, 30, 50, "DUMMY", 2),
    new Car(road.getLaneCenter(2), -700, 30, 50, "DUMMY", 2),
];

animate();

function save() {
    localStorage.setItem("bestBrain", JSON.stringify(bestCar.brain));
    // Also save Generation count
    localStorage.setItem("genCount", generationCount + 1);
}

function discard() {
    localStorage.removeItem("bestBrain");
    localStorage.removeItem("genCount");
}

function generateCars(N) {
    const cars = [];
    for (let i = 1; i <= N; i++) {
        cars.push(new Car(road.getLaneCenter(1), 100, 30, 50, "AI"));
    }
    return cars;
}

// Update game loop
function animate(time) {
    // Move traffic
    for (let i = 0; i < traffic.length; i++) {
        traffic[i].update(road.borders, []); // Empty array so traffic don't hit each other in this simple sim
    }

    // Move AI Cars
    for (let i = 0; i < cars.length; i++) {
        cars[i].update(road.borders, traffic);
    }

    // Find the best car (The one that moved lowest on the Y axis / furthest forward)
    bestCar = cars.find(
        c => c.y == Math.min(...cars.map(c => c.y))
    );

    updateUI();

    // Setup Car canvas context for 'camera' follow effect
    carCanvas.height = window.innerHeight; // Fill height dynamically
    networkCanvas.height = window.innerHeight * 0.5;

    carCtx.save();
    // Translate canvas so the best car is always centered at bottom-ish
    carCtx.translate(0, -bestCar.y + carCanvas.height * 0.7);

    // Draw Road
    road.draw(carCtx);

    // Draw Traffic
    for (let i = 0; i < traffic.length; i++) {
        traffic[i].draw(carCtx);
    }

    // Draw Cars. Fade out background cars, emphasize Best Car.
    carCtx.globalAlpha = 0.2;
    for (let i = 0; i < cars.length; i++) {
        cars[i].draw(carCtx, false); // Don't draw sensors for all
    }

    // Best Car
    carCtx.globalAlpha = 1;
    bestCar.draw(carCtx, true); // Draw sensor for best car

    carCtx.restore(); // Stop camera follow

    // Draw Neural Network of Best Car
    networkCtx.lineDashOffset = -time / 50; // Animates marching ants effect
    Visualizer.drawNetwork(networkCtx, bestCar.brain);

    requestAnimationFrame(animate);
}

// Interactivity UI logic
function updateUI() {
    document.getElementById("alive-count").innerText = cars.filter(c => !c.damaged).length;
    document.getElementById("total-count").innerText = cars.length;
    document.getElementById("best-fitness").innerText = Math.round(bestCar.fitness);

    // If all cars crash, automatically start new generation
    if (cars.every(c => c.damaged)) {
        save();
        location.reload();
    }
}

// Button Events
document.getElementById("save-btn").addEventListener("click", () => {
    save();
    alert("Brain saved!");
});
document.getElementById("discard-btn").addEventListener("click", () => {
    discard();
    alert("Brain deleted. It will be completely random on next load.");
});
document.getElementById("reset-btn").addEventListener("click", () => {
    discard();
    location.reload();
});
document.getElementById("next-gen-btn").addEventListener("click", () => {
    save();
    location.reload();
});
