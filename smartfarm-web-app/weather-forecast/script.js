const inputBox = document.querySelector('.input-box');
const searchBtn = document.getElementById('searchBtn');
const weather_img = document.querySelector('.weather-img');
const temperature = document.querySelector('.temperature');
const description = document.querySelector('.description');
const humidity = document.getElementById('humidity');
const wind_speed = document.getElementById('wind-speed');

const location_not_found = document.querySelector('.location-not-found');
const weather_body = document.querySelector('.weather-body');

async function checkWeather(city){
    const api_key = "4c4286de4f6a3794841e570fd8bc4a0b";
    const url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${api_key}`;

    const weather_data = await fetch(url).then(response => response.json());

    if(weather_data.cod === "404"){
        location_not_found.style.display = "block";
        weather_body.style.display = "none";
        return;
    }

    location_not_found.style.display = "none";
    weather_body.style.display = "flex";

    temperature.innerHTML = `${Math.round(weather_data.main.temp - 273.15)}°C`;
    description.innerHTML = `${weather_data.weather[0].description}`;
    humidity.innerHTML = `${weather_data.main.humidity}%`;
    wind_speed.innerHTML = `${weather_data.wind.speed} Km/H`;

    const main = weather_data.weather[0].main;


   switch (main.toLowerCase()) {
    case 'clouds':
        weather_img.src = "img/cloud.png";
        break;
    case 'clear':
        weather_img.src = "img/clear-sky.png";
        break;
    case 'rain':
    case 'drizzle':
        weather_img.src = "img/rain.png";
        break;
    case 'mist':
    case 'haze':
    case 'fog':
    case 'smoke':
        weather_img.src = "img/mist.png";
        break;
    case 'snow':
        weather_img.src = "img/snow.png";
        break;
    case 'thunderstorm':
        weather_img.src = "img/thunderstorm.png";
        break;
    default:
        weather_img.src = "img/default.png";
}


}

searchBtn.addEventListener('click', () => {
    const city = inputBox.value.trim();
    if(city) {
        checkWeather(city);
    }
});
