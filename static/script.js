
function selectMood(mood) {
    document.getElementById("mood").value = mood;
    document.getElementById("form-error").textContent = "";
}

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[character]);
}

function makeTags(items, className) {
    if (!Array.isArray(items)) return "";

    return items.map(item =>
        `<span class="${className}-tag">${escapeHTML(item)}</span>`
    ).join("");
}

async function getRecommendations() {
    const genre = document.getElementById("genre").value;
    const mood = document.getElementById("mood").value;
    const errorBox = document.getElementById("form-error");
    const loading = document.getElementById("loading");
    const grid = document.getElementById("movie-grid");
    const button = document.getElementById("recommend-button");
    const info = document.getElementById("result-info");

    errorBox.textContent = "";

    if (!genre || !mood) {
        errorBox.textContent = "Please select both a genre and a mood.";
        return;
    }

    loading.classList.remove("hidden");
    button.disabled = true;
    grid.innerHTML = "";
    info.textContent = "Finding recommendations...";

    try {
        const response = await fetch("/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ genre, mood })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Could not get recommendations.");
        }

        displayMovies(data.movies || [], data.genre, data.mood);

    } catch (error) {
        grid.innerHTML = `
            <div class="empty-state error-state">
                <div class="empty-icon">⚠️</div>
                <h3>We couldn't load your movies</h3>
                <p>${escapeHTML(error.message)}</p>
                <p>Check that Flask is running and try again.</p>
            </div>
        `;
        info.textContent = "Please try again.";
    } finally {
        loading.classList.add("hidden");
        button.disabled = false;
    }
}

function displayMovies(movies, genre, mood) {
    const grid = document.getElementById("movie-grid");
    const info = document.getElementById("result-info");

    grid.innerHTML = "";

    info.textContent =
        `${movies.length} recommendations for ${genre} + ${mood}`;

    if (!movies.length) {
        grid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🎬</div>
                <h3>No movies found</h3>
                <p>Try another genre or mood.</p>
            </div>
        `;
        return;
    }

    movies.forEach((movie, index) => {
        const title = escapeHTML(movie.title);
        const poster = typeof movie.poster === "string" ? movie.poster : "";
        const card = document.createElement("article");
        card.className = "movie-card";

        const posterHTML = poster
            ? `<img class="movie-poster" src="${escapeHTML(poster)}"
                    alt="Poster for ${title}" loading="lazy">`
            : `<div class="poster-placeholder">
                    <span>🎬</span>
                    <small>Poster unavailable</small>
               </div>`;

        card.innerHTML = `
            <div class="poster-container">
                ${posterHTML}
                <div class="rank-badge">#${index + 1}</div>
                <div class="score-badge">${Number(movie.score).toFixed(1)}%</div>
            </div>
            <div class="movie-info">
                <h3>${title}</h3>
                <div class="tags genre-tags">
                    ${String(movie.genres || "").split("|")
                        .filter(Boolean)
                        .map(g => `<span>${escapeHTML(g)}</span>`).join("")}
                </div>
                <div class="tags mood-tags">
                    ${makeTags(movie.moods, "mood")}
                </div>
                <div class="match-text">✦ ${Number(movie.score).toFixed(1)}% match score</div>
            </div>
        `;

        const img = card.querySelector(".movie-poster");
        if (img) {
            img.addEventListener("error", () => {
                const placeholder = document.createElement("div");
                placeholder.className = "poster-placeholder";
                placeholder.innerHTML =
                    "<span>🎬</span><small>Poster unavailable</small>";
                img.replaceWith(placeholder);
            }, { once: true });
        }

        grid.appendChild(card);
    });
}

document.addEventListener("keydown", event => {
    if (
        event.key === "Enter" &&
        ["genre", "mood"].includes(document.activeElement?.id)
    ) {
        getRecommendations();
    }
});
