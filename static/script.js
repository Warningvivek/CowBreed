function handleFile(input) {
    const file = input.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);

    document.getElementById('previewThumb').src = url;
    document.getElementById('previewImage').src = url;
    document.getElementById('previewName').textContent = file.name;

    document.getElementById('dropIdle').style.display = 'none';
    document.getElementById('dropReady').style.display = 'flex';

    document.getElementById('detectBtn').disabled = false;
}

function uploadImage() {
    const file = document.getElementById('fileInput').files[0];

    if (!file) {
        alert("Upload image first");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById('loadingState').style.display = 'flex';
    document.getElementById('resultSection').style.display = 'none';

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('loadingState').style.display = 'none';

        document.getElementById("breed").textContent = data.breed;
        document.getElementById("confidence").textContent = data.confidence + "%";
        document.getElementById("confFill").style.width = data.confidence + "%";
        document.getElementById("origin").textContent = data.origin;
        document.getElementById("description").textContent = data.description;

        document.getElementById("others").innerHTML =
            data.others.map(o => `
                <div class="other-row">
                    <span class="other-name">${o.breed}</span>
                    <span class="other-bar-wrap">
                        <span class="other-bar" style="width:${o.conf}%"></span>
                    </span>
                    <span class="other-pct">${o.conf}%</span>
                </div>
            `).join("");

        document.getElementById('resultSection').style.display = 'block';
    })
    .catch(err => {
        console.error(err);
        alert("Error detecting breed");
    });
}
