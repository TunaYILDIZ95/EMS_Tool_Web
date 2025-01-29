// function getCSRFToken() {
//     let csrfTokenElement = document.querySelector("[name=csrfmiddlewaretoken]");
//     if (csrfTokenElement) {
//         return csrfTokenElement.value;
//     } else {
//         console.error("❌ CSRF token not found in HTML");
//         return null;
//     }
// }

// document.addEventListener("DOMContentLoaded", function () {
//     let runSEButton = document.getElementById("run_SE");
//     let dropdown = document.getElementById("id_system");

//     runSEButton.addEventListener("click", function () {
//         let selectedFile = dropdown.value.trim();

//         if (!selectedFile || selectedFile === "back" || selectedFile === "..") {
//             alert("Please select a valid file before running State Estimation.");
//             return;
//         }

//         let csrfToken = getCSRFToken();
//         if (!csrfToken) {
//             alert("CSRF token is missing. Please refresh the page.");
//             return;
//         }

//         console.log("🔍 Sending POST request with CSRF:", csrfToken);

//         fetch("../run_state_estimation/", {
//             method: "POST",
//             headers: {
//                 "Content-Type": "application/json",
//                 "X-CSRFToken": csrfToken  // Send CSRF token
//             },
//             body: JSON.stringify({ filename: selectedFile })
//         })
//         .then(response => response.json())
//         .then(data => {
//             console.log("✅ SE Output:", data);
//             alert("State Estimation Completed: " + data.message);
//         })
//         .catch(error => console.error("❌ Error running SE:", error));
//     });
// });

document.addEventListener("DOMContentLoaded", function () {
    let runSEButton = document.getElementById("run_SE");
    let dropdown = document.getElementById("id_system");

    runSEButton.addEventListener("click", function () {
        let selectedFile = dropdown.value.trim();

        if (!selectedFile || selectedFile === "back" || selectedFile === "..") {
            alert("Please select a valid file before running State Estimation.");
            return;
        }

        let csrfToken = getCSRFToken();
        if (!csrfToken) {
            alert("CSRF token is missing. Please refresh the page.");
            return;
        }

        console.log("🔍 Sending POST request for State Estimation with CSRF:", csrfToken);

        fetch("../run_state_estimation/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken  // Send CSRF token
            },
            body: JSON.stringify({ filename: selectedFile })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert("Error: " + data.error);
                console.error("❌ SE Error:", data.details);
            } else {
                console.log("✅ SE Output:", data);

                // Call function to update table
                updateSETable(data.table_data);
            }
        })
        .catch(error => console.error("❌ Error running SE:", error));
    });

    function getCSRFToken() {
        let tokenElement = document.querySelector("[name=csrfmiddlewaretoken]");
        return tokenElement ? tokenElement.value : "";
    }

    function updateSETable(tableData) {
        let tableBody = document.getElementById("informationTableContent");
        tableBody.innerHTML = ""; // Clear existing table data

        let busList = tableData.bus_list;
        let voltages = tableData.voltages;
        let angles = tableData.angles;

        for (let i = 0; i < busList.length; i++) {
            let row = document.createElement("tr");

            // ✅ Split bus name and phase
            let busParts = busList[i].split(".");
            let busName = busParts[0];  // Extract Bus Name
            let phase = busParts.length > 1 ? busParts[1] : "N/A";  // Extract Phase or "N/A"

            let col1 = document.createElement("td");
            col1.textContent = i + 1;  // Index
            row.appendChild(col1);

            let col2 = document.createElement("td");
            col2.textContent = busName;  // Bus Name
            row.appendChild(col2);

            let col3 = document.createElement("td");
            col3.textContent = phase;  // Phase
            row.appendChild(col3);

            let col4 = document.createElement("td");
            col4.textContent = voltages[i].toFixed(4);  // Voltage
            row.appendChild(col4);

            let col5 = document.createElement("td");
            col5.textContent = angles[i].toFixed(2);  // Angle
            row.appendChild(col5);

            tableBody.appendChild(row);
        }
    }
});