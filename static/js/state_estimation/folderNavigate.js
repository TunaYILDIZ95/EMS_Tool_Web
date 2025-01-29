document.addEventListener("DOMContentLoaded", function () {
    let dropdown = document.getElementById("id_system");
    let currentFolder = "initial";
    let currentDirectoryDisplay = document.getElementById("current-directory-path");
    let currentFileDisplay = document.getElementById("current-file-path");
    // Attach event listener to the entire document (event delegation)
    document.addEventListener("change", function (event) {
        if (event.target && event.target.id === "id_system") {
            handleDropdownChange(event.target);
        }
    });

    function handleDropdownChange(dropdown) {
        let selectedItem = dropdown.value;
        console.log("Selected item:", selectedItem);

        // Handle "Go Back" functionality
        if (selectedItem === "back") {
            let parentFolder = document.getElementById("current-directory-path").innerText.split("/").slice(0, -1).join("/");
            console.log("Going back to:", parentFolder);
            updateCurrentDirectory(parentFolder);
            updateCurrentFile("");

            // If parentFolder is empty, return to the initial state
            if (!parentFolder || parentFolder === "") {
                loadInitialDropdown();
                console.log("Returning to initial state");
                return;
            }

            fetch(`../load_folder/?folder=${parentFolder}`)
                .then(response => response.json())
                .then(data => updateDropdown(data.items, parentFolder))
                .catch(error => console.error("Error loading parent folder:", error));
            return;
        }

        // Detect if the selected item is a folder
        let selectedOption = dropdown.options[dropdown.selectedIndex];
        let isFolder = selectedOption.getAttribute("data-folder") === "true";

        console.log("Is folder:", isFolder);

        if (isFolder) {
            let folderPath = selectedItem;  // Full path of folder selected
            console.log("Navigating to:", folderPath);
            updateCurrentDirectory(folderPath);

            fetch(`../load_folder/?folder=${folderPath}`)
                .then(response => response.json())
                .then(data => updateDropdown(data.items, folderPath))
                .catch(error => console.error("Error loading folder:", error));
        }
        else {
            updateCurrentFile(selectedItem);
        }
        
    }

    // Function to update dropdown dynamically
    function updateDropdown(items, currentFolder) {
        let dropdown = document.getElementById("id_system");
        dropdown.innerHTML = ""; // Clear current options
        dropdown.setAttribute("data-parent-folder", getParentFolder(currentFolder)); // Store parent folder

        // Add "Go Back" option if not at the initial state
        if (currentFolder !== "initial") {
            let placeHolder = document.createElement("option");
            placeHolder.value = "..";
            placeHolder.innerText = "------------";
            dropdown.appendChild(placeHolder);

            let backOption = document.createElement("option");
            backOption.value = "back";
            backOption.innerText = "⬅️ Go Back";
            dropdown.appendChild(backOption);
        }

        // Add new items (subfolders and files)
        items.forEach(item => {
            let option = document.createElement("option");
            option.value = currentFolder + "/" + item.name;
            option.innerText = item.is_folder ? "📁 " + item.name : "📄 " + item.name;
            option.setAttribute("data-folder", item.is_folder ? "true" : "false");
            dropdown.appendChild(option);
        });

        console.log("Dropdown updated with new options");
    }

    // Function to determine the parent folder path
    function getParentFolder(currentFolder) {
        if (currentFolder === "media/examples" || currentFolder === "initial") {
            return "initial";  // When we reach the main file selection
        }

        let parts = currentFolder.split("/");
        parts.pop();  // Remove the last folder/file name
        return parts.length > 0 ? parts.join("/") : "initial";  // Ensure it doesn't go below root
    }

    // Function to load the initial dropdown state
    function loadInitialDropdown() {
        fetch(``, { headers: { 'X-Requested-With': 'XMLHttpRequest' } })  // AJAX request
            .then(response => response.json())
            .then(data => {
                let dropdown = document.getElementById("id_system");
                dropdown.innerHTML = ""; // Clear options
                let placeHolder = document.createElement("option");
                placeHolder.value = "..";
                placeHolder.innerText = "------------";
                dropdown.appendChild(placeHolder);
                
                // Add user files
                if (data.user_files.length > 0) {
                    let userOptGroup = document.createElement("optgroup");
                    userOptGroup.label = "Your Files";
                    data.user_files.forEach(file => {
                        let option = document.createElement("option");
                        option.value = file.name;
                        option.innerText = "📄 " + file.name;
                        option.setAttribute("data-folder", "false");
                        userOptGroup.appendChild(option);
                    });
                    dropdown.appendChild(userOptGroup);
                }
    
                // Add example files & folders
                if (data.example_files.length > 0) {
                    let exampleOptGroup = document.createElement("optgroup");
                    exampleOptGroup.label = "Example Files";
                    data.example_files.forEach(file => {
                        let option = document.createElement("option");
                        option.value = file.name;
                        option.innerText = file.is_folder ? "📁 " + file.name : "📄 " + file.name;
                        option.setAttribute("data-folder", file.is_folder ? "true" : "false");
                        exampleOptGroup.appendChild(option);
                    });
                    dropdown.appendChild(exampleOptGroup);
                }
            })
            .catch(error => console.error("Error loading initial state:", error));
    }

    function updateCurrentDirectory(folderPath) {
        currentDirectoryDisplay.innerText = folderPath;
    }

    function updateCurrentFile(filePath) {
        currentFileDisplay.innerText = filePath;
    }

});