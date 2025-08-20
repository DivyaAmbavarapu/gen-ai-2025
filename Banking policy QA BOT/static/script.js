async function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    if (!userInput.trim()) return;

    // Show user message
    const chatBox = document.getElementById("chatBox");
    chatBox.innerHTML += `<div class="user">You: ${userInput}</div>`;

    // Send to backend
    const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    });

    const data = await response.json();

    // Show chatbot reply
    chatBox.innerHTML += `<div class="bot">Bot: ${data.response}</div>`;

    // Clear input
    document.getElementById("userInput").value = "";
}
