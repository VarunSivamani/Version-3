let highlightEnabled = false;
let highlightColor = 'yellow';

chrome.storage.sync.get(['highlightEnabled', 'highlightColor'], function(data) {
  highlightEnabled = data.highlightEnabled || false;
  highlightColor = data.highlightColor || 'yellow';
});

document.addEventListener('mouseup', function() {
  if (highlightEnabled) {
    highlightSelection();
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   console.log('Received message:', request); // Debugging line
  if (request.action === "updateHighlightState") {
    highlightEnabled = request.state;
  } else if (request.action === "updateHighlightColor") {
    highlightColor = request.color;
    // console.log('New highlight color:', highlightColor); // Debugging line
    updateExistingHighlights(highlightColor);
  }
});

function highlightSelection() {
  const selection = window.getSelection();
  if (selection.rangeCount > 0) {
    const range = selection.getRangeAt(0);
    const span = document.createElement('span');
    span.style.backgroundColor = highlightColor;
    span.style.color = 'black';
    span.className = 'extension-highlight';
    range.surroundContents(span);
  }
}

function updateExistingHighlights(color) {
  const highlights = document.querySelectorAll('.extension-highlight');
  highlights.forEach(highlight => {
    highlight.style.backgroundColor = color;
  });
}
