document.addEventListener('DOMContentLoaded', function() {
  const toggleSwitch = document.getElementById('highlightToggle');
  const status = document.getElementById('status');
  const colorOptions = document.querySelectorAll('.color-option');

  // Load the current state
  chrome.storage.sync.get(['highlightEnabled', 'highlightColor'], function(data) {
    toggleSwitch.checked = data.highlightEnabled || false;
    updateStatus(toggleSwitch.checked);
    
    const currentColor = data.highlightColor || 'yellow';
    setSelectedColor(currentColor);
  });

  toggleSwitch.addEventListener('change', function() {
    const isEnabled = this.checked;
    chrome.storage.sync.set({highlightEnabled: isEnabled}, function() {
      updateStatus(isEnabled);
      sendMessageToContentScript({action: "updateHighlightState", state: isEnabled});
    });
  });

  colorOptions.forEach(option => {
    option.addEventListener('click', function() {
      const color = this.dataset.color;
      chrome.storage.sync.set({highlightColor: color}, function() {
        setSelectedColor(color);
        sendMessageToContentScript({action: "updateHighlightColor", color: color});
      });
    });
  });

  function updateStatus(isEnabled) {
    status.textContent = isEnabled ? 'Highlighting is on' : 'Highlighting is off';
  }

  function setSelectedColor(color) {
    colorOptions.forEach(option => {
      if (option.dataset.color === color) {
        option.classList.add('selected');
      } else {
        option.classList.remove('selected');
      }
    });
  }

  function sendMessageToContentScript(message) {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, message);
      }
    });
  }
});
