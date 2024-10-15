chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({highlightEnabled: false, highlightColor: 'yellow'});
});

chrome.action.onClicked.addListener((tab) => {
  chrome.tabs.sendMessage(tab.id, { action: "highlight" });
});
