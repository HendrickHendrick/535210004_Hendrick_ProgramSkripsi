// Function to redirect to the home page when refreshing
window.onload = function() {
    const performanceEntries = performance.getEntriesByType("navigation");
    if (performanceEntries.length > 0) {
        const navigationType = performanceEntries[0].type;
        if (navigationType === "reload") {
            window.location.href = window.location.pathname + window.location.search + window.location.hash
        }
    } 
};




