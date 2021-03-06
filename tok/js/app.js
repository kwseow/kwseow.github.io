// replace these values with those generated in your TokBox Account
var apiKey = "46455592";
var sessionId = "1_MX40NjQ1NTU5Mn5-MTU3MzQ0ODQ2MTQyMX5FdEJoQ0tvZzcxU0NQZktRL1R5TDZCdGJ-fg";
var token = "T1==cGFydG5lcl9pZD00NjQ1NTU5MiZzaWc9YmY2MDA1MmU3Yjg3NTVkMjZjYWUxMTg5NTE1NGUxOGYxMDMxZjc1ZTpzZXNzaW9uX2lkPTFfTVg0ME5qUTFOVFU1TW41LU1UVTNNelEwT0RRMk1UUXlNWDVGZEVKb1EwdHZaemN4VTBOUVprdFJMMVI1VERaQ2RHSi1mZyZjcmVhdGVfdGltZT0xNTczNDcyNzA0Jm5vbmNlPTAuMjk2MDgzNzg1MjE5NTcyNTcmcm9sZT1wdWJsaXNoZXImZXhwaXJlX3RpbWU9MTU3NDA3NzUwNCZpbml0aWFsX2xheW91dF9jbGFzc19saXN0PQ==";

// (optional) add server code here
initializeSession();


// Handling all of our errors here by alerting them
function handleError(error) {
    if (error) {
      alert(error.message);
    }
  }
  
function initializeSession() {
    var session = OT.initSession(apiKey, sessionId);
  
    // Subscribe to a newly created stream
    session.on('streamCreated', function(event) {
        session.subscribe(event.stream, 'subscriber', {
          insertMode: 'append',
          width: '100%',
          height: '100%'
        }, handleError);
      });
      
    // Create a publisher
    var publisher = OT.initPublisher('publisher', {
      insertMode: 'append',
      width: '100%',
      height: '100%'
    }, handleError);
  
    // Connect to the session
    session.connect(token, function(error) {
      // If the connection is successful, publish to the session
      if (error) {
        handleError(error);
      } else {
        session.publish(publisher, handleError);
      }
    });
}
