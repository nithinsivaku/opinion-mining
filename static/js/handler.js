var process_input = function() {
    console.log("handle");
    var url = '/';
    var data = document.getElementById("text_input").value;
    $.post(url, data, function(res) {
        document.getElementById("text_input").value = res.pred;
    });
}

var clear_input = function() {
    console.log("reset");
    document.getElementById("text_input").value = "";
}