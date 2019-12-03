/**
 * Send input text to flask server
 * Display the result in Modal
 */
var process_input = function() {
    var content = document.getElementById("text_input").value;
    var data = JSON.stringify({ 'story': content });
    var story = { "content": content };
    $.ajax({
        type: "POST",
        contentType: "application/json;charset=utf-8",
        url: "/",
        traditional: "true",
        data: JSON.stringify(story),
        dataType: "json",
        success: function(res) {
            bgColor = res.pred == "Positive Content" ? "#89a15d" : "#912f2f";
            document.querySelector('.modal-header').style.backgroundColor = bgColor;
            $("#modal-title").text(res.pred);
            if (bgColor == "#89a15d") {
                $("#modal-body").text("This review can be trusted. ");
            } else {
                $("#modal-body").text("This review content appears to be fake. ");
            }
        },
        error: function(res) {
            document.querySelector('.modal-header').style.backgroundColor = "#f57207";
            $("#modal-title").text(res.responseJSON.pred);
            $("#modal-body").text("Input content cannot be blank");
        }
    });
}

/**
 * clear entered content
 */
var clear_input = function() {
    console.log("reset");
    document.getElementById("text_input").value = "";
}